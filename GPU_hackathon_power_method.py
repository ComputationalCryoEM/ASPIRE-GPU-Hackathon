CUDA = False

try:
    import cupy as np
    from cupy.linalg import norm
    import numpy
    from numpy import triu_indices
    CUDA = True
except ImportError:
    print("Running CPU version")
    import numpy as np
    from numpy.linalg import norm
    from numpy import triu_indices

if CUDA:
    try:
        import cupy_backends
        cupy_backends.cuda.api.runtime.getDevice()
        print("CuPy enabled")
    except cupy_backends.cuda.api.runtime.CUDARuntimeError:
        print("Running CPU version")
        import numpy as np
        from numpy.linalg import norm
        from numpy import triu_indices

from numpy import random
import sys
import itertools

#####################
# Global variables #
#####################
# J conjugating matrix is equivalent to elementwise multiplication of J_mask.
# We replace J @ A @ J with J_mask * A, where J = diag(-1, -1, 1).
# This reduces the number matrix multiplication kernels on the GPU.
J_mask = np.array([
    [1, 1, -1],
    [1, 1, -1],
    [-1, -1, 1],
])


#####################
# Utility Functions #
#####################

def pairs_to_linear(n, i, j):
    """
    Converts from all_pairs indexing (i, j), where i<j, to linear indexing.
    ie. (0, 1) --> 0 and (n-2, n-1) --> n * (n - 1)/2 - 1
    """

    linear_index = n*(n-1)//2 - (n-i)*(n-i-1)//2 + j - i - 1

    return linear_index


def all_triplets_batch(ijks, batch_size):
    """
    Given a set of all triplets we generate the triplets in batches.

    :param ijks: The set of all_triplets [ij, jk, ik]
    :param batch_size: The number of triplets to generate per batch.

    :yield: Batches of triplets of size batch_size x 3
    """
    for i in range(0, ijks.shape[0], batch_size):
        yield ijks[i:i+batch_size, :]

def all_triplets(n):
    """
    We construct all 3-tuples (i,j,k) where i<j<k. We then shuffle the tuples
    to prevent overloading individual histogram bins in sync_times_v. We return
    all triplets of pairs [(i, j), (j, k), (i, k)], where each pair is represented
    with the linear index performed by pairs_to_linear.

    :param n: The number of items to be indexed.
    :returns: All triplet [(i, j), (j, k), (i, k)], where i<j<k and each pair
    is converted to a linear index.
    """
    jk_vals = np.array(triu_indices(n, k=1)).T
    i_vals = np.tile(np.arange(n), (len(jk_vals), 1)).T.flatten()
    jk_vals = np.tile(jk_vals, (n,1))
    ijks = np.hstack((i_vals[:, None], jk_vals))
    ijks = ijks[ijks[:, 0] < ijks[:, 1], :]
    np.random.shuffle(ijks)
    return pairs_to_linear(n, 
                ijks[:, [0, 1, 0]],
                ijks[:, [1, 2, 2]],
                )
    
    
################
# Power Method #
################

def signs_times_v(vijs, vec, conjugate, edge_signs, triplets_iter):
    """
    Multiplication of the J-synchronization matrix by a candidate eigenvector.

    The J-synchronization matrix is a matrix representation of the handedness graph, Gamma, whose set of
    nodes consists of the estimates vijs and whose set of edges consists of the undirected edges between
    all triplets of estimates vij, vjk, and vik, where i<j<k. The weight of an edge is set to +1 if its
    incident nodes agree in handednes and -1 if not.

    The J-synchronization matrix is of size (n-choose-2)x(n-choose-2), where each entry corresponds to
    the relative handedness of vij and vjk. The entry (ij, jk), where ij and jk are retrieved from the
    all_pairs indexing, is 1 if vij and vjk are of the same handedness and -1 if not. All other entries
    (ij, kl) hold a zero.

    Due to the large size of the J-synchronization matrix we construct it on the fly as follows.
    For each triplet of outer products vij, vjk, and vik, the associated elements of the J-synchronization
    matrix are populated with +1 or -1 and multiplied by the corresponding elements of
    the current candidate eigenvector supplied by the power method. The new candidate eigenvector
    is updated for each triplet.

    :param vijs: (n-choose-2)x3x3 array, where each 3x3 slice holds the outer product of vi and vj.

    :param vec: The current candidate eigenvector of length n-choose-2 from the power method.

    :return: New candidate eigenvector of length n-choose-2. The product of the J-sync matrix and vec.
    """
    # For each triplet of nodes we apply the 4 configurations of conjugation and determine the
    # relative handedness based on the condition that vij @ vjk - vik = 0 for synchronized nodes.
    # We then construct the corresponding entries of the J-synchronization matrix with 'edge_signs'
    # corresponding to the conjugation configuration producing the smallest residual for the above
    # condition. Finally, we the multiply the 'edge_signs' by the cooresponding entries of 'vec'.
    v = vijs
    new_vec = np.zeros_like(vec)
    bins = np.arange(len(new_vec) + 1)
    for ijk in triplets_iter:
        # Retrieve all vijs for the batch of triplets indices ijk.
        # Vijk has shape (batch_size)x3x3x3.
        Vijk = v[ijk]

        # J @ Vijk @ J
        Vijk_J = J_mask[None, None, ...] * Vijk


        conjugated_pairs = np.where(
            conjugate[np.newaxis, ..., np.newaxis, np.newaxis],
            np.expand_dims(Vijk_J, axis=1),
            np.expand_dims(Vijk, axis=1),
        )

        # flatten the pairs array to limit number of kernel launches
        old_shape = conjugated_pairs.shape
        conjugated_pairs = conjugated_pairs.reshape((-1, 3, 3, 3))
        residual = norm(
            conjugated_pairs[:, 0, ...] @  # x
            conjugated_pairs[:, 1, ...] -  # y
            conjugated_pairs[:, 2, ...],  # z
            axis=(1, 2),
        )
        # convert back to original size to maintain sanity
        residual = residual.reshape(old_shape[0:2])

        min_residual = np.argmin(residual, axis=1)

        # Assign edge weights
        S = edge_signs[min_residual]

        # Update multiplication of signs times vec
        new_ele = S[:, [0, 0, 0]] * vec[ijk[:, [1, 0, 0]]] + S[:, [2, 1, 1]] * vec[ijk[:, [2, 2, 1]]]
        new_vec += np.histogram(ijk[:], weights=new_ele[:], bins=bins)[0]
    return new_vec

def J_sync_power_method(vijs, batch_size):
    """
    Calculate the leading eigenvector of the J-synchronization matrix
    using the power method.

    As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
    use the power method to compute the eigenvalues and eigenvectors,
    while constructing the matrix on-the-fly.

    :param vijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

    :return: An array of length n-choose-2 consisting of 1 or -1, where the sign of the
    i'th entry indicates whether the i'th relative orientation matrix will be J-conjugated.
    """

    # Set power method tolerance and maximum iterations.
    epsilon = 1e-3
    max_iters = 1000
    random.seed(42)

    # Initialize candidate eigenvectors
    n_vijs = vijs.shape[0]
    vec = np.asarray(random.randn(n_vijs))
    vec /= norm(vec)
    residual = 1

    # Initialize entries for the J-sync matrix:
    # There are 4 possible configurations of relative handedness for each triplet (vij, vjk, vik).
    # 'conjugate' expresses which node of the triplet must be conjugated (True) to achieve synchronization.
    conjugate = np.array(
        [[False, False, False],
         [True, False, False],
         [False, True, False],
         [False, False, True],],
    dtype=bool)

    # 'edges' corresponds to whether conjugation agrees between the pairs (vij, vjk), (vjk, vik),
    # and (vik, vij). True if the pairs are in agreement, False otherwise.
    edges = conjugate == conjugate[:, [1, 2, 0]]

    # The corresponding entries in the J-synchronization matrix are +1 if the pair of nodes agree, -1 if not.
    edge_signs = np.where(edges, 1, -1)

    # number of images (inverting n choose 2)
    n_img = int((1+np.sqrt(1+8*len(vijs)))/2)
    
    # generate shuffled indices for triplets
    triplets = all_triplets(n_img)
    
    # initialize vec_new to prevent blocking garbage collection of vec
    vec_new = vec
    # Power method iterations
    for itr in range(max_iters):
        triplets_iter = all_triplets_batch(triplets, batch_size)
        vec_new = signs_times_v(vijs, vec, conjugate, edge_signs, triplets_iter)
        vec_new /= norm(vec_new)
        residual = norm(vec_new - vec)
        vec = vec_new
        if residual < epsilon:
            print(f'Converged after {itr} iterations of the power method.')
            break
    else:
        print('max iterations')

    # We need only the signs of the eigenvector
    J_sync = np.sign(vec_new)

    return J_sync

# problem size to load
n = int(sys.argv[1])

batch_size = (n * (n-1)) // 2

# load input data
vijs = np.load(f"vijs_conj_n{n}.npy")

J_sync_vec = J_sync_power_method(vijs, batch_size)

# save to disk
np.save(f"J_sync_vec_n{n}.npy", J_sync_vec)
