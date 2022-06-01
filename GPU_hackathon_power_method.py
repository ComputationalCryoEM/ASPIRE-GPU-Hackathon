import numpy as np
from numpy.linalg import norm
from numpy import random

#####################
# Global variables #
#####################
J = np.diag((-1, -1, 1))

#####################
# Utility Functions #
#####################

def all_pairs(n):
    """
    All pairs indexing (i,j) for i<j.

    :param n: The number of items to be indexed.
    :return: All n-choose-2 pairs (i,j), i<j.
    """
    pairs = [(i, j) for i in range(n) for j in range(n) if i < j]

    return pairs

def pairs_to_linear(n, i, j):
    """
    Converts from all_pairs indexing (i, j), where i<j, to linear indexing.
    ie. (0, 1) --> 0 and (n-2, n-1) --> n * (n - 1)/2 - 1
    """

    linear_index = n*(n-1)//2 - (n-i)*((n-i)-1)//2 + j - i - 1

    return linear_index

def all_triplets(n):
    """
    All 3-tuples (i,j,k) where i<j<k.

    :param n: The number of items to be indexed.
    :returns: All 3-tuples (i,j,k), i<j<k.
    """
    triplets = (
        (i, j, k) for i in range(n)
        for j in range(i+1, n)
        for k in range(j+1, n)
    )

    return triplets


def J_conjugate(A):
    """
    Conjugate the 3x3 matrix A by the diagonal matrix J=diag((-1, -1, 1)).

    :param A: A 3x3 matrix.
    :return: J*A*J
    """
    return J @ A @ J


################
# Power Method #
################

def signs_times_v(vijs, vec, conjugate, edge_signs):
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
    
    # All pairs (i,j) and triplets (i,j,k) where i<j<k
    n_img = int((1+np.sqrt(1+8*len(vijs)))/2)  # Extract number of images from vijs.
    triplets = all_triplets(n_img)

    # For each triplet of nodes we apply the 4 configurations of conjugation and determine the
    # relative handedness based on the condition that vij @ vjk - vik = 0 for synchronized nodes.
    # We then construct the corresponding entries of the J-synchronization matrix with 'edge_signs'
    # corresponding to the conjugation configuration producing the smallest residual for the above
    # condition. Finally, we the multiply the 'edge_signs' by the cooresponding entries of 'vec'.
    v = vijs
    new_vec = np.zeros_like(vec)
    for (i, j, k) in triplets:
        ijk = pairs_to_linear(n_img, np.array([i, j, i]), np.array([j, k, k]))
        Vijk = v[ijk]
        Vijk_J = J_conjugate(Vijk)

        conjugated_pairs = np.where(
            conjugate[..., np.newaxis, np.newaxis],
            [Vijk_J],
            [Vijk],
        )

        residual = norm(
            conjugated_pairs[:, 0, ...] @  # x
            conjugated_pairs[:, 1, ...] -  # y
            conjugated_pairs[:, 2, ...],  # z
            axis=(1, 2),
        )

        min_residual = np.argmin(residual)

        # Assign edge weights
        S = edge_signs[min_residual]

        # Update multiplication of signs times vec
        new_vec[ijk] += S[0] * vec[ijk[[1, 0, 0]]] + S[[2, 1, 1]] * vec[ijk[[2, 2, 1]]]

    return new_vec

def J_sync_power_method(vijs):
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
    vec = random.randn(n_vijs)
    vec /= norm(vec)
    residual = 1

    # Initialize entries for the J-sync matrix:
    # There are 4 possible configurations of relative handedness for each triplet (vij, vjk, vik).
    # 'conjugate' expresses which node of the triplet must be conjugated (True) to achieve synchronization.
    conjugate = np.empty((4, 3), bool)
    conjugate[0] = [False, False, False]
    conjugate[1] = [True, False, False]
    conjugate[2] = [False, True, False]
    conjugate[3] = [False, False, True]

    # 'edges' corresponds to whether conjugation agrees between the pairs (vij, vjk), (vjk, vik),
    # and (vik, vij). True if the pairs are in agreement, False otherwise.
    edges = np.empty((4, 3), bool)
    edges[:, 0] = conjugate[:, 0] == conjugate[:, 1]
    edges[:, 1] = conjugate[:, 1] == conjugate[:, 2]
    edges[:, 2] = conjugate[:, 2] == conjugate[:, 0]

    # The corresponding entries in the J-synchronization matrix are +1 if the pair of nodes agree, -1 if not.
    edge_signs = np.where(edges, 1, -1)

    # Power method iterations
    for itr in range(max_iters):
        vec_new = signs_times_v(vijs, vec, conjugate, edge_signs)
        vec_new /= norm(vec_new)
        residual = norm(vec_new - vec)
        vec = vec_new
        if residual < epsilon:
            print(f'Converged after {itr} iterations of the power method.')
            break
    else:
        print('max iterations')

    # We need only the signs of the eigenvector
    J_sync = np.sign(vec)

    return J_sync

vijs = np.load("vijs_conj_n50.npy")

J_sync_vec = J_sync_power_method(vijs)

np.save("J_sync_vec_n50.npy", J_sync_vec)
