import numpy as np
from numpy import random
from numpy.linalg import norm
import os

def J_conjugate(A):
    """
    Conjugate the 3x3 matrix A by the diagonal matrix J=diag((-1, -1, 1)).

    :param A: A 3x3 matrix.
    :return: J*A*J
    """
    J = np.diag((-1, -1, 1))

    return J @ A @ J

def buildOuterProducts(n_img):
    # Build random third rows, ground truth vis (unit vectors)
    gt_vis = np.zeros((n_img, 3), dtype=np.float32)
    random.seed(42)
    for i in range(n_img):
        v = random.randn(3)
        gt_vis[i] = v / norm(v)

    # Find outer products viis and vijs for i<j
    nchoose2 = int(n_img * (n_img - 1) / 2)
    vijs = np.zeros((nchoose2, 3, 3))

    # All pairs (i,j) where i<j
    pairs = [(i, j) for i in range(n_img) for j in range(n_img) if i < j]

    for k, (i, j) in enumerate(pairs):
        vijs[k] = np.outer(gt_vis[i], gt_vis[j])

    # J-conjugate some of these outer products (every other element).
    vijs_conj = vijs.copy()
    vijs_conj[::2] = J_conjugate(vijs_conj[::2])
    fn = f"vijs_conj_n{n_img}.npy"
    np.save(fn, vijs_conj)
    os.chmod(fn, 0o777)

buildOuterProducts(5)
