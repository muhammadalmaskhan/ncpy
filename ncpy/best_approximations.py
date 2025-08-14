"""
Best Approximations Module for ncpy
-----------------------------------
Includes:
- least_squares: Solve linear least squares problems
- gram_schmidt: Perform Gram-Schmidt orthonormalization
"""

import numpy as np
import numpy.linalg as npl

def least_squares(A, b):
    """
    Solve the linear least squares problem: minimize ||b - Ax||_2.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Coefficient matrix.
    b : array_like, shape (m,) or (m, 1)
        Right-hand side vector.

    Returns
    -------
    x : ndarray, shape (n, 1)
        Solution vector.
    residual : float
        Sum of squared residuals.
    rank : int
        Rank of the matrix A.
    singular_values : ndarray
        Singular values of A.
    """
    x, residual, rank, s = npl.lstsq(A, b, rcond=None)
    return x, residual, rank, s

def gram_schmidt(X):
    """
    Perform the Modified Gram-Schmidt orthonormalization.

    Parameters
    ----------
    X : array_like, shape (m, n)
        Input matrix with column vectors to be orthonormalized.

    Returns
    -------
    E : ndarray, shape (m, n)
        Orthonormalized column vectors.
    """
    X = np.array(X, dtype=float)
    m, n = X.shape
    E = np.zeros((m, n))
    # First vector normalization
    E[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
    for i in range(1, n):
        E[:, i] = X[:, i]
        for j in range(i):
            E[:, i] -= np.dot(E[:, j], E[:, i]) * E[:, j]
        E[:, i] /= np.linalg.norm(E[:, i])
    return E
