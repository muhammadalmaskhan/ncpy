import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for k in range(n-1):
        i_max = k + np.argmax(np.abs(A[k:,k]))
        if A[i_max, k] == 0:
            raise ValueError("Matrix is singular.")
        if i_max != k:
            A[[k,i_max]] = A[[i_max,k]]
            b[[k,i_max]] = b[[i_max,k]]
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i, k:] -= factor*A[k, k:]
            b[i] -= factor*b[k]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:]))/A[i,i]
    return x

def gauss_jordan(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for i in range(n):
        piv = i + np.argmax(np.abs(A[i:,i]))
        if A[piv,i] == 0:
            raise ValueError("Matrix is singular.")
        A[[i,piv]] = A[[piv,i]]
        b[[i,piv]] = b[[piv,i]]
        factor = A[i,i]
        A[i,:] /= factor
        b[i] /= factor
        for j in range(n):
            if j != i:
                factor = A[j,i]
                A[j,:] -= factor*A[i,:]
                b[j] -= factor*b[i]
    return b

def lu_decomposition(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(n):
        for k in range(i, n):
            U[i,k] = A[i,k] - np.dot(L[i,:i], U[:i,k])
        L[i,i] = 1.0
        for k in range(i+1, n):
            if U[i,i] == 0:
                raise ValueError("Zero pivot encountered in LU.")
            L[k,i] = (A[k,i] - np.dot(L[k,:i], U[:i,i])) / U[i,i]
    return L, U

def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x))/D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i,:i], x[:i])
            s2 = np.dot(A[i,i+1:], x[i+1:])
            x[i] = (b[i] - s1 - s2)/A[i,i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x
    return x

def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=1000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    r = b - A.dot(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x
