import numpy as np

def poly_least_squares(x, y, deg):
    """Polynomial least squares fit; returns coefficients highest power first (like numpy.polyfit)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    V = np.vander(x, N=deg+1, increasing=False)
    coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
    return coeffs

def linear_least_squares(x, y):
    """Linear fit y ~ a*x + b. Returns (a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def exp_fit(x, y):
    """Fit y = a * exp(b*x). Requires y > 0. Returns (a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("y must be positive for exponential fit.")
    Y = np.log(y)
    b, loga = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, Y, rcond=None)[0]
    a = np.exp(loga)
    return a, b

def log_fit(x, y):
    """Fit y = a + b*ln(x). Requires x > 0. Returns (a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0):
        raise ValueError("x must be positive for logarithmic fit.")
    X = np.log(x)
    b, a = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, y, rcond=None)[0]
    return a, b
