import numpy as np

def lagrange_interpolation(x_points, y_points, x):
    """Evaluate Lagrange interpolating polynomial at x."""
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x_points)
    result = np.zeros_like(x, dtype=float)
    for i in range(n):
        li = np.ones_like(x, dtype=float)
        for j in range(n):
            if i != j:
                li *= (x - x_points[j])/(x_points[i]-x_points[j])
        result += y_points[i]*li
    return result

def newton_divided_differences_coeffs(x, y):
    """Return Newton divided differences coefficients for nodes x and values y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    coef = y.astype(float).copy()
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1])/(x[j:n] - x[0:n-j])
    return coef

def newton_interpolate(x_nodes, y_nodes, x):
    """Evaluate Newton polynomial (using divided differences) at x."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    coef = newton_divided_differences_coeffs(x_nodes, y_nodes)
    x = np.asarray(x, dtype=float)
    n = len(coef)
    p = np.zeros_like(x, dtype=float) + coef[-1]
    for k in range(n-2, -1, -1):
        p = coef[k] + (x - x_nodes[k]) * p
    return p

def linear_interpolation(x_points, y_points, x):
    """Piecewise linear interpolation evaluated at x (x_points must be sorted)."""
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x = np.asarray(x, dtype=float)
    idxs = np.searchsorted(x_points, x) - 1
    idxs = np.clip(idxs, 0, len(x_points)-2)
    x0, x1 = x_points[idxs], x_points[idxs+1]
    y0, y1 = y_points[idxs], y_points[idxs+1]
    t = (x - x0) / (x1 - x0)
    return y0 + t*(y1 - y0)

def natural_cubic_spline_coeffs(x, y):
    """Compute natural cubic spline second derivatives at knots and per-interval coefficients."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    al = np.zeros(n)
    al[1:-1] = 3*( (y[2:] - y[1:-1])/h[1:] - (y[1:-1] - y[:-2])/h[:-1] )
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (al[i] - h[i-1]*z[i-1]) / l[i]
    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(2*c[j] + c[j+1])/3
        d[j] = (c[j+1] - c[j]) / (3*h[j])
    a = y[:-1]
    return a, b, c[:-1], d

def cubic_spline(x_points, y_points, x):
    """Evaluate natural cubic spline at x."""
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    a,b,c,d = natural_cubic_spline_coeffs(x_points, y_points)
    x = np.asarray(x, dtype=float)
    idxs = np.searchsorted(x_points, x) - 1
    idxs = np.clip(idxs, 0, len(x_points)-2)
    dx = x - x_points[idxs]
    return a[idxs] + b[idxs]*dx + c[idxs]*(dx**2) + d[idxs]*(dx**3)

def neville(x_points, y_points, x):
    """Neville's algorithm evaluated at scalar x."""
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    x = float(x)
    n = len(x_points)
    Q = y_points.astype(float).copy()
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            Q[i] = ((x - x_points[i-j])*Q[i] - (x - x_points[i])*Q[i-1]) / (x_points[i] - x_points[i-j])
    return Q[-1]
