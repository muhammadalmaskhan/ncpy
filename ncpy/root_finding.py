import numpy as np

def bisection(f, a, b, tol=1e-8, max_iter=100):
    """Bisection method for f(x)=0 on [a,b] where f(a) and f(b) have opposite signs."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    for _ in range(max_iter):
        c = 0.5*(a+b)
        fc = f(c)
        if abs(fc) < tol or 0.5*(b-a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5*(a+b)

def newton_raphson(f, df, x0, tol=1e-8, max_iter=100):
    """Newton–Raphson method for f(x)=0 given derivative df."""
    x = float(x0)
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        if abs(fx) < tol:
            return x
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero at x = {}".format(x))
        x -= fx/dfx
    return x

def secant(f, x0, x1, tol=1e-8, max_iter=100):
    """Secant method for f(x)=0 without requiring derivative."""
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        denom = (f1 - f0)
        if abs(denom) < np.finfo(float).eps:
            return x1
        x2 = x1 - f1*(x1-x0)/denom
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)
    return x1

def fixed_point(g, x0, tol=1e-8, max_iter=100):
    """Fixed-point iteration x = g(x)."""
    x = float(x0)
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x
