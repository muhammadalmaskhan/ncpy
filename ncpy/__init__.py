
"""
Numerical Methods Package
"""
from .root_finding import *
from .interpolation import *
from .integration import *
from .differentiation import *
from .linear_systems import *

import numpy as np
from scipy import optimize

# --- Root Finding Methods ---

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """Bisection method for root finding."""
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Function must have opposite signs at a and b")
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """Newton-Raphson method."""
    x = x0
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        if abs(fx) < tol:
            return x
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero")
        x -= fx / dfx
    return x

# --- Least Squares ---
def linear_least_squares(x, y):
    """Linear least squares fitting y = ax + b"""
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

# --- Interpolation ---
def lagrange_interpolation(x_points, y_points, x):
    """Lagrange interpolation for given data points."""
    total = 0
    n = len(x_points)
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        total += term
    return total

# --- Numerical Integration ---
def trapezoidal_rule(f, a, b, n=1000):
    """Trapezoidal rule for numerical integration."""
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

def simpsons_rule(f, a, b, n=1000):
    """Simpson's rule for numerical integration."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += f(a + i * h) * (4 if i % 2 == 1 else 2)
    return s * h / 3
