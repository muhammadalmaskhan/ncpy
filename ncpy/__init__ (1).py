"""
nCPy - Numerical Computing in Python
------------------------------------
A compact Python library for common numerical methods.
"""

# Root-finding methods
# (Assuming you already have these in your package)
from .roots import bisection, newton_raphson, secant, fixed_point

# Interpolation methods
from .interpolation import lagrange_interpolation, newton_divided_diff, linear_interpolation, cubic_spline, neville

# Approximation methods
from .approximation import polynomial_least_squares, exponential_fit, logarithmic_fit

# Integration methods
from .integration import trapezoidal, simpson13, simpson38, romberg, gaussian_quadrature

# Differentiation methods
from .differentiation import forward_difference, backward_difference, central_difference, richardson_extrapolation, numerical_gradient

# Linear system solvers
from .linear_systems import gaussian_elimination, gauss_jordan, lu_decomposition, jacobi, gauss_seidel, conjugate_gradient

# Best approximations
from .best_approximations import least_squares, gram_schmidt
