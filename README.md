# ncpy

[![PyPI Version](https://img.shields.io/pypi/v/ncpy.svg)](https://pypi.org/project/ncpy/)
[![License](https://img.shields.io/pypi/l/ncpy.svg)](https://github.com/yourusername/ncpy/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/ncpy)](https://pepy.tech/project/ncpy)

**ncpy** — Numerical Computing in Python.

`ncpy` is a compact, educational Python library that implements common numerical methods for courses and quick prototyping.  
It is built on top of **NumPy** and (optionally) **SciPy**, and provides easy-to-use functions for root finding, interpolation, approximation, integration, differentiation, and solving linear systems.

---

## ✨ Features

- **Root-finding**: Bisection, Newton–Raphson, Secant, Fixed-point iteration  
- **Interpolation**: Lagrange, Newton divided differences, Linear, Cubic spline, Neville’s method  
- **Approximation / Curve fitting**: Polynomial least squares, Exponential fit, Logarithmic fit  
- **Numerical integration**: Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg, Gaussian quadrature  
- **Numerical differentiation**: Forward, Backward, Central differences, Richardson extrapolation, Numerical gradient  
- **Linear systems**: Gaussian elimination, Gauss–Jordan, LU decomposition, Jacobi, Gauss–Seidel, Conjugate Gradient

---
## Quick Example
from ncpy import roots

### Example: Newton–Raphson method
f = lambda x: x**2 - 2
df = lambda x: 2*x

root = roots.newton_raphson(f, df, x0=1.0)
print("Root:", root)

---

## 📦 Installation

```bash
pip install ncpy
