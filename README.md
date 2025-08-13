# ncpy

[![PyPI Version](https://img.shields.io/pypi/v/ncpy.svg)](https://pypi.org/project/ncpy/)
[![License](https://img.shields.io/pypi/l/ncpy.svg)](https://github.com/yourusername/ncpy/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/ncpy)](https://pepy.tech/project/ncpy)

**ncpy** — Numerical Computing in Python.

`ncpy` is a compact, educational Python library that implements common numerical methods for courses, assignments, and quick prototyping.  
Built on **NumPy** and (optionally) **SciPy**, it offers easy-to-use functions for:

- Root finding  
- Interpolation  
- Curve fitting / Approximation  
- Numerical integration  
- Numerical differentiation  
- Solving linear systems  

---

## Why use `ncpy`?

✅ **One package, many methods** — no need to import multiple libraries  
✅ **Lightweight & beginner-friendly** — great for teaching & learning numerical methods  
✅ **Educational** — functions are implemented clearly for understanding algorithms  
✅ **Fast enough** — powered by NumPy for efficiency  

---

## ✨ Features Overview

| Category               | Methods |
|------------------------|---------|
| **Root-finding**       | Bisection, Newton–Raphson, Secant, Fixed-point iteration |
| **Interpolation**      | Lagrange, Newton divided differences, Linear, Cubic spline, Neville’s method |
| **Approximation**      | Polynomial least squares, Exponential fit, Logarithmic fit |
| **Integration**        | Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg, Gaussian quadrature |
| **Differentiation**    | Forward, Backward, Central differences, Richardson extrapolation, Numerical gradient |
| **Linear Systems**     | Gaussian elimination, Gauss–Jordan, LU decomposition, Jacobi, Gauss–Seidel, Conjugate Gradient |

---
---
## Example

- Root finding - Newton Raphson
---
from ncpy import newton_raphson

f = lambda x: x**2 - 2
df = lambda x: 2*x

root = newton_raphson(f, df, x0=1.0)
print("Root:", root)  # ~1.4142

--- 
## 📦 Installation

```bash
pip install ncpy

 
