# ncpy
<p align="center">
  <a href="https://pypi.org/project/ncpy/"><img src="https://img.shields.io/pypi/v/ncpy.svg" alt="PyPI Version"></a>
  <a href="https://github.com/muhammadalmaskhan/ncpy/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/ncpy.svg" alt="License"></a>
  <a href="https://pepy.tech/project/ncpy"><img src="https://static.pepy.tech/badge/ncpy" alt="Downloads"></a>
  <a href="https://pypi.org/project/ncpy/"><img src="https://img.shields.io/pypi/pyversions/ncpy.svg" alt="Python Versions"></a>
  <a href="https://github.com/muhammadalmaskhan/ncpy/stargazers"><img src="https://img.shields.io/github/stars/muhammadalmaskhan/ncpy?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/muhammadalmaskhan/ncpy/commits/main"><img src="https://img.shields.io/github/last-commit/muhammadalmaskhan/ncpy.svg" alt="Last Commit"></a>
  <a href="https://github.com/muhammadalmaskhan/ncpy/issues"><img src="https://img.shields.io/github/issues/muhammadalmaskhan/ncpy.svg" alt="Open Issues"></a>
</p>

**ncpy** — Numerical Computing in Python.

`ncpy` is a compact, educational Python library that implements common numerical methods for quick prototyping and expermentations.  
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
| **Best Approximations**| Least Squares, Gram–Schmidt Orthonormalization |
---
## Examples
---

- Root finding - Newton Raphson
```bash
from ncpy import newton_raphson

f = lambda x: x**2 - 2
df = lambda x: 2*x

root = newton_raphson(f, df, x0=1.0)
print("Root:", root)  # ~1.4142

```
- Interpolation — Lagrange

```bash
from ncpy import lagrange_interpolation

x_points = [0, 1, 2]
y_points = [1, 3, 2]
print(lagrange_interpolation(x_points, y_points, 1.5))

```
- Numerical Integration — Simpson's 1/3 Rule

```bash
from ncpy import simpson13
import math

area = simpson13(math.sin, 0, math.pi, n=100)
print(area)  # ~2.0
```

--- 
## 📦 Installation

```bash
pip install ncpy
```
---


 
## 📍 Visitors
<a href="https://clustrmaps.com/site/1c7ls" title="ClustrMaps"> <img src="https://www.clustrmaps.com/map_v2.png?d=GL51q-0KRQlIUM81XiBM7GahOkVrKk88t7NjiXWTcaI&cl=ffffff" /> </a> ```