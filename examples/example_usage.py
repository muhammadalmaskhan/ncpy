
from ncpy import bisection, newton_raphson, linear_least_squares, lagrange_interpolation, trapezoidal_rule, simpsons_rule
import numpy as np

# Example usage:
f = lambda x: x**3 - x - 2
df = lambda x: 3*x**2 - 1

print("Bisection root:", bisection(f, 1, 2))
print("Newton-Raphson root:", newton_raphson(f, df, 1.5))

x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 2, 1, 3])
a, b = linear_least_squares(x_data, y_data)
print(f"Linear fit: y = {a}x + {b}")

print("Lagrange interpolation at x=1.5:", lagrange_interpolation(x_data, y_data, 1.5))

print("Trapezoidal integration:", trapezoidal_rule(np.sin, 0, np.pi))
print("Simpson's integration:", simpsons_rule(np.sin, 0, np.pi))
