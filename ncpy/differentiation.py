import numpy as np

def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=1e-5):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)

def richardson_extrapolation(f, x, h=1e-2):
    D1 = central_diff(f, x, h)
    D2 = central_diff(f, x, h/2)
    return D2 + (D2 - D1)/3

def numerical_gradient(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = 1.0
        grad[i] = (f(x + h*e) - f(x - h*e)) / (2*h)
    return grad
