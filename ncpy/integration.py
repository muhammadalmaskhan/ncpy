import numpy as np

def trapezoidal(f, a, b, n=100):
    h = (b-a)/n
    s = 0.5*(f(a)+f(b))
    for i in range(1, n):
        s += f(a + i*h)
    return s*h

def simpson13(f, a, b, n=100):
    if n % 2 == 1:
        n += 1
    h = (b-a)/n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 == 1 else 2)*f(a + i*h)
    return s*h/3

def simpson38(f, a, b, n=99):
    r = n % 3
    if r != 0:
        n += (3 - r)
    h = (b-a)/n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (3 if i % 3 != 0 else 2)*f(a + i*h)
    return 3*h*s/8

def romberg(f, a, b, max_level=5):
    R = np.zeros((max_level, max_level), dtype=float)
    for k in range(max_level):
        n = 2**k
        R[k,0] = trapezoidal(f, a, b, n)
        for j in range(1, k+1):
            R[k,j] = R[k,j-1] + (R[k,j-1] - R[k-1,j-1])/(4**j - 1)
    return R[max_level-1, max_level-1]

def gauss_legendre(f, a, b, n=5):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    t = 0.5*(x+1)*(b-a) + a
    return 0.5*(b-a) * np.sum(w * f(t))
