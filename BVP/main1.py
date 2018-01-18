from bvp import finite_differences
from bvp import least_squares
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

def p(x):
    return 0.0

def q(x):
    return 1.0

def f(x):
    return 1.0

def exact_solution(x):
    return 1 - np.sin(x) - np.cos(x)


def basis_function(x, k, a, b, alpha0, alpha1, beta0, beta1, A, B):
    if k == 0:
        tmp = (alpha0 * B - beta0 * A) * 1.0 / (b - a) / alpha0 / beta0
        return tmp * x + A * 1.0 / alpha0 - a * tmp
    else:
        return x**(k - 1) * (x - a) * (x - b)


def least_squares_solution(x, c, a, b, alpha0, alpha1, beta0, beta1, A, B):
    res = basis_function(x, 0, a, b, alpha0, alpha1, beta0, beta1, A, B)
    for k in range(len(c)):
        res += c[k] * basis_function(x, k + 1, a, b, alpha0, alpha1, beta0, beta1, A, B)
    return res


a, b, n, alpha0, alpha1, beta0, beta1, A, B, basis_func_num, dot_num = \
    0.0, np.pi / 2, int(10), 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2, 500
x_numerical = np.linspace(a, b, n + 1)
x_exact = np.linspace(a, b, dot_num)
basis_func_coeffs = least_squares(basis_function, p, q, f, a, b, alpha0, alpha1, beta0, beta1, A, B, basis_func_num)
y_exact, y_numerical = \
    exact_solution(x_exact), finite_differences(p, q, f, a, b, alpha0, alpha1, beta0, beta1, A, B, n)
plt.figure(1, figsize=(15, 7.5))
plt.grid(True)
plt.plot(x_exact, y_exact, 'k-',
         x_numerical, y_numerical, 'b--',
         x_exact, least_squares_solution(x_exact, basis_func_coeffs, a, b, alpha0, alpha1, beta0, beta1, A, B), 'm--')
plt.plot(a, A, 'ro', b, B, 'ro')
for i in range(len(x_numerical)):
    print(exact_solution(x_numerical[i]), y_numerical[i], np.abs(exact_solution(x_numerical[i]) - y_numerical[i]))
# x = np.linspace(a, b, 100)
# plt.axis([a, b, 0, 5])
plt.show()
plt.close()
