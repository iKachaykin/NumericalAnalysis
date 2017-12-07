import numpy as np
from scipy.misc import derivative


def simple_iteration(f, a, b, dot_num=1000, eps=1e-12):
    der_vals = derivative(lambda x: f(x), np.linspace(a, b, dot_num), dx=1e-6)
    while np.abs(np.sign(der_vals).sum()) < der_vals.size:
        if f(a) * f((a + b) / 2.0) < 0:
            b = (a + b) / 2.0
        else:
            a = (a + b) / 2.0
        der_vals = derivative(lambda x: f(x), np.linspace(a, b, dot_num), dx=1e-6)
    abs_deriv_max, abs_deriv_min = np.abs(der_vals).max(), np.abs(der_vals).min()
    k = 2 / (abs_deriv_max + abs_deriv_min)
    if np.sign(der_vals)[0] < 0:
        k *= -1
    x = a
    while np.abs(f(x)) > eps:
        x = x - k * f(x)
    return x
