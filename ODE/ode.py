import numpy as np
import numpy.polynomial.legendre as leg
from scipy.misc import derivative


def Euler(f, y_0, a, b, n=10):
    if a >= b:
        a, b = b, a
    x_vals, y_res, h = np.linspace(a, b, n + 1), np.empty(n + 1), (b - a) * 1.0 / n
    y_res[0] = y_0
    for i in range(1, n + 1):
        y_res[i] = y_res[i - 1] + h * f(x_vals[i - 1], y_res[i - 1])
    return [np.array(x_vals), np.array(y_res), len(x_vals)]


def Runge_Kutta4_auto(f, y_0, a, b, h0=0.01, eps=0.0001, eps_b=0.01):
    hi, x_vals, y_res, i, method_order, err_temp = h0, [a], [y_0], 1, 4, 0
    stop_integrate = False
    while b - x_vals[i - 1] >= eps_b and not stop_integrate:
        stop_finding_hi = False
        while not stop_finding_hi:
            y_hi = y_res[i - 1] + find_Runge_Kutta4_delt(f, x_vals[i - 1], y_res[i - 1], hi)
            y_hi_div2_tmp = y_res[i - 1] + find_Runge_Kutta4_delt(f, x_vals[i - 1], y_res[i - 1], hi / 2.0)
            y_hi_div2 = y_hi_div2_tmp + find_Runge_Kutta4_delt(f, x_vals[i - 1] + hi / 2.0, y_hi_div2_tmp, hi / 2.0)
            err_temp = np.abs((y_hi_div2 - y_hi) / (2 ** method_order - 1))
            if err_temp >= eps:
                hi /= 2.0
            else:
                if b - (x_vals[i - 1] + hi) < eps_b:
                    stop_integrate = True
                    break
                x_vals.append(x_vals[i - 1] + hi)
                y_res.append(y_hi_div2)
                stop_finding_hi = True
                i += 1
        if err_temp < eps / (2 ** method_order):
            hi *= 2
    x_vals.append(b)
    y_res.append(y_res[i - 1] + find_Runge_Kutta4_delt(f, x_vals[i - 1], y_res[i - 1], b - x_vals[i - 1]))
    return [np.array(x_vals), np.array(y_res), len(x_vals) - 1]


def Runge_Kutta4_const(f, y_0, a, b, n=10):
    if a >= b:
        a, b = b, a
    x_vals, y_res, h = np.linspace(a, b, n + 1), np.empty(n + 1), (b - a) * 1.0 / n
    y_res[0] = y_0
    for i in range(1, n + 1):
        y_res[i] = y_res[i - 1] + find_Runge_Kutta4_delt(f, x_vals[i - 1], y_res[i - 1], h)
    return [np.array(x_vals), np.array(y_res), len(x_vals)]


def find_Runge_Kutta4_delt(f, xi, yi, h):
    k1 = h * f(xi, yi)
    k2 = h * f(xi + h / 2.0, yi + k1 / 2.0)
    k3 = h * f(xi + h / 2.0, yi + k2 / 2.0)
    k4 = h * f(xi + h, yi + k3)
    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def Adams_interp(f, y_0, a, b, n=10, init_val_num=5):
    if a >= b:
        a, b = b, a
    if init_val_num >= n:
        raise "Number of initial values must be lower than number of all values!"
    x_vals, h, y_res, hyp_diff = np.linspace(a, b, n + 1), (b - a) * 1.0 / n, np.empty(n + 1), 1
    y_res[0] = y_0
    for i in range(1, init_val_num + 1):
        y_res[i] = y_res[i - 1] + find_Runge_Kutta4_delt(f, x_vals[i - 1], y_res[i - 1], h)
    for i in range(init_val_num + 1, n + 1):
        y_res[i] = simple_iteration(lambda yi: yi - y_res[i - 1] - h *
            np.dot(all_Ak(init_val_num)[init_val_num + 1:0:-1], f(x_vals[i - init_val_num - 1:i], y_res[i - init_val_num - 1:i])) -
            h * Ak(-1, init_val_num) * f(x_vals[i], yi), y_res[i - 1], y_res[i - 1] + hyp_diff)
    return [np.array(x_vals), np.array(y_res), len(x_vals) - 1]


def Gaussian(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    if N > 100 or N < 0:
        N = 10
    results = []
    legcoeffs = np.where(np.arange(N + 2) == N + 1, 1, 0)
    x_vals = leg.legroots(legcoeffs)
    weight_vals = 2 / (1 - x_vals**2) / (leg.legval(x_vals, leg.legder(legcoeffs)) ** 2)
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            weight = -1
            left, right = right, left
        result = (right - left) / 2.0 * integrand((right - left) / 2.0 * x_vals + (right + left) / 2.0).dot(weight_vals)
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def tmp_f(x, k, n):
    res = np.ones_like(x)
    for i in range(-1, n + 1):
        if i != k:
            res *= x + i
    return res


def Ak(k, n):
    res = (-1) ** (k + 1)
    for i in range(1, k + 2):
        res /= i
    for i in range(1, n - k + 1):
        res /= i
    return res * Gaussian(lambda x: tmp_f(x, k, n), 0, 1)


def all_Ak(n):
    res = np.empty(n + 2)
    for i in range(n + 2):
        res[i] = Ak(i - 1, n)
    return res


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