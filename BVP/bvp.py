import numpy as np
import numpy.polynomial.legendre as leg
import numpy.linalg as linalg
from scipy.misc import derivative

def L(y, x, p, q, k, a, b, alpha0, alpha1, beta0, beta1, A, B):
    return derivative(y, x, n=2, dx=1e-6, args=tuple([k, a, b, alpha0, alpha1, beta0, beta1, A, B])) + \
           p(x) * derivative(y, x, dx=1e-6, args=tuple([k, a, b, alpha0, alpha1, beta0, beta1, A, B])) + \
           q(x) * y(x, k, a, b, alpha0, alpha1, beta0, beta1, A, B)


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
        result = (right - left) / 2 * integrand((right - left) / 2 * x_vals + (right + left) / 2).dot(weight_vals)
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def finite_differences(p, q, f, a, b, alpha0, alpha1, beta0, beta1, A, B, n=10):
    if np.abs(np.abs(alpha1) + np.abs(alpha0)) < 1e-8 or np.abs(np.abs(beta1) + np.abs(beta0)) < 1e-8:
        raise "Argument exception: \"alpha\" or \"beta\" arguments were invalid!"
    system_matrix, h, x_vals, free_coeffs = \
        np.zeros((n+1, n+1)), (b - a) * 1.0 / n, np.linspace(a, b, n + 1), np.empty(n+1)
    k1, k2, F0, Fn, Ai, Bi, Ci, Fi = alpha1 * 1.0 / (alpha1 - h * alpha0), beta1 * 1.0 / (beta1 + h * beta0), \
                     h * 1.0 * A / (alpha1 - h * alpha0), -h * 1.0 * B / (beta1 + h * beta0), 0.0, 0.0, 0.0, 0.0
    system_matrix[0][0] = -1.0
    system_matrix[0][1] = k1
    system_matrix[n][n - 1] = k2
    system_matrix[n][n] = -1.0
    free_coeffs[0] = F0
    free_coeffs[n] = Fn
    for i in range(1, n):
        system_matrix[i][i - 1] = 1 - h / 2 * p(x_vals[i])
        system_matrix[i][i] = -2 + h ** 2 * q(x_vals[i])
        system_matrix[i][i + 1] = 1 + h / 2 * p(x_vals[i])
        free_coeffs[i] = h ** 2 * f(x_vals[i])
    return linalg.solve(system_matrix, free_coeffs)


def least_squares(u, p, q, f, a, b, alpha0, alpha1, beta0, beta1, A, B, n=10):
    system_matrix, free_coeffs = np.empty((n, n)), np.empty(n)
    for i in range(n):
        free_coeffs[i] = Gaussian(lambda x: (f(x) - L(u, x, p, q, 0, a, b, alpha0, alpha1, beta0, beta1, A, B)) *
                         L(u, x, p, q, i + 1, a, b, alpha0, alpha1, beta0, beta1, A, B), a, b, N=15)
        for j in range(i, n):
            system_matrix[i][j] = system_matrix[j][i] = \
                Gaussian(lambda x: L(u, x, p, q, i + 1, a, b, alpha0, alpha1, beta0, beta1, A, B) *
                                   L(u, x, p, q, j + 1, a, b, alpha0, alpha1, beta0, beta1, A, B), a, b, N=15)
    return linalg.solve(system_matrix, free_coeffs)
