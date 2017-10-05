import numpy as np
import numpy.polynomial.legendre as leg
import scipy.interpolate as interp


def left_rectangles(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    results = []
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            left, right = right, left
            weight = -1
        x_vals = np.linspace(left, right, N + 1)
        result = integrand(x_vals[:x_vals.size - 1]).sum() * (right - left) / N
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def right_rectangles(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    results = []
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            left, right = right, left
            weight = -1
        x_vals = np.linspace(left, right, N + 1)
        result = integrand(x_vals[1:]).sum() * (right - left) / N
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def central_rectangles(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    results = []
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            left, right = right, left
            weight = -1
        x_vals = np.linspace(left, right, N + 1)
        result = integrand((x_vals[1:] + x_vals[:x_vals.size - 1]) / 2).sum() * (right - left) / N
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def trapezoid(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    results = []
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            left, right = right, left
            weight = -1
        x_vals = np.linspace(left, right, N + 1)
        result = (integrand(x_vals[1:x_vals.size - 1]).sum() + (integrand(x_vals[0]) + integrand(x_vals[x_vals.size - 1])) / 2) * (right - left) / N
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


def Simpson(integrand, left_b_arr, right_b_arr, N=10):
    if not isinstance(left_b_arr, np.ndarray) or not isinstance(right_b_arr, np.ndarray):
        left_b_arr, right_b_arr = np.array(left_b_arr), np.array(right_b_arr)
    if left_b_arr.ravel().size != right_b_arr.ravel().size:
        return None
    results = []
    for left, right in zip(left_b_arr.ravel(), right_b_arr.ravel()):
        weight = 1
        if left >= right:
            left, right = right, left
            weight = -1
        x_vals = np.linspace(left, right, 2 * N + 1)
        result = (2 * integrand(x_vals[2:2 * N - 1:2]).sum() + integrand(left) + integrand(right) + 4 * integrand(x_vals[1::2]).sum())\
                 * (right - left) / N / 6
        results.append(result * weight)
    return np.array(results).reshape(left_b_arr.shape)


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


def create_approximating_figure_left_rectangles(integrand, nodes, current_i, dot_num, N=10):
    return integrand(nodes[current_i]) * np.ones(dot_num)


def create_approximating_figure_right_rectangles(integrand, nodes, current_i, dot_num, N=10):
    return integrand(nodes[current_i + 1]) * np.ones(dot_num)


def create_approximating_figure_central_rectangles(integrand, nodes, current_i, dot_num, N=10):
    return integrand((nodes[current_i] + nodes[current_i + 1]) / 2) * np.ones(dot_num)


def create_approximating_figure_trapeze(integrand, nodes, current_i, dot_num, N=10):
    return ((np.linspace(nodes[current_i], nodes[current_i + 1], dot_num) - nodes[current_i]) /
    (nodes[current_i + 1] - nodes[current_i])) * (integrand(nodes[current_i + 1]) - integrand(nodes[current_i])) + (
        integrand(nodes[current_i]))


def create_approximating_figure_Simpson(integrand, nodes, current_i, dot_num, N=10):
    lagrange_nodes = np.array([nodes[current_i], (nodes[current_i] + nodes[current_i + 1]) / 2, nodes[current_i + 1]])
    lagrange_poly = interp.lagrange(lagrange_nodes, integrand(lagrange_nodes))
    return lagrange_poly(np.linspace(nodes[current_i],
            nodes[current_i + 1], dot_num))


def Gaussian_node_vals(N=10):
    return leg.legroots(np.where(np.arange(N + 2) == N + 1, 1, 0))