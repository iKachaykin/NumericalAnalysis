import numpy as np


min_eps = 1e-52


cpdef args_preprocessing(double t0, double[:] x0, double a, double b, f, double calc_eps, double h_initial):

    if not (a <= t0 <= b):
        raise ValueError('t0 must be from a segment [a, b] (a <= b)')

    tmp_1 = np.array(x0)
    tmp_2 = np.array(f(t0, x0))

    if calc_eps < 0.0:
        raise ValueError('"calc_eps" could not be negative!')

    if h_initial <= 0.0:
        raise ValueError('"h_initial" must be positive!')

    return 0


def convolve_args(f, args=None):

    if args is not None:
        return lambda t, x: f(t, x, args)
    else:
        return f


cpdef calculate_delta(double ti, double[:] xi, double hi, f):

    phi0 = hi * np.array(f(ti, xi))
    phi1 = hi * np.array(f(ti + hi / 2.0, xi + phi0 / 2.0))
    phi2 = hi * np.array(f(ti + hi / 2.0, xi + phi1 / 2.0))
    phi3 = hi * np.array(f(ti + hi, xi + phi2))

    return (phi0 + 2 * phi1 + 2 * phi2 + phi3) / 6.0


cpdef solve_ode(double t0, double[:] x0, double a, double b, f, double calc_eps=1e-12, double h_initial=0.1,
                args=None):

    f_tmp = convolve_args(f, args)
    args_preprocessing(t0, x0, a, b, f_tmp, calc_eps, h_initial)

    ti = t0
    hi = np.minimum(h_initial, b - t0)
    xi = np.array(x0).copy()

    results_x = [xi.copy()]
    result_t = [ti]

    while True:

        x_first = xi + calculate_delta(ti, xi, hi, f_tmp)

        x_tmp = xi + calculate_delta(ti, xi, hi / 2.0, f_tmp)
        x_second = x_tmp + calculate_delta(ti, x_tmp, hi / 2.0, f_tmp)

        eps_half = np.linalg.norm(x_second - x_first) / 15.0
        eps_full = eps_half * 16.0

        if eps_half > calc_eps:

            hi *= 0.5
            continue

        ti += hi
        xi = x_second.copy()

        results_x.append(xi.copy())
        result_t.append(ti)

        if eps_full < calc_eps:
            hi *= 2

        if b - ti - hi < -min_eps:
            hi = b - ti

        if np.abs(b - ti) < min_eps:
            break

    ti = t0
    hi = t0 - a if h_initial is None else np.minimum(h_initial, t0 - a)
    xi = np.array(x0).copy()

    while True:

        x_first = xi + calculate_delta(ti, xi, -hi, f_tmp)

        x_tmp = xi + calculate_delta(ti, xi, -hi / 2.0, f_tmp)
        x_second = x_tmp + calculate_delta(ti, x_tmp, -hi / 2.0, f_tmp)

        eps_half = np.linalg.norm(x_second - x_first) / 15.0
        eps_full = eps_half * 16.0

        if eps_half > calc_eps:

            hi *= 0.5
            continue

        ti -= hi
        xi = x_second.copy()

        results_x.insert(0, xi.copy())
        result_t.insert(0, ti)

        if eps_full < calc_eps:
            hi *= 2

        if ti - hi - a < -min_eps:
            hi = ti - a

        if np.abs(ti - a) < min_eps:
            break

    return np.array(result_t), np.array(results_x).T

