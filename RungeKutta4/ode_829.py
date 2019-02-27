import numpy as np
import matplotlib.pyplot as plt
import RungeKutta4 as rk4
import RungeKutta4_cython as rk4_cython
import time as time


def ode_f(t, x):
    return np.array([2.0 * x[0] - 4.0 * x[1] + 4.0 * np.exp(-2.0 * t),
                     2 * x[0] - 2 * x[1]])


def ode_exact_solution(t):
    return np.array([2.0 * np.cos(2.0 * t),
                     np.cos(2.0 * t) + np.sin(2.0 * t) + np.exp(-2.0 * t)])


if __name__ == '__main__':

    a, b = -1.0, 3.0
    grid_dot_num = 500
    figsize = (15.0, 7.5)
    t0, x0 = 0.0, np.array([2.0, 2.0])
    h_initial = 0.1

    t_exact = np.linspace(a, b, grid_dot_num)
    x_exact = ode_exact_solution(t_exact)

    tp = time.time()
    t_numerical, x_numerical = rk4.solve_ode(t0, x0, a, b, ode_f, calc_eps=1e-8, h_initial=h_initial)
    tp = time.time() - tp

    tc = time.time()
    t_numerical_cython, x_numerical_cython = rk4_cython.solve_ode(t0, x0, a, b, ode_f, calc_eps=1e-8,
                                                                  h_initial=h_initial)
    tc = time.time() - tc

    print('||t1 - t2|| = %f\n||x1 - x2|| = %f\nTime python: %f\nTime cython: %f\n' %
          (np.linalg.norm(t_numerical - t_numerical_cython), np.linalg.norm(x_numerical - x_numerical_cython),
           tp, tc))

    plt.figure(figsize=figsize)
    plt.plot(t_exact, x_exact[0], 'g-', t_exact, x_exact[1], 'b-', t_numerical, x_numerical[0], 'r--',
             t_numerical, x_numerical[1], 'k--')
    plt.grid(True)
    plt.show()
    plt.close()
