import numpy as np
import NBodyProblem as nbp
import matplotlib.pyplot as plt
import RungeKutta4 as rk4
import matplotlib.animation as animation
import scipy.optimize as optimize
from time import time
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import interp1d as interp
from mpl_toolkits.mplot3d import Axes3D


def init(lines_and_bodies):
    for l_or_b in lines_and_bodies:
        l_or_b.set_data([], [])
        l_or_b.set_3d_properties([])
    return lines_and_bodies


def update(lines_and_bodies, data, i):
    body_num = len(lines_and_bodies) // 2
    for j in range(body_num):
        lines_and_bodies[j].set_data(data[body_num * nbp.number_of_dimension + 3 * j, :i],
                                     data[body_num * nbp.number_of_dimension + 3 * j + 1, :i])
        lines_and_bodies[j].set_3d_properties(data[body_num * nbp.number_of_dimension + 3 * j + 2, :i])
    for j in range(body_num):
        lines_and_bodies[j+body_num].set_data([data[body_num * nbp.number_of_dimension + 3 * j, i]],
                                              [data[body_num * nbp.number_of_dimension + 3 * j + 1, i]])
        lines_and_bodies[j+body_num].set_3d_properties([data[body_num * nbp.number_of_dimension + 3 * j + 2, i]])
    return lines_and_bodies


def update_body(line, xdata, ydata, zdata, i):
    line.set_data([xdata[i]], [ydata[i]])
    line.set_3d_properties([zdata[i]])
    return line,


def deviation(masses, t_given, x_given, t0, T, x0, calc_eps, h_initial, equation_number, print_iter):

    t_current, x_current = rk4.solve_ode(t0, x0, t0, T, nbp.nbody_problem_ode_right_side, calc_eps=calc_eps,
                                         h_initial=h_initial, args=(masses, equation_number), print_iter=print_iter)

    x_given_interp, x_current_interp = \
        interp(t_given, x_given, kind='cubic', bounds_error=False, fill_value='extrapolate') if \
            t_given.size > 3 else interp(t_given, x_given, bounds_error=False, fill_value='extrapolate'),\
        interp(t_current, x_current, kind='cubic', bounds_error=False, fill_value='extrapolate') if \
            t_current.size > 3 else interp(t_current, x_current, bounds_error=False, fill_value='extrapolate')

    return quad(lambda t: np.linalg.norm(x_current_interp(t) - x_given_interp(t)) ** 2, t0, T)[0]


if __name__ == '__main__':

    figsize = (15.0, 7.5)
    lines, bodies = [], []
    colors = ('y', 'r', 'b', 'g')
    grid_dot_number = 5000
    body_number = 4
    equation_number = 2 * nbp.number_of_dimension * body_number
    masses = np.array([2*100**3, 15**3, 20**3, 8**3])
    momenta_initial, coordinates_initial = \
        np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
            [1.0, 1.0, -1.0]
        ]), \
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.11]
        ])
    x0 = nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial)
    t0, T = 0.0, 60.0 * 60
    t = np.linspace(t0, T, grid_dot_number)
    calc_eps, h_initial = 0.0001, 1.0
    print_iter = False

    bounds = [(0, 10**7) for _ in range(body_number)]
    time_for_optimization = 0.0

    solution = rk4.solve_ode(t0, nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial), t0, T,
                             nbp.nbody_problem_ode_right_side, calc_eps=calc_eps, h_initial=h_initial,
                             args=(masses, equation_number), print_iter=print_iter)

    solution_scipy = odeint(lambda x, t: nbp.nbody_problem_ode_right_side(t, x, (masses, equation_number)),
                            nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial),
                            solution[0]).T

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-0.05, 0.05)
    # ax.set_ylim(-0.05, 0.05)
    # ax.set_zlim(-0.05, 0.05)

    for i in range(body_number):
        lines.append(ax.plot(solution[1][body_number * nbp.number_of_dimension + 3 * i],
                             solution[1][body_number * nbp.number_of_dimension + 3 * i + 1],
                             solution[1][body_number * nbp.number_of_dimension + 3 * i + 2], colors[i] + '-')[0])

        bodies.append(ax.plot([coordinates_initial[i, 0]], [coordinates_initial[i, 1]], [coordinates_initial[i, 2]],
                              colors[i] + 'o')[0])

    lines_and_bodies = []
    lines_and_bodies.extend(lines)
    lines_and_bodies.extend(bodies)

    ani = animation.FuncAnimation(fig,
                                  lambda index: update(
                                      lines_and_bodies,
                                      solution[1],
                                      index),
                                  frames=np.arange(0, solution[1].shape[1], 3),
                                  init_func=lambda: init(lines_and_bodies),
                                  interval=100)

    print('Starting optimization...')

    time_for_optimization = time()

    optimization_result = optimize.differential_evolution(deviation, bounds,
                                                          args=(solution[0], solution[1], t0, T, x0, calc_eps,
                                                                h_initial, equation_number, print_iter))

    time_for_optimization = time() - time_for_optimization

    masses_sol = optimization_result.x
    print('Masses: {0}\nDeviation: {1}\nTime for optimization: {2}'
          .format(masses_sol, deviation(masses_sol, solution[0], solution[1], t0, T, x0, calc_eps, h_initial,
                                        equation_number, print_iter),
                  time_for_optimization))

    plt.show()
    plt.close()
