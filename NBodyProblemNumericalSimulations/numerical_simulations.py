import numpy as np
import NBodyProblem as nbp
import matplotlib.pyplot as plt
import RungeKutta4 as rk4
import matplotlib.animation as animation
from astropy.constants import iau2015 as constants


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


if __name__ == '__main__':

    figsize = (15.0, 7.5)
    lines, bodies = [], []
    colors = ('y', 'r', 'b', 'g')
    grid_dot_number = 5000
    year_number = 2
    body_number = 4
    scale = float(constants.M_sun.value)
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
    t0, T = 0.0, 60.0 * 60
    t = np.linspace(t0, T, grid_dot_number)
    calc_eps, h_initial = 0.000001, 1.0

    solution = rk4.solve_ode(t0, nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial), t0, T,
                             nbp.nbody_problem_ode_right_side, calc_eps=calc_eps, h_initial=h_initial,
                             args=(masses, scale))

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
                                  interval=25)

    plt.show()
    plt.close()
