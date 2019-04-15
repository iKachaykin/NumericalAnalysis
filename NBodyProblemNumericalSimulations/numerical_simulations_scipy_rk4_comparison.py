import numpy as np
import NBodyProblem as nbp
import RungeKutta4 as rk4
from time import time
from tqdm import tqdm
from scipy.integrate import odeint


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
    t0, T = 0.0, 60.0 * 60
    calc_eps, h_initial = 0.0001, 1.0

    repeat_num = 1000
    av_time_of_ode_calculations = 0.0
    av_time_of_ode_calculations_scipy = 0.0

    for _ in tqdm(range(repeat_num)):

        time_of_ode_calculations = time()

        solution = rk4.solve_ode(t0, nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial), t0, T,
                                 nbp.nbody_problem_ode_right_side, calc_eps=calc_eps, h_initial=h_initial,
                                 args=(masses, equation_number), print_iter=False)

        time_of_ode_calculations = time() - time_of_ode_calculations
        av_time_of_ode_calculations += time_of_ode_calculations

        time_of_ode_calculations_scipy = time()

        solution_scipy = odeint(lambda x, t: nbp.nbody_problem_ode_right_side(t, x, (masses, equation_number)),
                                nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial),
                                solution[0]).T

        time_of_ode_calculations_scipy = time() - time_of_ode_calculations_scipy
        av_time_of_ode_calculations_scipy += time_of_ode_calculations_scipy

    av_time_of_ode_calculations /= repeat_num
    av_time_of_ode_calculations_scipy /= repeat_num
    print('rk4: %f' % av_time_of_ode_calculations)
    print('scipy: %f' % av_time_of_ode_calculations_scipy)
