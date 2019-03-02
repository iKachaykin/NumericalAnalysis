import numpy as np
import NBodyProblem as nbp
import matplotlib.pyplot as plt
import RungeKutta4 as rk4
from astropy.constants import iau2015 as constants
from astropy.constants.codata2014 import c
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':

    figsize = (15.0, 7.5)
    grid_dot_number = 5000
    year_number = 5
    body_number = 2
    masses = np.array([float(constants.M_sun.value), float(constants.M_earth.value)])
    momenta_initial, coordinates_initial = \
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), \
        np.array([
            [0.0, 0.0, 0.0],
            [float(c.value) * 8 * 60, 0.0, 0.0]
        ])
    t0, T = 0.0, 365 * 24 * 60 * 60.0 * year_number
    t = np.linspace(t0, T, grid_dot_number)
    calc_eps, h_initial = 0.1, 3600

    solution = rk4.solve_ode(t0, nbp.momenta_and_coordinates_to_vector(momenta_initial, coordinates_initial), t0, T,
                             nbp.nbody_problem_ode_right_side, calc_eps=calc_eps, h_initial=h_initial, args=1.0)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection='3d')

    for i in range(body_number):
        ax.plot(solution[1][body_number * nbp.number_of_dimension + 3 * i],
                solution[1][body_number * nbp.number_of_dimension + 3 * i + 1],
                solution[1][body_number * nbp.number_of_dimension] + 3 * i + 2, 'r-')

    plt.show()
    plt.close()
