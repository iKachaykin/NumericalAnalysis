import numpy as np
from scipy.constants import gravitational_constant
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool


number_of_dimension = int(3)


def vector_to_momenta_and_coordinates(vector):

    if len(vector.shape) != 1 or vector.size % (2 * number_of_dimension) != 0:
        raise ValueError('"vector" must have type "ndarray" with shape=(%d * N,), where N is a positive integer!' %
                         (2 * number_of_dimension))

    body_number = vector.size // (2 * number_of_dimension)
    return (vector[:body_number * number_of_dimension].reshape(body_number, number_of_dimension),
            vector[body_number * number_of_dimension:].reshape(body_number, number_of_dimension))


def momenta_and_coordinates_to_vector(momenta, coordinates):

    if len(momenta.shape) != 2 or len(coordinates.shape) != 2 or momenta.shape[1] != number_of_dimension or \
            coordinates.shape[1] != number_of_dimension or momenta.shape[0] != coordinates.shape[0]:
        raise ValueError('"momenta" and "coordinates" must have type "ndarray" with shape=(N, %d),'
                         'where N is a positive integer!' % number_of_dimension)

    return np.concatenate((momenta.ravel(), coordinates.ravel()))


def nbody_problem_ode_right_side(t, x, args):

    momenta, coordinates = vector_to_momenta_and_coordinates(x)
    masses, eq_num = args

    # coordinates_der = momenta / masses.reshape(masses.size, 1)
    #
    # momenta_der = np.ones_like(coordinates_der)
    #
    # i = 0
    # j = 0
    #
    # while i < momenta_der.shape[0]:
    #     j = 0
    #     while j < number_of_dimension:
    #         momenta_der[i, j] = -gravitational_constant * ((coordinates[i, j] - coordinates[:i, j]) * masses[i] *
    #                                                        masses[:i] / np.linalg.norm(
    #                     coordinates[i] - coordinates[:i], axis=1) ** 3).sum() - \
    #                             gravitational_constant * ((coordinates[i, j] - coordinates[i + 1:, j]) * masses[i] *
    #                                                       masses[i + 1:] / np.linalg.norm(
    #                     coordinates[i] - coordinates[i + 1:], axis=1) ** 3).sum()
    #         j += 1
    #     i += 1
    #
    # return momenta_and_coordinates_to_vector(momenta_der, coordinates_der)

    right_side = np.zeros(eq_num)
    body_num = eq_num // (2 * number_of_dimension)

    for k in range(right_side.size):

        if k < number_of_dimension * body_num:

            sum_inds = np.concatenate((np.arange(np.floor(k / number_of_dimension)),
                                       np.arange(np.floor(k / number_of_dimension)+1, body_num)))
            sum_inds = np.array(sum_inds, dtype=np.int32)
            right_side[k] = -gravitational_constant * masses[int(np.floor(k / number_of_dimension))] * np.sum(
                masses[sum_inds] * (x[number_of_dimension * body_num + k] -
                                    x[number_of_dimension * body_num + k % number_of_dimension +
                                      number_of_dimension * sum_inds]) /
                np.sqrt((x[number_of_dimension * body_num + number_of_dimension *
                           int(np.floor(k / number_of_dimension))] -
                         x[number_of_dimension * body_num + number_of_dimension * sum_inds])**2 +
                        (x[number_of_dimension * body_num +
                           number_of_dimension * int(np.floor(k / number_of_dimension)) + 1] -
                         x[number_of_dimension * body_num + 1 + number_of_dimension * sum_inds])**2 +
                        (x[number_of_dimension * body_num +
                           number_of_dimension * int(np.floor(k / number_of_dimension)) + 2] -
                         x[number_of_dimension * body_num + 2 + number_of_dimension * sum_inds])**2) ** 3)

        else:
            right_side[k] = x[k - number_of_dimension * body_num] / \
                            masses[int(np.floor((k - number_of_dimension * body_num) / number_of_dimension))]

    return right_side


def momenta_der_i_component(i, coordinates, masses):

    result = np.zeros(number_of_dimension)
    j = 0

    while j < number_of_dimension:
        result[j] = -gravitational_constant * ((coordinates[i, j] - coordinates[:i, j])
                                               * masses[i] * masses[:i] /
                                               np.linalg.norm(coordinates[i] - coordinates[:i], axis=1) ** 3).sum() -\
                    gravitational_constant * ((coordinates[i, j] - coordinates[i+1:, j])
                                              * masses[i] * masses[i+1:] /
                                              np.linalg.norm(coordinates[i] - coordinates[i+1:], axis=1) ** 3).sum()
        j += 1

    return result
