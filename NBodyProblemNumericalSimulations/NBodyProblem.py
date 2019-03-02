import numpy as np
import RungeKutta4 as rk4
from scipy.constants import gravitational_constant


number_of_dimension = int(3)


def vector_to_momenta_and_coordinates(vector):

    if len(vector.shape) != 1 or vector.size % (2 * number_of_dimension) != 0:
        raise ValueError('"vector" must have type "ndarray" with shape=(%d * N,), where N is a positive integer!' %
                         (2 * number_of_dimension))

    body_number = vector.size // (2 * number_of_dimension)
    return (vector[:body_number * number_of_dimension].reshape(body_number, number_of_dimension),
            vector[body_number * number_of_dimension:].reshape(body_number, number_of_dimension))


# def vector_to_momenta_and_coordinates_m(vector_m):
#
#     if len(vector_m.shape) != 2 or vector_m.shape[0] % (2 * number_of_dimension) != 0:
#         raise ValueError('"vector_m" must have type "ndarray" with shape=(%d * N, M), where N is a positive integer!' %
#                          (2 * number_of_dimension))
#
#     pass


def momenta_and_coordinates_to_vector_m(momenta_m, coordinates_m):

    pass


def momenta_and_coordinates_to_vector(momenta, coordinates):

    if len(momenta.shape) != 2 or len(coordinates.shape) != 2 or momenta.shape[1] != number_of_dimension or \
            coordinates.shape[1] != number_of_dimension or momenta.shape[0] != coordinates.shape[0]:
        raise ValueError('"momenta" and "coordinates" must have type "ndarray" with shape=(N, %d),'
                         'where N is a positive integer!' % number_of_dimension)

    return np.concatenate((momenta.ravel(), coordinates.ravel()))


def nbody_problem_ode_right_side(t, x, args):

    momenta, coordinates = vector_to_momenta_and_coordinates(x)

    return np.ones_like(x)