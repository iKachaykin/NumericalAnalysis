import numpy as np
import numpy.polynomial.polynomial as poly


def gen_y_sequence(input_matrix):
    if not isinstance(input_matrix, np.ndarray):
        raise "gen_y_sequence takes only NumPy arrays as arguments"
    if len(input_matrix.shape) > 2 or input_matrix.shape[0] != input_matrix.shape[1]:
        raise "The argument must be square matrix"
    res, eps = [], 0.000001
    res.append(np.random.rand(input_matrix.shape[0]))
    for i in range(1, input_matrix.shape[0] + 1):
        res.append(np.dot(input_matrix, res[i - 1]))
    return np.vstack(res[::-1])


def Krilov_method(input_matrix):
    y_seq = gen_y_sequence(input_matrix)
    tmp = np.linalg.det(y_seq.T[:, 1:])
    pol_coeffs = np.linalg.solve(y_seq.T[:, 1:], -y_seq.T[:, 0])
    pol_coeffs = pol_coeffs[::-1]
    eig_vals = poly.polyroots(np.hstack((pol_coeffs, 1)))
    eig_vectors = []
    for eigv in eig_vals:
        betta_vector = []
        betta_vector.append(1)
        for i in range(input_matrix.shape[0] - 1):
            betta_vector.append(poly.polyval(eigv, np.hstack((pol_coeffs[pol_coeffs.size - i - 1:], 1))))
        betta_vector = np.array(betta_vector)
        eig_vectors.append((y_seq.T[:, 1:] * betta_vector).T.sum(axis=0))
    return eig_vals, np.array(eig_vectors)
