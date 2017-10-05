import Krilov
import numpy as np


# A = np.arange(0, 31, 2).reshape(4, 4)
# A = np.int32(np.random.rand(3, 3) * 20 - 10)
# A = np.array([[1, 3, 1], [3, 1, 1], [1, 1, 3]])
A = np.array([[5, 4, 4], [4, 4, 5], [4, 5, 4]])
# A = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print(np.linalg.eigvals(A), "\n", Krilov.Krilov_method(A)[0], "\n\n")
eig_vals, eig_vects = np.linalg.eig(A) # Krilov.Krilov_method(A)
for val, vect in zip(eig_vals, eig_vects):
    print("res:\n", np.dot(A, vect), '\n', val * vect, '\n', vect, '\n\n\n')
