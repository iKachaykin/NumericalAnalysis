import numpy as np
import NBodyProblem as nbp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':

    p, q = np.random.randint(0, 10, (6, 3)), np.random.randint(0, 10, (6, 3))

    print(p, q)
    v = nbp.momenta_and_coordinates_to_vector(p, q)
    print(v)
    print(nbp.vector_to_momenta_and_coordinates(v))

    fig = plt.figure(figsize=(15.0, 7.5))

    ax1 = fig.add_subplot(121, projection='3d')
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, 500), np.linspace(-1.0, 1.0, 500))
    zz = xx**2 + yy**2
    ax1.plot_wireframe(xx, yy, zz)

    ax2 = fig.add_subplot(122, projection='3d')
    t = np.linspace(-10.0, 10.0, 500)
    x = np.cos(t)
    y = np.sin(t)
    z = t.copy()
    ax2.plot(x, y, z, 'r-')

    plt.show()
    plt.close()
