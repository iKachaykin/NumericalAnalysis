import matplotlib.pyplot as plt
from equation_solution import *


def equation(x):
    return x * x - np.sin(x)


a, b, x_scale, dot_num, fg_size = 0.25, 3, 0.5, 1000, (15, 7.5)
x_vals = np.linspace(a - x_scale * (b - a), b + x_scale * (b - a), dot_num)
y_vals, dy_div_dx_vals = equation(x_vals), derivative(equation, x_vals, dx=1e-6)
y_min, y_max = np.min([y_vals.min(), dy_div_dx_vals.min()]), np.max([y_vals.max(), dy_div_dx_vals.max()])
sol = simple_iteration(equation, a, b)
plt.figure(figsize=fg_size)
plt.grid(True)
plt.plot(x_vals, y_vals, 'b-', x_vals, dy_div_dx_vals, 'r-',
         [a, a], [y_min, y_max], 'k--', [b, b], [y_min, y_max], 'k--',
         [a - x_scale * (b - a), b + x_scale * (b - a)], [0, 0], 'k-', [sol, sol], [y_min, y_max], 'c--')
plt.suptitle("The solution of this equation is: " + (r"$x_0 \approx %f$" % sol))
plt.show()
plt.close()