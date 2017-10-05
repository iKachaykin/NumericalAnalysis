from integrals import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg


def f(x):
    return 1 / (4 * np.exp(x) + 9 * np.exp(-x))


def norm_f(f, x, a, b):
    return (b - a) / 2 * f((b - a) / 2 * x + (b + a) / 2)


col_labels = ("Метод", "Выч. значение", "Точное значение", "Абс. погрешность", "Оценка погрешности")
methods_names = ("Левых пр.", "Правых пр.", "Центральных пр.",
                 "Трапеций", "Симпсона", "Гаусса")
funcs_create_figs = (create_approximating_figure_left_rectangles, create_approximating_figure_right_rectangles,
                     create_approximating_figure_central_rectangles, create_approximating_figure_trapeze,
                     create_approximating_figure_Simpson)
interval_num, left_border, right_border, b_interval_dot_num, s_interval_dot_num, additional_interval_koeff = \
    10, 0, 1, 500, 100, 1
index_Gaussian = 5
exact_val = np.float((np.arctan(2 * np.exp(right_border) / 3) - np.arctan(2 * np.exp(left_border) / 3)) / 6)
x_vals = np.linspace(left_border - additional_interval_koeff * (right_border - left_border),
                     right_border + additional_interval_koeff * (right_border - left_border), b_interval_dot_num)
x_node_vals = np.linspace(left_border, right_border, interval_num + 1)
f_vals = np.arange(100)
y_vals_left = \
np.where(np.sign(f(left_border)) * np.linspace(0, np.sign(f(left_border)) * np.abs(f(x_vals)).max(), x_vals.size) <
         np.sign(f(left_border)) * f(left_border),
         np.linspace(0,  np.sign(f(left_border)) * np.abs(f(x_vals)).max(), x_vals.size), 0)
y_vals_right = \
np.where(np.sign(f(right_border)) * np.linspace(0, np.sign(f(right_border)) * np.abs(f(x_vals)).max(), x_vals.size) <
         np.sign(f(right_border)) * f(right_border),
         np.linspace(0, np.sign(f(right_border)) * np.abs(f(x_vals)).max(), x_vals.size), 0)
y_vals_left_in_Gaussian = \
np.where(np.sign(norm_f(f, -1, left_border, right_border)) *
         np.linspace(0, np.sign(norm_f(f, -1, left_border, right_border)) * np.abs(norm_f(f, x_vals, left_border, right_border)).max(), x_vals.size) <
         np.sign(norm_f(f, -1, left_border, right_border)) * norm_f(f, -1, left_border, right_border),
         np.linspace(0, np.sign(norm_f(f, -1, left_border, right_border)) * np.abs(norm_f(f, x_vals, left_border, right_border)).max(), x_vals.size), 0)
y_vals_right_in_Gaussian = \
np.where(np.sign(norm_f(f, 1, left_border, right_border)) *
         np.linspace(0, np.sign(norm_f(f, 1, left_border, right_border)) * np.abs(norm_f(f, x_vals, left_border, right_border)).max(), x_vals.size) <
         np.sign(norm_f(f, 1, left_border, right_border)) * norm_f(f, 1, left_border, right_border),
         np.linspace(0, np.sign(norm_f(f, 1, left_border, right_border)) * np.abs(norm_f(f, x_vals, left_border, right_border)).max(), x_vals.size), 0)
result_values = (left_rectangles(f, left_border, right_border, interval_num),
                 right_rectangles(f, left_border, right_border, interval_num),
                 central_rectangles(f, left_border, right_border, interval_num),
                 trapezoid(f, left_border, right_border, interval_num),
                 Simpson(f, left_border, right_border, interval_num),
                 Gaussian(f, left_border, right_border, interval_num))
data = []
for result, name in zip(result_values, methods_names):
    data.append([name, result, exact_val, np.abs(result - exact_val)])
figures = []
for i in range(len(methods_names)):
    figures.append(plt.figure(figsize=(12, 7)))
    figures[i].suptitle("Метод " + methods_names[i])
figures.append(plt.figure(figsize=(12, 7)))
figures[len(methods_names)].suptitle("Таблица с результатами вычислений")
for i in range(len(methods_names)):
    plt.figure(i + 1)
    ax = plt.subplot()
    ax.fill_between(np.linspace(left_border, right_border, b_interval_dot_num), np.zeros(b_interval_dot_num),
                    f(np.linspace(left_border, right_border, b_interval_dot_num)), color='blue', alpha=0.3)
    for j in range(x_node_vals.size - 1):
        if i == index_Gaussian:
            break
        f_vals = funcs_create_figs[i](f, x_node_vals, j, s_interval_dot_num, interval_num)
        ax.fill_between(np.linspace(x_node_vals[j], x_node_vals[j + 1], s_interval_dot_num),
                        np.zeros(s_interval_dot_num), f_vals, color='red', alpha=0.1)
    ax.grid(True, zorder=5)
    if i == index_Gaussian:
        x_node_vals = Gaussian_node_vals(interval_num)
    plt.plot(x_vals, f(x_vals), 'r-', np.ones_like(x_vals) * left_border, y_vals_left, 'k-',
             np.ones_like(x_vals) * right_border, y_vals_right, 'k-', x_vals, np.zeros_like(x_vals), 'k-',
             x_node_vals, np.zeros_like(x_node_vals), 'k|')
plt.plot(x_vals, norm_f(f, x_vals, left_border, right_border), 'b-', np.ones_like(x_vals) * -1, y_vals_left_in_Gaussian,
         'c-', np.ones_like(x_vals), y_vals_right_in_Gaussian, 'c-')
ax.fill_between(np.linspace(-1, 1, b_interval_dot_num), np.zeros(b_interval_dot_num),
                norm_f(f, np.linspace(-1, 1, b_interval_dot_num), left_border, right_border), color='magenta',
                alpha=0.1)
plt.figure(7)
plt.axis('off')
ax = plt.subplot()
tab = plt.table(cellText=data, loc='center', colLabels=col_labels, cellLoc='center')
plt.show()
plt.close()
