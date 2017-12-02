import matplotlib.pyplot as plt
from ode import *


def f(x, y):
    return (y + x * x + y * y) / x


def y(x):
    return x * np.tan(x - 1)


a, b, y_0, methods_num, exact_sol_dot_num, scale, border_dot_num, fg_size, error_num, Runge_Kutta_order = \
    1, 2, 0, 4, 500, 0.0, 2, (15, 7.5), 2, 4
error_names = ("Глобальное отклонение", "Среднеквадратическое отклонение")
methods_names = ("Метод Рунге-Кутта авт.ш.", "Метод Эйлера", "Метод Рунге-Кутта п.ш.", "Интерполяционный метод Адамса")
col_labels = ("Номер узла", "Узлы", "Точное значение решения", "Вычисленное значение", "Локальная абсолютная погрешность")
methods = (Runge_Kutta4_auto, Euler, Runge_Kutta4_const, Adams_interp)
data = []
x_exact_vals = np.linspace(a - (b - a) * scale, b + (b - a) * scale, exact_sol_dot_num)
y_exact_vals = y(x_exact_vals)
for method_i in range(methods_num):
    plt.figure(2 * method_i + 1, figsize=fg_size)
    plt.suptitle(methods_names[method_i])
    if method_i == 0:
        x_res_nodes, y_res_nodes, interval_nums = tuple(methods[method_i](f, y_0, a, b, h0=1e-5))
        init_Adams_values_num = int(interval_nums / 2)
    elif method_i != methods_num - 1:
        x_res_nodes, y_res_nodes, tmp = tuple(methods[method_i](f, y_0, a, b, interval_nums))
    else:
        x_res_nodes, y_res_nodes, tmp = tuple(methods[method_i](f, y_0, a, b, interval_nums, init_Adams_values_num))
    sqrt_error = np.sqrt(((y(x_res_nodes) - y_res_nodes) ** 2).sum())
    global_error = (np.abs(y(x_res_nodes) - y_res_nodes)).sum()
    plt.plot(x_exact_vals, y_exact_vals, 'b-', x_res_nodes, y_res_nodes, 'r-',
             np.ones(border_dot_num) * a, np.linspace(0, y_exact_vals.max(), border_dot_num), 'k--',
             np.ones(border_dot_num) * b, np.linspace(0, y_exact_vals.max(), border_dot_num), 'k--')
    ax = plt.subplot()
    ax.grid(True)
    plt.figure(2 * method_i + 2, figsize=fg_size)
    data.clear()
    for i in range(interval_nums + 1):
        data.append([i, x_res_nodes[i], y(x_res_nodes[i]), y_res_nodes[i], np.abs(y(x_res_nodes[i]) - y_res_nodes[i])])
    data.append([error_names[0], "", "", "", global_error])
    data.append([error_names[1], "", "", "", sqrt_error])
    plt.suptitle("Таблица значений. " + methods_names[method_i])
    plt.axis('off')
    ax = plt.subplot()
    tab = plt.table(cellText=data, loc='center', colLabels=col_labels, cellLoc='center')
plt.show()
plt.close()
