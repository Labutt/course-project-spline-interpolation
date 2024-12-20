import numpy as np
import matplotlib.pyplot as plt


def cubic_spline(x, y):
    n = len(x)
    a = y.copy()
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    h = np.diff(x)

    A = np.zeros((n, n))     # Система уравнений для нахождения коэффициентов
    rhs = np.zeros(n)

    for i in range(1, n-1): # Условия для внутренних узлов
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 3 * ((a[i+1] - a[i]) / h[i] - (a[i] - a[i-1]) / h[i-1])

    A[0, 0] = 1    # Граничные условия (естественный сплайн)
    A[-1, -1] = 1

    c = np.linalg.solve(A, rhs)     # Решаем систему уравнений

    for i in range(n-1):
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3     # Находим b и d
        d[i] = (c[i+1] - c[i]) / (3*h[i])

    return a, b, c, d

def spline_interpolate(x, y, x_new):
    a, b, c, d = cubic_spline(x, y)
    y_new = np.zeros_like(x_new)

    for i in range(len(x_new)):         # Находим соответствующий сегмент
        for j in range(len(x) - 1):
            if x[j] <= x_new[i] <= x[j + 1]:
                dx = x_new[i] - x[j]
                y_new[i] = a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
                break

    return y_new, a, b, c, d

def print_interpolation_function(x, a, b, c, d, decimals=2):
    print("Интерполяционные функции для сегментов:")
    for i in range(len(x) - 1):
        print(f"S_{i}(x) = {round(a[i], decimals)} + {round(b[i], decimals)}(x - {round(x[i], decimals)}) + {round(c[i], decimals)}(x - {round(x[i], decimals)})^2 + {round(d[i], decimals)}(x - {round(x[i], decimals)})^3, для {round(x[i], decimals)} <= x <= {round(x[i + 1], decimals)}")

def interpolate_at_value(x, a, b, c, d, value):
    if value < x[0] or value > x[-1]:     # Проверяем, находится ли значение в пределах x
        raise ValueError("Значение x находится вне диапазона интерполяции.")

    for j in range(len(x) - 1):     # Находим соответствующий сегмент
        if x[j] <= value <= x[j + 1]:
            dx = value - x[j]
            return a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3             # Вычисляем y по интерполяционной функции

    return None  # Если значение не найдено (что маловероятно)
#-----------------------------ЗАДАННЫЕ ТОЧКИ (погода)------------------

x = np.array([0, 2, 4, 6, 8]) # Исходные данные
y = np.array([4, 3, 0, 4, 3])

x_new = np.linspace(np.min(x), np.max(x), 100) # Новые точки для интерполяции
y_new, a, b, c, d = spline_interpolate(x, y, x_new)

print_interpolation_function(x, a, b, c, d) # Печать интерполяционной функции
print('y(1.5) = ',round(interpolate_at_value(x, a, b, c, d, 1.5), 2))

plt.scatter(x, y, color='red', label='Исходные точки') # Визуализация
plt.plot(x_new, y_new, label='Интерполированный сплайн')
plt.legend()
plt.title('Интерполяция кубическими сплайнами (заданные точки погоды)')
plt.xlabel('Номер дня')
plt.ylabel('Погода')
plt.grid()
plt.show()

#-----------------------------СЛУЧАЙНЫЕ ТОЧКИ------------------
np.random.seed(6)  # Для воспроизводимости
n_points = 6  # Количество точек
x = np.sort(np.random.uniform(0, 5, n_points))  # Случайные x-координаты
y = np.random.uniform(1, 6, n_points)  # Случайные y-координаты


x_new = np.linspace(np.min(x), np.max(x), 100) # Новые точки для интерполяции
y_new, a, b, c, d = spline_interpolate(x, y, x_new)


print_interpolation_function(x, a, b, c, d) # Печать интерполяционной функции
print('y(1.5) = ',round(interpolate_at_value(x, a, b, c, d, 1.5), 2))
# Визуализация
plt.scatter(x, y, color='red', label='Исходные точки')
plt.plot(x_new, y_new, label='Интерполированный сплайн')
plt.legend()
plt.title('Интерполяция кубическими сплайнами (случайные точки)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()