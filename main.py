import numpy as np
import matplotlib.pyplot as plt


def polyfit(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    poly = np.poly1d(coeffs)
    return poly


def lagrange(x, y, x_i):
    n = len(x)
    interp_y = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_i - x[j]) / (x[i] - x[j])
        interp_y += term
    return interp_y


def newton(x, y, x_interp):
    n = len(x)
    coeffs = np.zeros(n)
    coeffs[0] = y[0]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            y[j] = (y[j] - y[j - 1]) / (x[j] - x[j - i])
        coeffs[i] = y[i]
    interp_y = 0
    for i in range(n):
        term = coeffs[i]
        for j in range(i):
            term *= (x_interp - x[j])
        interp_y += term
    return interp_y


x_data = np.array([0, 2, 3, 4, 7, 9])
y_data = np.array([4, 26, 58, 112, 466, 922])
deg = len(x_data) - 1
input_x = int(input("f(x); x=? "))
x_range = np.arange(x_data.min(), x_data.max() + 1)

poly = polyfit(x_data, y_data, deg)
poly_y = poly(x_range)

lagrange_y = lagrange(x_data, y_data, input_x)

newton_y = newton(x_data, y_data, input_x)

plt.figure(figsize=(8, 6))

plt.scatter(x_data, y_data, color='red', label='Data')

plt.plot(x_range, poly_y, linestyle='-', color='blue', label='Polyfit')
plt.plot(input_x, lagrange_y, marker='o', markersize=8, color='green', label='Lagrange')
plt.plot(input_x, newton_y, marker='x', markersize=10, color='orange', label="Newton")


plt.text(input_x, lagrange_y + 300, f'Lagrange: f({input_x}) = {lagrange_y}', fontsize=8, verticalalignment='bottom')
plt.text(input_x, newton_y + 400, f'Newton: f({input_x}) = {newton_y}', fontsize=8, verticalalignment='bottom')
plt.text(input_x, poly(input_x) + 500, f'Polyfit: f({input_x}) = {poly(input_x)}', fontsize=8,
         verticalalignment='bottom')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation')
plt.legend()
plt.grid(True)
plt.show()
