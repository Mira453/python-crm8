import numpy as np
import matplotlib.pyplot as plt

# Параметри задачі
h = 0.2
x = np.arange(0.1, 1.1 + h, h)
N = len(x)

# Початкові значення
A = np.zeros((N, N))
b = np.zeros(N)

# Гранична умова: y'(0.1) = 1 → y_{-1} = y_1 - 2h
# Розв’язок першого рівняння
A[0, 0] = -2 - h**2
A[0, 1] = 1 - h * np.exp(x[0])
b[0] = 2 * h**2 - (1 + h * np.exp(x[0])) * (-2 * h)

# Внутрішні точки
for i in range(1, N - 1):
    A[i, i - 1] = 1 + h * np.exp(x[i]) / 2
    A[i, i] = -2 - h**2
    A[i, i + 1] = 1 - h * np.exp(x[i]) / 2
    b[i] = 2 * h**2

# Остання точка: y_N = y_{N-1} / (1 + h)
A[N - 1, N - 2] = -1 / (1 + h)
A[N - 1, N - 1] = 1
b[N - 1] = 0

# Розв’язок СЛАР
y = np.linalg.solve(A, b)

# Вивід результатів
for i in range(N):
    print(f"x = {x[i]:.1f}, y = {y[i]:.6f}")

# Побудова графіка
plt.plot(x, y, 'o-', label='Чисельне рішення')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Крайова задача: метод скінченних різниць')
plt.grid()
plt.legend()
plt.show()
