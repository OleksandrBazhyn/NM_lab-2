import numpy as np

def is_tridiagonal(mat):
    """Перевіряє, чи є матриця тридіагональною."""
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and mat[i, j] != 0:
                return False
    return True

def thomas_algorithm(a, b, c, d):
    """
    Реалізація методу прогонки (алгоритму Томаса) для тридіагональних матриць.
    a - піддіагональ (n-1 елементів)
    b - діагональ (n елементів)
    c - наддіагональ (n-1 елементів)
    d - вектор правої частини
    """
    n = len(d)
    # Прямий хід
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i-1] * c_[i-1]
        if i < n - 1:
            c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i-1] * d_[i-1]) / denom

    # Зворотний хід
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    return x

def solve_system(matrix, vector):
    """Основна функція для перевірки і розв'язання СЛАР."""
    if is_tridiagonal(matrix):
        print("Матриця тридіагональна. Використовуємо метод прогонки.")
        n = len(vector)
        a = np.diag(matrix, -1)  # Піддіагональ
        b = np.diag(matrix)      # Діагональ
        c = np.diag(matrix, 1)   # Наддіагональ
        return thomas_algorithm(a, b, c, vector)
    else:
        print("Матриця не тридіагональна. Використовуємо метод Гаусса.")
        return np.linalg.solve(matrix, vector)

# Вхідні дані
matrix = np.array([
    [-9, -2, -6, 1],
    [4, -9, 1, 0],
    [1, -1, 6, -1],
    [-5, 2, -1, -8]
])
vector = np.array([2, -1, 6, 4])

# Розв'язок
solution = solve_system(matrix, vector)
print("Розв'язок СЛАР:", solution)