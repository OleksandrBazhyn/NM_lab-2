import numpy as np
from prettytable import PrettyTable

def print_table(matrix, step):
    """Функція для друку матриці у вигляді таблиці."""
    n = matrix.shape[0]
    table = PrettyTable()
    headers = [f"x{i + 1}" for i in range(n)] + ["b"]
    table.field_names = headers
    for row in matrix:
        table.add_row([f"{x:.4f}" for x in row])
    print(f"\nКрок {step}:\n")
    print(table)

def gauss_elimination_verbose(matrix, vector):
    """
    Метод Гаусса для розв'язання СЛАР з виведенням деталей.
    matrix - матриця коефіцієнтів (n x n)
    vector - вектор правої частини (n)
    """
    n = len(vector)
    # Розширена матриця
    augmented_matrix = np.hstack((matrix, vector.reshape(-1, 1)))
    print("Початкова розширена матриця:")
    print_table(augmented_matrix, "Початок")

    # Прямий хід: приведення до трикутного вигляду
    for i in range(n):
        # Пошук головного елемента в стовпці
        max_row = i + np.argmax(abs(augmented_matrix[i:, i]))
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]  # Обмін рядків

        # Нормалізація головного елемента
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
        print_table(augmented_matrix, f"Прямий хід (крок {i + 1})")

    # Зворотний хід: знаходження розв'язку
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])) / augmented_matrix[i, i]
        print(f"\nРозрахунок x[{i + 1}]: {x[i]:.4f}")

    return x

# Вхідні дані
matrix = np.array([
    [-9, -2, -6, 1],
    [4, -9, 1, 0],
    [1, -1, 6, -1],
    [-5, 2, -1, -8]
], dtype=float)

vector = np.array([2, -1, 6, 4], dtype=float)

# Розв'язання методом Гаусса з детальним виведенням
solution = gauss_elimination_verbose(matrix, vector)
print("\nРозв'язок СЛАР методом Гаусса:", solution)
