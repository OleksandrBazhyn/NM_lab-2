import numpy as np
from prettytable import PrettyTable

# Модифікована функція для методу Якобі з виведенням проміжних результатів
def jacobi_method_verbose(A, b, tol=1e-5, max_iterations=100):
    n = len(b)
    x = np.zeros(n)  # Початкове наближення (вектор нулів)
    x_new = np.zeros(n)
    table = PrettyTable()
    table.field_names = ["Ітерація"] + [f"x_{i+1}" for i in range(n)] + ["Максимальна змінна"]
    
    for iteration in range(max_iterations):
        max_change = 0  # Для збереження максимальної зміни між ітераціями
        
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
            max_change = max(max_change, abs(x_new[i] - x[i]))
        
        # Додавання даних про ітерацію в таблицю
        table.add_row([iteration + 1] + [f"{val:.6f}" for val in x_new] + [f"{max_change:.6e}"])
        
        # Перевірка умови зупинки
        if max_change < tol:
            print(table)
            return x_new, iteration + 1
        
        x = x_new.copy()
    
    print(table)
    return x, max_iterations

# Матриця та вектор з попередніх обчислень
matrix = np.array([[-9, -2, -6, 1],
                   [4, -9, 1, 0],
                   [1, -1, 6, -1],
                   [-5, 2, -1, -8]])

vector = np.array([2, -1, 6, 4])

print("Розв'язок методом Якобі:")
tol_input = input("Будь ласка, введіть бажану точність обчислень: 10 в степені -")
tol = 10 ** (-float(tol_input))

# Виконання методу Якобі з детальним виведенням
jacobi_solution, num_iterations = jacobi_method_verbose(matrix, vector, tol)

print("Вектор розв'язку:", jacobi_solution)
print("Кількість ітерацій:", num_iterations)
