import numpy as np
from prettytable import PrettyTable

# Читання матриці з файлу
def read_matrix_from_file(file_name):
    try:
        with open(file_name, 'r') as input_file:
            matrix = []
            for line in input_file:
                if line.strip():
                    row = list(map(float, line.split()))
                    matrix.append(row)
            return np.array(matrix)
    except FileNotFoundError:
        print(f"Error: Unable to open file {file_name}")
        return None
    
# Виведення матриці
def display_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{elem:.2f}" for elem in row))

def split_matrix(matrix):
    # Витягуємо квадратну матрицю A (всі стовпці, крім останнього)
    A = matrix[:, :-1]
    # Витягуємо вектор стовпчик b (останній стовпець)
    b = matrix[:, -1]
    return A, b

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

if __name__ == "__main__":
    file_name = "matrix_j.txt"
    matrix = read_matrix_from_file(file_name)

    if matrix is not None:
        print("Input Matrix:")
        display_matrix(matrix)
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

        # Перевірка правильності розв'язку
        A, b = split_matrix(matrix)