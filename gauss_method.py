import numpy as np

def load_matrix(file_path):
    """Завантажує матрицю з файлу"""
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                row = list(map(float, line.split()))
                matrix.append(row)
    return np.array(matrix)

def print_matrix(matrix, title="Matrix"):
    """Виводить матрицю з вертикальною рискою, якщо це розширена матриця"""
    print(f"\n{title}:")
    rows, cols = matrix.shape
    # Якщо кількість стовпців на 1 більша за кількість рядків, це розширена матриця
    is_augmented = cols == rows + 1
    for row in matrix:
        if is_augmented:
            # Вивід із розділенням основної частини і правого вектора
            print(' '.join(f"{val: .2f}" for val in row[:-1]) + " | " + f"{row[-1]: .2f}")
        else:
            # Звичайний вивід
            print(' '.join(f"{val: .2f}" for val in row))

def gaussian_elimination(matrix):
    """Метод Гаусса для розв'язання системи лінійних рівнянь"""
    size = matrix.shape[0]
    inverse_matrix = np.identity(size)
    determinant = 1.0
    solutions = np.zeros(size)

    # Forward elimination (прямий хід)
    for i in range(size):
        pivot = matrix[i, i]
        determinant *= pivot
        matrix[i] /= pivot
        inverse_matrix[i] /= pivot

        print_matrix(matrix, f"Після нормалізації рядка {i + 1}")
        print_matrix(inverse_matrix, "Обернена матриця після нормалізації")
        
        for j in range(i + 1, size):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            inverse_matrix[j] -= inverse_matrix[i] * factor

            print_matrix(matrix, f"Після усунення елемента {i + 1}, {j + 1}")
            print_matrix(inverse_matrix, "Обернена матриця після усунення елемента")

    # Backward substitution (зворотний хід)
    print("\nВиконується зворотне підстановлення...")
    solutions[size - 1] = matrix[size - 1, size]
    for i in range(size - 2, -1, -1):
        sum = 0
        for j in range(i + 1, size):
            sum += matrix[i, j] * solutions[j]
        solutions[i] = matrix[i, size] - sum

    print(f"\nРішення: {solutions}")

    # Reversing the process for inverse matrix (обчислення оберненої матриці)
    for i in range(size - 1, -1, -1):
        for j in range(i):
            factor = matrix[j, i]
            matrix[j] -= matrix[i] * factor
            inverse_matrix[j] -= inverse_matrix[i] * factor

    return solutions, determinant, inverse_matrix

def main():
    # Завантажуємо матрицю з файлу
    matrix = load_matrix("matrix_g.txt")
    size = matrix.shape[0]
    print_matrix(matrix, "Початкова матриця системи рівнянь")

    # Створюємо додаткову матрицю для вирішення
    matrix_augmented = np.hstack((matrix[:, :-1], matrix[:, -1][:, np.newaxis]))

    # Розв'язуємо систему рівнянь методом Гаусса
    solutions, determinant, inverse_matrix = gaussian_elimination(matrix_augmented)

    # Виводимо результат
    print("\nРішення системи рівнянь:")
    for i, sol in enumerate(solutions, 1):
        print(f"x{i} = {sol:.2f}")

    print(f"\nДетермінант матриці: {determinant:.2f}")

    print("\nОбернена матриця:")
    print_matrix(inverse_matrix, "Обернена матриця")

if __name__ == "__main__":
    main()
