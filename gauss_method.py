import numpy as np

def load_matrix(file_path):
    try:
        with open(file_path, 'r') as file:
            matrix = []
            for line in file:
                if line.strip():
                    matrix.append(list(map(float, line.split())))
            return np.array(matrix)
    except FileNotFoundError:
        print(f"Помилка: Неможливо відкрити файл {file_path}")
        return None

def gaussian_elimination(matrix):
    size = len(matrix)
    augmented_matrix = matrix.copy()
    determinant = 1
    inverse_matrix = np.eye(size)

    for k in range(size):
        pivot = augmented_matrix[k, k]
        determinant *= pivot
        augmented_matrix[k] /= pivot
        inverse_matrix[k] /= pivot

        for i in range(k + 1, size):
            factor = augmented_matrix[i, k]
            augmented_matrix[i] -= augmented_matrix[k] * factor
            inverse_matrix[i] -= inverse_matrix[k] * factor

    solutions = np.zeros(size)
    solutions[-1] = augmented_matrix[-1, -1]  # Останній елемент правої частини

    for i in range(size - 2, -1, -1):
        solutions[i] = augmented_matrix[i, -1] - np.dot(
            augmented_matrix[i, i + 1:size], solutions[i + 1:]
        )

    for i in range(size - 1, 0, -1):
        for j in range(i):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= augmented_matrix[i] * factor
            inverse_matrix[j] -= inverse_matrix[i] * factor

    return solutions, determinant, inverse_matrix

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.6f}" for val in row))

def main():
    matrix = load_matrix("matrix_g.txt")
    if matrix is None:
        return

    size = len(matrix)
    if matrix.shape[1] != size + 1:
        print("Помилка: Невірна кількість стовпців у матриці.")
        return

    solutions, determinant, inverse_matrix = gaussian_elimination(matrix)

    print("\nThe solution to the system of equations is:")
    for i, solution in enumerate(solutions, start=1):
        print(f"x{i} = {solution:.6f}")

    print(f"\nThe determinant of the matrix is: {determinant:.6f}")

    print("\nThe inverse matrix is:")
    print_matrix(inverse_matrix)

if __name__ == "__main__":
    main()
