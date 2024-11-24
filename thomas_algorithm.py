import numpy as np

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

# Перевірка на тридіагональність
def is_tridiagonal_matrix(matrix):
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if abs(i - j) > 1 and matrix[i, j] != 0:
                return False
    return True

# Розв'язання системи методом прогонки (метод Томаса)
def solve_tridiagonal(matrix):
    size = matrix.shape[0]
    if not is_tridiagonal_matrix(matrix[:, :-1]):
        print("Error: The matrix is not tridiagonal. This method only works for tridiagonal matrices.")
        return None

    alpha = np.zeros(size)
    beta = np.zeros(size)

    # Ініціалізація
    alpha[0] = -matrix[0, 1] / matrix[0, 0]
    beta[0] = matrix[0, -1] / matrix[0, 0]

    # Прямий хід
    for i in range(1, size):
        denominator = -matrix[i, i] - alpha[i - 1] * matrix[i, i - 1]
        alpha[i] = matrix[i, i + 1] / denominator if i < size - 1 else 0
        beta[i] = (matrix[i, i - 1] * beta[i - 1] - matrix[i, -1]) / denominator

    # Зворотній хід
    solutions = np.zeros(size)
    solutions[-1] = (matrix[-1, -2] * beta[-2] - matrix[-1, -1]) / (
        -matrix[-1, -1] - matrix[-1, -2] * alpha[-2]
    )

    for i in range(size - 2, -1, -1):
        solutions[i] = solutions[i + 1] * alpha[i] + beta[i]

    return solutions

# Основна функція
if __name__ == "__main__":
    file_name = "matrix_ta.txt"
    matrix = read_matrix_from_file(file_name)

    if matrix is not None:
        print("Input Matrix:")
        display_matrix(matrix)

        solutions = solve_tridiagonal(matrix)
        if solutions is not None:
            print("\nThe solution is:")
            for i, sol in enumerate(solutions, start=1):
                print(f"x{i} = {sol:.4f}")
