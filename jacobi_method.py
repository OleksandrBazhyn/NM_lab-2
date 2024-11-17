import numpy as np
from prettytable import PrettyTable

import numpy as np

# Implementation of the Jacobi method for solving a SLAE
def jacobi_method(A, b, tol=1e-5, max_iterations=100):
    n = len(b)
    x = np.zeros(n)  # Initial approximation (vector of zeros)
    x_new = np.zeros(n)
    
    for iteration in range(max_iterations):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        
        # Checking the stop condition
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, iteration + 1
        
        x = x_new.copy()
    
    return x, max_iterations

# Generation of a 4x4 random matrix and a vector of the right side
matrix = np.array([[-9, -2, -6, 1],
                   [4, -9, 1, 0],
                   [1, -1, 6, -1],
                   [-5, 2, -1, -8]])

vector = np.array([2, -1, 6, 4])

# Check and modify for diagonal dominance (if necessary)
def is_diagonally_dominant(mat):
    for i in range(mat.shape[0]):
        if abs(mat[i, i]) < sum(abs(mat[i, j]) for j in range(mat.shape[1]) if j != i):
            return False
    return True

while not is_diagonally_dominant(matrix):
    matrix = np.random.randint(-9, 10, (4, 4))

# Solution of the system by the Jacobi method
jacobi_solution, num_iterations = jacobi_method(matrix, vector, tol=1e-5)

# Output of results
print("Розв'язок методом Якобі:")
print("Вектор розв'язку:", jacobi_solution)
print("Кількість ітерацій:", num_iterations)
