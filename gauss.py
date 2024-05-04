import numpy as np

def gauss_seidel(A, b, tolerance=1e-10, max_iterations=1000):
    x = np.array([2, 2], dtype=np.double)
    for k in range(max_iterations):
        print(x)
        x_old = x.copy()

        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x_old[i + 1:])) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            break

    return x

# Define the matrix A and vector b
A = np.array([[3, 2], [1, 4]])
b = np.array([18, 14])



# Solve the system
x = gauss_seidel(A, b)

print("Solution:", x)
