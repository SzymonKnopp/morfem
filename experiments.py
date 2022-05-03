import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import dia_array


def diagonal_heavy_matrix(size: int, max_abs_value: float, density: float = 0.5) -> dia_array:
    density = min(max(density, 0), 1)
    half_diag_prob = np.geomspace(1, 1 + density, size - 1)
    diag_probabilities = np.concatenate((half_diag_prob, [2], np.flip(half_diag_prob))) - np.ones(size * 2 - 1)
    diag_multiplier = diag_probabilities

    matrix = dia_array((size, size))
    for diag_index in range(size):
        if diag_probabilities[diag_index] > random.random():
            items_count = diag_index + 1
            diag_lower = np.random.uniform(-max_abs_value, max_abs_value, items_count) * np.array([diag_multiplier[diag_index]] * items_count)
            diag_upper = np.random.uniform(-max_abs_value, max_abs_value, items_count) * np.array([diag_multiplier[diag_index]] * items_count)
            matrix.setdiag(diag_lower, diag_index - size + 1)
            if diag_index != size - 1:  # don't add main diagonal twice
                matrix.setdiag(diag_upper, size - diag_index - 1)
    return matrix


# experiment based on electrodynamic analysis of discretized two-port element
N = 1000  # discrete element count
M = 2  # gate count
max_abs_value = 10
density = 0.02

mat_gamma = diagonal_heavy_matrix(N, max_abs_value, density)
mat_g = diagonal_heavy_matrix(N, max_abs_value, density)
mat_c = diagonal_heavy_matrix(N, max_abs_value, density)
mat_b = np.random.uniform(-max_abs_value, max_abs_value, (N, M))
mat_i = np.ones((M, M))
print("Test data generated")

s = 2.5
mat_system = mat_gamma + s * mat_g + s * s * mat_c
mat_impulse = s * mat_b @ mat_i
print("Equation matrices calculated")

fig, plots = plt.subplots(1, 2)
fig.set_figheight(10)
fig.set_figwidth(10)

lu, pivot = lu_factor(mat_system.toarray())
mat_e = lu_solve((lu, pivot), mat_impulse)
print("Equation solved")

plots[0].spy(mat_system.toarray())
plots[0].set_title("System matrix")

plots[1].spy(lu)
plots[1].set_title("LU")
print("LU factorisation performed")

fig.show()
print("Done")
