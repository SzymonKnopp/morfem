import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import dia_array


# funkcja generująca losową macierz o wartościach niezerowych skupionych wokół przekątnej
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


def mat_e_in_frequency_point(mat_gamma: dia_array, mat_g: dia_array, mat_c: dia_array, mat_b: np.ndarray, mat_i: np.ndarray, s: float) -> np.ndarray:
    mat_system = mat_gamma + s * mat_g + s * s * mat_c
    mat_impulse = s * mat_b @ mat_i

    # PYTANIE: Dlaczego korzystać z faktoryzacji LU? Macierz systemowa jest zależna od częstotliwości więc nie możemy
    # raz policzyć L i U i korzystać z nich za każdym razem. Czemu nie skorzystać z szybszej metody iteracyjnej?
    lu, pivot = lu_factor(mat_system.toarray())
    mat_e = lu_solve((lu, pivot), mat_impulse)
    print("Solved for E in s = ", s)
    return mat_e


def project(subject: np.array, subspace: np.array) -> np.ndarray:
    mat_projected = subspace.T @ subject @ subspace
    return mat_projected


# experiment based on electrodynamic analysis of discretized two-port element
if __name__ == "__main__":
    N = 1000  # discrete element count
    M = 2  # gate count
    max_abs_value = 10
    density = 0.02 # gęstość wygenerowanej macierzy
    frequency_points_to_solve = [3, 3.5, 4, 4.5, 5]

    # testy przeprowadzane są na wygenerowanych losowo macierzach zamiast charakterystykach rzeczywistego dwójnika
    # przez co wynik sweepu przez częstotliwość raczej nie da niczego przypominającego charakterystykę anteny
    # PYTANIE: Skąd wziąć macierze rzeczywistej anteny do testów? Internet niestety nie pomógł.
    mat_gamma = diagonal_heavy_matrix(N, max_abs_value, density)
    mat_g = diagonal_heavy_matrix(N, max_abs_value, density)
    mat_c = diagonal_heavy_matrix(N, max_abs_value, density)
    mat_b = np.random.uniform(-max_abs_value, max_abs_value, (N, M))
    mat_i = np.ones((M, M))  # zgodnie z tym co Pan powiedział, macierz natężeń składa się z jedynek
    print("Test data_csv generated")

    # składanie bazy projekcyjnej
    mat_q = None
    for s in frequency_points_to_solve:
        mat_e = mat_e_in_frequency_point(mat_gamma, mat_g, mat_c, mat_b, mat_i, s)
        mat_q = np.hstack((mat_q, mat_e)) if (mat_q is not None) else mat_e

    # orthonormalize Q
    mat_q = np.linalg.qr(mat_q)[0]

    # projekcja
    mat_gamma_r = project(mat_gamma, mat_q)
    mat_g_r = project(mat_g, mat_q)
    mat_c_r = project(mat_c, mat_q)
    mat_b_r = mat_q.T @ mat_b

    fig, plots = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    plots[0].spy(mat_gamma.toarray())
    plots[0].set_title("Original Gamma")

    plots[1].spy(mat_gamma_r)
    plots[1].set_title("Reduced Gamma")

    # przemiatanie częstotliwości
    for s in range(30, 50):
        s /= 10
        # mat_z_r = ... - Jak poprawnie obliczyć Generalized Impedance Matrix? Szczegóły pytania w mailu.
        # mat_y_r = ...
        # mat_s_r = ...

    fig.show()
    print("Done")
