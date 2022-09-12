import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.constants import pi, epsilon_0
from scipy.sparse import csc_array
from implementation import solve_finite_element_method_with_model_order_reduction, solve_finite_element_method


def generalized_scattering_matrix(frequency_point: float, e_mat: csc_array, b_mat_local: csc_array):
    gim = 1j * 2 * pi * frequency_point * epsilon_0 * e_mat.T @ b_mat_local  # 3.28
    gam = np.linalg.inv(gim)
    id = np.eye(gam.shape[0])
    gsm = 2 * np.linalg.inv(id + gam) - id
    return gsm


def every_nth_value(array, n: int):
    result = []
    anchor = 0
    while anchor < len(array):
        result.append(array[anchor])
        anchor += n
    return result


def reference_solution(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2):
    gsm_of_frequency = np.empty([gate_count, gate_count, frequency_points.size], dtype=complex)

    b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method(
        frequency_points,
        gate_count, c_mat, gamma_mat, b_mat, kte1, kte2
    )

    for i in range(frequency_points.size):
        gsm_of_frequency[:, :, i] = generalized_scattering_matrix(frequency_points[i], e_mat_in_frequency[i], b_mat_in_frequency[i])

    return gsm_of_frequency


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    ref_gsm = reference_solution(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    reduction_points_amount = []
    execution_time = []
    avg_error = []
    for reduction_points_distance in range(1, 41):
        gsm_of_frequency = np.empty([gate_count, gate_count, frequency_points.size], dtype=complex)

        start = time.time()

        reduction_points = every_nth_value(frequency_points, reduction_points_distance)
        b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method_with_model_order_reduction(
            frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2
        )

        error = np.empty(frequency_points.size)
        for i in range(frequency_points.size):
            gsm_of_frequency[:, :, i] = generalized_scattering_matrix(frequency_points[i], e_mat_in_frequency[i], b_mat_in_frequency[i])
            error[i] = (norm(gsm_of_frequency[:, :, i] - ref_gsm[:, :, i]))

        reduction_points_amount.append(len(reduction_points))
        execution_time.append(time.time() - start)
        avg_error.append(error.mean())

        print("Solved for reduction_points_distance = ", reduction_points_distance)

    plt.plot(reduction_points_amount, execution_time)
    plt.semilogy(reduction_points_amount, avg_error)
    plt.legend(["Execution time", "Average error"])
    plt.title("TITLE AAAAA")
    plt.xlabel("Number of reduction points")
    plt.ylabel("that gonna be complicated, two scales")
    plt.show()
    print("Done")

# w1 A + w2 * A2 + w3  A3 = w4 * B <- odwrócony interfejs
