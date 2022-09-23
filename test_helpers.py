import numpy as np
from scipy.constants import pi, epsilon_0
from scipy.sparse import csc_array
from implementation import solve_finite_element_method, solve_finite_element_method_with_model_order_reduction


def generalized_scattering_matrix(frequency_point: float, e_mat: csc_array, b_mat_local: csc_array):
    gim = 1j * 2 * pi * frequency_point * epsilon_0 * e_mat.T @ b_mat_local  # 3.28
    gam = np.linalg.inv(gim)
    id = np.eye(gam.shape[0])
    gsm = 2 * np.linalg.inv(id + gam) - id
    return gsm


def equally_distributed_points(source: np.ndarray, amount: int):
    if amount > source.size:
        raise Exception("amount can't be greater than the number of points in the source")

    indices = np.linspace(0, source.size - 1, amount, dtype=int)
    return source[indices]


def finite_element_method_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2):
    gsm_of_frequency = np.empty([gate_count, gate_count, frequency_points.size], dtype=complex)

    b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method(
        frequency_points,
        gate_count, c_mat, gamma_mat, b_mat, kte1, kte2
    )

    for i in range(frequency_points.size):
        gsm_of_frequency[:, :, i] = generalized_scattering_matrix(frequency_points[i], e_mat_in_frequency[i], b_mat_in_frequency[i])

    return gsm_of_frequency


def finite_element_method_model_order_reduction_gsm(frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2):
    gsm_of_frequency = np.empty([gate_count, gate_count, frequency_points.size], dtype=complex)

    b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method_with_model_order_reduction(
        frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2
    )

    for i in range(frequency_points.size):
        gsm_of_frequency[:, :, i] = generalized_scattering_matrix(frequency_points[i], e_mat_in_frequency[i], b_mat_in_frequency[i])

    return gsm_of_frequency
