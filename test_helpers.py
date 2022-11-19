import time
import numpy as np
from scipy.constants import pi, epsilon_0
from scipy.sparse import csc_array
from implementation import solve_finite_element_method, solve_finite_element_method_with_model_order_reduction


def generalized_scattering_matrix(frequency_point: float, e: csc_array, b: csc_array):
    gim = 1j * 2 * pi * frequency_point * epsilon_0 * e.T @ b  # 3.28
    gam = np.linalg.inv(gim)
    id = np.eye(gam.shape[0])
    gsm = 2 * np.linalg.inv(id + gam) - id
    return gsm


def equally_distributed_points(source: np.ndarray, amount: int):
    if amount > source.size:
        raise Exception("amount can't be greater than the number of points in the source")

    indices = np.linspace(0, source.size - 1, amount, dtype=int)
    return source[indices]


def finite_element_method_gsm(frequency_points, gate_count, in_c, in_gamma, in_b, in_kte1, in_kte2):
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    start = time.time()
    x_in_domain, b_in_domain = solve_finite_element_method(frequency_points, in_c, in_gamma, in_b, in_kte1, in_kte2)
    print("No MOR: ", time.time() - start, " s")

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_in_domain[i])

    return gsm_in_frequency


def finite_element_method_model_order_reduction_gsm(frequency_points, gate_count, in_c, in_gamma, in_b, in_kte1, in_kte2):
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    start = time.time()
    x_in_domain, b_reduced_in_domain = solve_finite_element_method_with_model_order_reduction(frequency_points, in_c, in_gamma, in_b, in_kte1, in_kte2)
    print("MOR: ", time.time() - start, " s")

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_reduced_in_domain[i])

    return gsm_in_frequency
