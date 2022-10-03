import math
import numpy as np
from scipy.constants import pi, epsilon_0, c
from scipy.sparse import csc_array
from implementation import solve_finite_element_method, solve_finite_element_method_with_model_order_reduction


def wave_impedance_te(kcte, f0):
    mi0 = 4 * pi * 1e-7
    k0 = (2 * pi * f0) / c
    gte = 1 / math.sqrt(k0 ** 2 - kcte ** 2)
    if gte.imag != 0:  # kcte and f0 handled only as scalars, no vector values support
        gte = 1j * math.fabs(gte)
    return 2 * pi * f0 * mi0 * gte


def impulse_vector(frequency_point: float, b_mat: csc_array, kte1: float, kte2: float):
    zte1 = wave_impedance_te(kte1, frequency_point)
    zte2 = wave_impedance_te(kte2, frequency_point)

    di = np.diag(np.sqrt([1 / zte1, 1 / zte2]))
    return b_mat @ di


def system_matrix(frequency_point: float, c_mat: csc_array, gamma_mat: csc_array):
    k0 = (2 * pi * frequency_point) / c
    a_mat = c_mat - k0 ** 2 * gamma_mat
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?


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
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    a_in_domain = [csc_array((c_mat.shape[0], c_mat.shape[1]))] * frequency_points.size
    b_in_domain = np.zeros([frequency_points.size, c_mat.shape[0], gate_count])
    for i in range(frequency_points.size):
        a_in_domain[i] = system_matrix(frequency_points[i], c_mat, gamma_mat)
        b_in_domain[i] = impulse_vector(frequency_points[i], b_mat, kte1, kte2)

    x_in_domain = solve_finite_element_method(a_in_domain, b_in_domain)

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_in_domain[i])

    return gsm_in_frequency


def finite_element_method_model_order_reduction_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2):
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    a_in_domain = [csc_array((c_mat.shape[0], c_mat.shape[1]))] * frequency_points.size
    b_in_domain = np.zeros([frequency_points.size, c_mat.shape[0], gate_count])
    for i in range(frequency_points.size):
        a_in_domain[i] = system_matrix(frequency_points[i], c_mat, gamma_mat)
        b_in_domain[i] = impulse_vector(frequency_points[i], b_mat, kte1, kte2)

    x_in_domain, b_reduced_in_domain = solve_finite_element_method_with_model_order_reduction(a_in_domain, b_in_domain, frequency_points)

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_reduced_in_domain[i])

    return gsm_in_frequency
