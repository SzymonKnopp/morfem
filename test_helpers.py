import time
import math
import numpy as np
from scipy.constants import pi, epsilon_0, c as c_lightspeed
from scipy.sparse import csc_array
from implementation import solve_finite_element_method, solve_finite_element_method_with_model_order_reduction


def wave_impedance_te(kcte, f0):
    mi0 = 4 * pi * 1e-7
    k0 = (2 * pi * f0) / c_lightspeed
    gte = 1 / math.sqrt(k0 ** 2 - kcte ** 2)
    if gte.imag != 0:  # kcte and f0 handled only as scalars, no vector values support
        gte = 1j * math.fabs(gte)
    return 2 * pi * f0 * mi0 * gte


def impulse_vector(frequency_point: float, in_b: csc_array, in_kte1: float, in_kte2: float):
    zte1 = wave_impedance_te(in_kte1, frequency_point)
    zte2 = wave_impedance_te(in_kte2, frequency_point)

    di = np.diag(np.sqrt([1 / zte1, 1 / zte2]))  # TODO: współczynnikiem przy B będzie macierz zawierająca zte1 i zte2, uwzględnić w estymatorze błędu!
    return in_b @ di


def system_matrix(frequency_point: float, in_c: csc_array, in_gamma: csc_array):
    k0 = (2 * pi * frequency_point) / c_lightspeed
    a_mat = in_c - k0 ** 2 * in_gamma
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?


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
