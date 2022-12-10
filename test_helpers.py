import time
import math
import numpy as np
from scipy.constants import pi, epsilon_0, c as c_lightspeed
from scipy.sparse import csc_array
from implementation import morfem, solve_finite_element_method, ModelDefinition


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


def finite_element_method_gsm(frequency_points, gate_count, in_c, in_gamma, in_b):
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    md = ModelDefinition(
        frequency_points,
        in_c,
        csc_array(in_c.shape),
        in_gamma,
        in_b,
        lambda t: 1,
        lambda t: t,
        lambda t: t ** 2,
        lambda t: b_coefficient(t)
    )
    start = time.time()
    x_in_domain, b_in_domain = solve_finite_element_method(md)
    print("No MOR: ", time.time() - start, " s")

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_in_domain[i])

    return gsm_in_frequency


def finite_element_method_model_order_reduction_gsm(frequency_points, gate_count, in_c, in_gamma, in_b):
    gsm_in_frequency = np.zeros([frequency_points.size, gate_count, gate_count], dtype=complex)

    start = time.time()
    x_in_domain, b_reduced_in_domain = morfem(frequency_points, in_c, csc_array(in_c.shape), in_gamma, in_b, t_b=lambda t: b_coefficient(t))
    print("MOR: ", time.time() - start, " s")

    for i in range(frequency_points.size):
        gsm_in_frequency[i] = generalized_scattering_matrix(frequency_points[i], x_in_domain[i], b_reduced_in_domain[i])

    return gsm_in_frequency


def b_coefficient(t: float):
    kte = 54.5976295582387
    return math.sqrt(math.sqrt(((2 * pi * t) / c_lightspeed) ** 2 - kte ** 2) / t)
