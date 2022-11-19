import math
from scipy.sparse import csc_array, issparse
from scipy.constants import pi, c as c_lightspeed


def system_matrix(frequency_point: float, in_c: csc_array, in_gamma: csc_array):
    a_mat = in_c - frequency_point ** 2 * in_gamma
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?


def impulse_vector(frequency_point: float, in_b: csc_array):
    b = b_coefficient(frequency_point) * in_b
    # TODO: allow B returned here to be sparse?
    return b.todense() if issparse(b) else b


def b_coefficient(t: float):
    kte = 54.5976295582387
    return math.sqrt(math.sqrt(((2 * pi * t) / c_lightspeed) ** 2 - kte ** 2) / t)
