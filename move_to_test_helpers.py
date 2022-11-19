import math
import numpy as np
from scipy.sparse import csc_array
from scipy.constants import pi, c as c_lightspeed


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
    a_mat = in_c - frequency_point ** 2 * in_gamma
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?