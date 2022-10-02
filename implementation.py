import math
from typing import List
import numpy as np
from numpy.linalg import norm
from scipy.constants import pi, c
from scipy.sparse import csc_array, isspmatrix_csc
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


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
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM


def e_in_frequency_point(frequency_point: float, c_mat: csc_array, gamma_mat: csc_array, b_mat: csc_array):
    a_mat = system_matrix(frequency_point, c_mat, gamma_mat)
    if isspmatrix_csc(a_mat):  # LU factorization - 3.27
        e_mat = splu(a_mat).solve(b_mat)
    else:
        lu = lu_factor(a_mat)
        e_mat = lu_solve(lu, b_mat)
    return e_mat


def projection_base(frequency_points: np.ndarray, reduction_points: List[float], c_mat: csc_array, gamma_mat: csc_array, b_mat: csc_array, kte1: float, kte2: float, gate_count: int, plot_residual=True):
    q_mat = np.empty((c_mat.shape[0], gate_count * len(reduction_points)))

    for i in range(len(reduction_points)):
        b_mat_local = impulse_vector(reduction_points[i], b_mat, kte1, kte2)
        q_mat[:, 2*i:2*i+2] = e_in_frequency_point(reduction_points[i], c_mat, gamma_mat, b_mat_local)

    q_mat = np.linalg.svd(q_mat, full_matrices=False)[0]  # orthonormalize base Q

    c_r_mat = q_mat.T @ c_mat @ q_mat  # TODO: calculate q_mat.T once and reuse?
    gamma_r_mat = q_mat.T @ gamma_mat @ q_mat
    b_r_mat = q_mat.T @ b_mat

    residual_in_frequency = np.empty(frequency_points.size)
    for i in range(frequency_points.size):
        a_mat = system_matrix(frequency_points[i], c_mat, gamma_mat)
        b_local_mat = impulse_vector(frequency_points[i], b_mat, kte1, kte2)
        residual = a_mat @ q_mat @ e_in_frequency_point(frequency_points[i], c_r_mat, gamma_r_mat, impulse_vector(frequency_points[i], b_r_mat, kte1, kte2)) - b_local_mat
        residual_in_frequency[i] = norm(residual)

    if plot_residual:
        plt.semilogy(frequency_points, residual_in_frequency)
        plt.title("Residual in frequency")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Residual")
        plt.show()

    return q_mat


def solve_finite_element_method_with_model_order_reduction(frequency_points: np.ndarray, reduction_points: List[float], gate_count: int, c_mat: csc_array, gamma_mat: csc_array, b_mat: csc_array, kte1: float, kte2: float):
    q_mat = projection_base(frequency_points, reduction_points, c_mat, gamma_mat, b_mat, kte1, kte2, gate_count)
    # reduce model order - 5.5
    c_r_mat = q_mat.T @ c_mat @ q_mat  # TODO: calculate q_mat.T once and reuse?
    gamma_r_mat = q_mat.T @ gamma_mat @ q_mat
    b_r_mat = q_mat.T @ b_mat

    b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method(frequency_points, gate_count, c_r_mat, gamma_r_mat, b_r_mat, kte1, kte2)

    return b_mat_in_frequency, e_mat_in_frequency


def solve_finite_element_method(frequency_points: np.ndarray, gate_count: int, c_mat: csc_array, gamma_mat: csc_array, b_mat: csc_array, kte1: float, kte2: float):
    b_mat_in_frequency: List[csc_array] = []  # TODO: memory pre-allocation?
    e_mat_in_frequency: List[csc_array] = []
    for i in range(frequency_points.size):
        b_mat_local = impulse_vector(frequency_points[i], b_mat, kte1, kte2)
        e_mat = e_in_frequency_point(frequency_points[i], c_mat, gamma_mat, b_mat_local)
        b_mat_in_frequency.append(b_mat_local)
        e_mat_in_frequency.append(e_mat)

    return b_mat_in_frequency, e_mat_in_frequency
