import math
import numpy as np
from typing import List
from numpy.linalg import norm
from scipy.constants import pi, c
from scipy.sparse import csc_array, isspmatrix_csc
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

ERROR_THRESHOLD = 1e-6


def solve_finite_element_method(a_in_domain: List[csc_array] | np.ndarray, b_in_domain: np.ndarray):
    x_in_domain = np.zeros(b_in_domain.shape)
    for i in range(x_in_domain.shape[0]):
        x = solve_linear(a_in_domain[i], b_in_domain[i])
        x_in_domain[i] = x

    return x_in_domain


def solve_finite_element_method_with_model_order_reduction(a_in_domain: List[csc_array], b_in_domain: np.ndarray, domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, reduction_rate=0.93):
    q = projection_base(a_in_domain, b_in_domain, reduction_rate, domain, in_c, in_gamma, in_b)

    # reduce model order - 5.5
    a_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], q.shape[1]])
    b_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], b_in_domain.shape[2]])
    for i in range(b_in_domain.shape[0]):
        a_reduced_in_domain[i] = q.T @ a_in_domain[i] @ q  # TODO: calculate q_mat.T once and reuse?
        b_reduced_in_domain[i] = q.T @ b_in_domain[i]

    x_in_domain = solve_finite_element_method(a_reduced_in_domain, b_reduced_in_domain)

    return x_in_domain, b_reduced_in_domain


def projection_base_equally_distributed(a_in_domain: List[csc_array], b_in_domain: np.ndarray, reduction_rate, domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array):
    if reduction_rate < 0 or reduction_rate >= 1:
        raise Exception("reduction rate must be in range <0, 1)")

    reduction_indices = np.linspace(  # indices of points to be added to projection base Q
        0,
        b_in_domain.shape[0] - 1,
        math.floor(b_in_domain.shape[0] * (1 - reduction_rate)),
        dtype=int
    )
    vector_count = b_in_domain.shape[2]  # how many vectors does a single Ax = b solution contain
    q = np.empty((b_in_domain.shape[1], vector_count * reduction_indices.size))

    for i in range(reduction_indices.size):
        a = a_in_domain[reduction_indices[i]]
        b = b_in_domain[reduction_indices[i]]
        q[:, vector_count*i:vector_count*i+vector_count] = solve_linear(a, b)

    q = np.linalg.svd(q, full_matrices=False)[0]  # orthonormalize Q

    error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)

    return q


def projection_base(a_in_domain: List[csc_array], b_in_domain: np.ndarray, reduction_rate, domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array):
    q = np.hstack((  # starting points in projection base
        solve_linear(a_in_domain[0], b_in_domain[0]),
        solve_linear(a_in_domain[-1], b_in_domain[-1])
    ))
    error_in_iteration = np.empty((0, b_in_domain.shape[0]))
    while True:
        q_new, error = new_solution_for_projection_base(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)
        error_in_iteration = np.vstack((error_in_iteration, error))
        if q_new is None:
            break
        q = np.hstack((q, q_new))
        q = np.linalg.svd(q, full_matrices=False)[0]  # orthonormalize Q

    plt.title("Error in iterations")
    plt.xlabel("Domain")
    plt.ylabel("Error")
    for i in range(error_in_iteration.shape[0]):
        plt.semilogy(domain, error_in_iteration[i], label=f"iter {i}")
    plt.legend()
    plt.show()

    return q


def new_solution_for_projection_base(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray):
    # error = residual_norm(a_in_domain, b_in_domain, q, domain)
    error = error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)
    idx_max = error.argmax()

    if error[idx_max] < ERROR_THRESHOLD:
        return None, error

    return solve_linear(a_in_domain[idx_max], b_in_domain[idx_max]), error


def residual_norm(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, domain: np.ndarray):
    residual_norm_in_domain = np.empty(b_in_domain.shape[0])
    for i in range(b_in_domain.shape[0]):
        a_reduced = q.T @ a_in_domain[i] @ q  # TODO: calculate q.T once and reuse?
        b_reduced = q.T @ b_in_domain[i]
        residual = a_in_domain[i] @ q @ solve_linear(a_reduced, b_reduced) - b_in_domain[i]
        residual_norm_in_domain[i] = norm(residual)

    return residual_norm_in_domain


def error_estimator(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray):
    err_est_in_domain = np.empty(b_in_domain.shape[0])

    # offline phase - calculate factors not determined by domain TODO: calculate and reuse things like qh_ch
    qh_ch_c_q = h(q) @ h(in_c) @ in_c @ q
    qh_ch_g_q = h(q) @ h(in_c) @ in_gamma @ q
    qh_ch_b = h(q) @ h(in_c) @ in_b
    qh_gh_c_q = h(q) @ h(in_gamma) @ in_c @ q
    qh_gh_g_q = h(q) @ h(in_gamma) @ in_gamma @ q
    qh_gh_b = h(q) @ h(in_gamma) @ in_b
    bh_c_q = h(in_b) @ in_c @ q
    bh_g_q = h(in_b) @ in_gamma @ q
    bh_b = h(in_b) @ in_b

    # online phase - sweep through domain
    for i in range(domain.size):
        a_reduced = q.T @ a_in_domain[i] @ q  # TODO: calculate q.T once and reuse?
        b_reduced = q.T @ b_in_domain[i]
        x = solve_linear(a_reduced, b_reduced)

        err_est_in_domain[i] = norm(
            h(x) @ qh_ch_c_q @ x - k(domain[i]) ** 2 * h(x) @ qh_ch_g_q @ x -
            factor_for_b(domain[i]) * h(x) @ qh_ch_b - k(domain[i]) ** 2 * h(x) @ qh_gh_c_q @ x +
            k(domain[i]) ** 4 * h(x) @ qh_gh_g_q @ x + factor_for_b(domain[i]) * k(domain[i]) ** 2 * h(x) @ qh_gh_b -
            factor_for_b(domain[i]) * bh_c_q @ x + factor_for_b(domain[i]) * k(domain[i]) ** 2 * bh_g_q @ x +
            factor_for_b(domain[i]) ** 2 * bh_b
        )

    plt.semilogy(domain, err_est_in_domain)
    plt.title("Error estimator in domain")
    plt.xlabel("Domain")
    plt.ylabel("Error estimator")
    plt.show()

    return err_est_in_domain


def solve_linear(a: csc_array | np.ndarray, b: np.ndarray):
    """solves Ax = b"""
    if isspmatrix_csc(a):  # LU factorization - 3.27
        e_mat = splu(a).solve(b)
    else:
        lu = lu_factor(a)
        e_mat = lu_solve(lu, b)
    return e_mat


def h(array: np.ndarray | csc_array):
    """hermitian conjugate"""
    if array.ndim != 2:
        raise Exception("array has to be two-dimensional")

    return array.conj().T


def k(frequency: float):
    return (2 * pi * frequency) / c


def factor_for_b(frequency: float):
    return math.sqrt(1 / wave_impedance_te(54.5976295582387, frequency))


def wave_impedance_te(kcte, f0):
    mi0 = 4 * pi * 1e-7
    k0 = (2 * pi * f0) / c
    gte = 1 / math.sqrt(k0 ** 2 - kcte ** 2)
    if gte.imag != 0:  # kcte and f0 handled only as scalars, no vector values support
        gte = 1j * math.fabs(gte)
    return 2 * pi * f0 * mi0 * gte
