import time
import math
import numpy as np
from typing import List
from numpy.linalg import norm
from scipy.constants import pi, c as c_lightspeed
from scipy.sparse import csc_array, isspmatrix_csc
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

ERROR_THRESHOLD = 1e-4


def solve_finite_element_method(a_in_domain: List[csc_array] | np.ndarray, b_in_domain: np.ndarray):
    x_in_domain = np.zeros(b_in_domain.shape)
    for i in range(x_in_domain.shape[0]):
        x = solve_linear(a_in_domain[i], b_in_domain[i])
        x_in_domain[i] = x

    return x_in_domain


def solve_finite_element_method_with_model_order_reduction(a_in_domain: List[csc_array], b_in_domain: np.ndarray, domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, kte1, kte2, reduction_rate=0.993):
    start = time.time()
    q = projection_base(a_in_domain, b_in_domain, reduction_rate, domain, in_c, in_gamma, in_b, kte1, kte2)
    print("Projection base: ", time.time() - start, " s")

    # reduce model order - 5.5
    a_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], q.shape[1]])
    b_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], b_in_domain.shape[2]])

    start = time.time()
    for i in range(b_in_domain.shape[0]):
        a_reduced_in_domain[i] = q.T @ a_in_domain[i] @ q  # TODO: calculate q_mat.T once and reuse?
        b_reduced_in_domain[i] = q.T @ b_in_domain[i]
    print("A and B pre-calculation: ", time.time() - start, " s")

    start = time.time()
    x_in_domain = solve_finite_element_method(a_reduced_in_domain, b_reduced_in_domain)
    print("FEM: ", time.time() - start, " s")

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

    # error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)

    return q


def projection_base(a_in_domain: List[csc_array], b_in_domain: np.ndarray, reduction_rate, domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, kte1, kte2):
    start = time.time()
    q = np.hstack((  # starting points in projection base
        solve_linear(a_in_domain[0], b_in_domain[0]),
        solve_linear(a_in_domain[-1], b_in_domain[-1])
    ))

    ch_c = h(in_c) @ in_c
    ch_g = h(in_c) @ in_gamma
    gh_c = h(in_gamma) @ in_c
    gh_g = h(in_gamma) @ in_gamma
    ch_b = h(in_c) @ in_b
    gh_b = h(in_gamma) @ in_b
    bh_c = h(in_b) @ in_c
    bh_g = h(in_b) @ in_gamma

    opm = OfflinePhaseMatrices()
    opm.qh_ch_c_q = h(q) @ ch_c @ q
    opm.qh_ch_g_q = h(q) @ ch_g @ q
    opm.qh_gh_c_q = h(q) @ gh_c @ q
    opm.qh_gh_g_q = h(q) @ gh_g @ q
    opm.qh_ch_b = h(q) @ ch_b
    opm.qh_gh_b = h(q) @ gh_b
    opm.bh_c_q = bh_c @ q
    opm.bh_g_q = bh_g @ q
    opm.bh_b = h(in_b) @ in_b

    print("Before offline: ", time.time() - start, " s")

    error_in_iteration = np.empty((0, b_in_domain.shape[0]))
    while True:
        q_new, error = new_solution_for_projection_base(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain, kte1, kte2, opm)
        error_in_iteration = np.vstack((error_in_iteration, error))
        if q_new is None:
            break

        start = time.time()
        # expand offline phase matrices
        opm.qh_ch_c_q = expand_matrix(opm.qh_ch_c_q, q, ch_c, q_new)
        opm.qh_ch_g_q = expand_matrix(opm.qh_ch_g_q, q, ch_g, q_new)
        opm.qh_gh_c_q = expand_matrix(opm.qh_gh_c_q, q, gh_c, q_new)
        opm.qh_gh_g_q = expand_matrix(opm.qh_gh_g_q, q, gh_g, q_new)
        opm.qh_ch_b = np.vstack((opm.qh_ch_b, h(q_new) @ ch_b))
        opm.qh_gh_b = np.vstack((opm.qh_gh_b, h(q_new) @ gh_b))
        opm.bh_c_q = np.hstack((opm.bh_c_q, bh_c @ q_new))
        opm.bh_g_q = np.hstack((opm.bh_g_q, bh_g @ q_new))

        q = np.hstack((q, q_new))
        print("Offline: ", time.time() - start, " s")

    q = np.linalg.svd(q, full_matrices=False)[0]  # orthonormalize Q

    plt.title("Estymator błędu w iteracjach")
    plt.xlabel("Dziedzina")
    plt.ylabel("Błąd")
    for i in range(error_in_iteration.shape[0]):
        plt.semilogy(domain, error_in_iteration[i], label=f"iter {i}")
    plt.legend()
    plt.show()

    return q


def new_solution_for_projection_base(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray, kte1, kte2, opm):
    # error = residual_norm(a_in_domain, b_in_domain, q, domain)
    error = error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain, kte1, kte2, opm)
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


def error_estimator(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray, kte1, kte2, opm):
    err_est_in_domain = np.empty(b_in_domain.shape[0])

    start = time.time()
    in_c_reduced = q.T @ in_c @ q  # TODO: calculate q.T once and reuse?
    in_gamma_reduced = q.T @ in_gamma @ q
    in_b_reduced = q.T @ in_b
    print("Reduce C, Gamma, B: ", time.time() - start, " s")

    solve_time = 0
    add_time = 0

    # online phase - sweep through domain
    for i in range(domain.size):
        start = time.time()
        a = system_matrix(domain[i], in_c_reduced, in_gamma_reduced)
        b = impulse_vector(domain[i], in_b_reduced, kte1, kte2)
        x = solve_linear(a, b)
        solve_time += time.time() - start

        start = time.time()
        err_est_in_domain[i] = norm(
            h(x) @ opm.qh_ch_c_q @ x - k(domain[i]) ** 2 * h(x) @ opm.qh_ch_g_q @ x -
            factor_for_b(domain[i]) * h(x) @ opm.qh_ch_b - k(domain[i]) ** 2 * h(x) @ opm.qh_gh_c_q @ x +
            k(domain[i]) ** 4 * h(x) @ opm.qh_gh_g_q @ x + factor_for_b(domain[i]) * k(domain[i]) ** 2 * h(x) @ opm.qh_gh_b -
            factor_for_b(domain[i]) * opm.bh_c_q @ x + factor_for_b(domain[i]) * k(domain[i]) ** 2 * opm.bh_g_q @ x +
            factor_for_b(domain[i]) ** 2 * opm.bh_b
        )
        add_time += time.time() - start

    print("Online - solve: ", solve_time, " s")
    print("Online - add: ", add_time, " s")

    # plt.semilogy(domain, err_est_in_domain)
    # plt.title("Error estimator in domain")
    # plt.xlabel("Domain")
    # plt.ylabel("Error estimator")
    # plt.show()

    return err_est_in_domain


def error_estimator_manual_offline_calculation(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray, opm):
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
    # plt.show()

    return err_est_in_domain


def expand_matrix(original: np.ndarray, old_q: np.ndarray, middle: np.ndarray, new_part_q: np.ndarray):
    """calculates new reduced part of matrix and reconstructs the original without recalculating it"""
    top_left = original
    top_right = h(old_q) @ middle @ new_part_q
    bottom_left = h(new_part_q) @ middle @ old_q
    bottom_right = h(new_part_q) @ middle @ new_part_q

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))

    return np.vstack((top, bottom))


def impulse_vector(frequency_point: float, in_b: csc_array, in_kte1: float, in_kte2: float):
    zte1 = wave_impedance_te(in_kte1, frequency_point)
    zte2 = wave_impedance_te(in_kte2, frequency_point)

    di = np.diag(np.sqrt([1 / zte1, 1 / zte2]))  # TODO: współczynnikiem przy B będzie macierz zawierająca zte1 i zte2, uwzględnić w estymatorze błędu!
    return in_b @ di


def system_matrix(frequency_point: float, in_c: csc_array, in_gamma: csc_array):
    k0 = (2 * pi * frequency_point) / c_lightspeed
    a_mat = in_c - k0 ** 2 * in_gamma
    return (a_mat + a_mat.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?


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
    return (2 * pi * frequency) / c_lightspeed


def factor_for_b(frequency: float):
    return math.sqrt(1 / wave_impedance_te(54.5976295582387, frequency))


def wave_impedance_te(kcte, f0):
    mi0 = 4 * pi * 1e-7
    k0 = (2 * pi * f0) / c_lightspeed
    gte = 1 / math.sqrt(k0 ** 2 - kcte ** 2)
    if gte.imag != 0:  # kcte and f0 handled only as scalars, no vector values support
        gte = 1j * math.fabs(gte)
    return 2 * pi * f0 * mi0 * gte


class OfflinePhaseMatrices:
    qh_ch_c_q = None
    qh_ch_g_q = None
    qh_ch_b = None
    qh_gh_c_q = None
    qh_gh_g_q = None
    qh_gh_b = None
    bh_c_q = None
    bh_g_q = None
    bh_b = None
