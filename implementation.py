import time
import math
import numpy as np
from typing import List
from numpy.linalg import norm
from scipy.sparse import csc_array, isspmatrix_csc
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

from move_to_test_helpers import wave_impedance_te, system_matrix, impulse_vector

ERROR_THRESHOLD = 1e-7
USE_EQUALLY_DISTRIBUTED = False
EQUALLY_DISTRIBUTED_REDUCTION_RATE = 0.95  # in range <0, 1)
PLOT_GREEDY_ITERATIONS = False
USE_OPM = False  # orthonormalize only new vectors in projection base, expand offline phase matrices


def solve_finite_element_method(domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, kte1, kte2):
    x_in_domain = np.zeros((domain.size, in_b.shape[0], in_b.shape[1]))
    for i in range(domain.size):
        x_in_domain[i] = solve_fem_point(domain[i], in_gamma, in_c, in_b, kte1, kte2)

    b_in_domain = np.zeros((domain.size, in_b.shape[0], in_b.shape[1]))
    for i in range(domain.size):
        b_in_domain[i] = impulse_vector(domain[i], in_b, kte1, kte2)

    return x_in_domain, b_in_domain


def solve_finite_element_method_with_model_order_reduction(domain: np.ndarray, in_c_reduced: csc_array, in_gamma_reduced: csc_array, in_b_reduced: csc_array, kte1, kte2):
    start = time.time()
    q = projection_base(domain, in_c_reduced, in_gamma_reduced, in_b_reduced, kte1, kte2) if not USE_EQUALLY_DISTRIBUTED else projection_base_equally_distributed(domain, in_c_reduced, in_gamma_reduced, in_b_reduced, kte1, kte2)
    print("Projection base: ", time.time() - start, " s")

    # reduce model order - 5.5
    q_t = q.T
    in_c_reduced = q_t @ in_c_reduced @ q
    in_gamma_reduced = q_t @ in_gamma_reduced @ q
    in_b_reduced = q_t @ in_b_reduced

    return solve_finite_element_method(domain, in_c_reduced, in_gamma_reduced, in_b_reduced, kte1, kte2)


def projection_base_equally_distributed(domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, kte1, kte2):
    reduction_indices = np.linspace(  # indices of points to be added to projection base Q
        0,
        domain.size - 1,
        math.floor(domain.size * (1 - EQUALLY_DISTRIBUTED_REDUCTION_RATE)),
        dtype=int
    )
    vector_count = in_b.shape[1]  # how many vectors does a single Ax = b solution contain
    q = np.empty((in_b.shape[0], vector_count * reduction_indices.size))

    for i in range(reduction_indices.size):
        q[:, vector_count*i:vector_count*i+vector_count] = solve_fem_point(domain[reduction_indices[i]], in_gamma, in_c, in_b, kte1, kte2)

    q = np.linalg.svd(q, full_matrices=False)[0]  # orthonormalize Q

    # error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)

    return q


def projection_base(domain: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, kte1, kte2):
    time_stats = TimeStatistics()
    time_stats.start_clock()
    whole_time_clock = time_stats.clock

    initial_vectors = np.hstack((  # starting points in projection base
        solve_fem_point(domain[0], in_gamma, in_c, in_b, kte1, kte2),
        solve_fem_point(domain[-1], in_gamma, in_c, in_b, kte1, kte2),
    ))
    q = np.linalg.svd(initial_vectors, full_matrices=False)[0]  # orthonormalize

    opm = OfflinePhaseMatrices()

    if USE_OPM:
        ch_c = h(in_c) @ in_c
        ch_g = h(in_c) @ in_gamma
        gh_c = h(in_gamma) @ in_c
        gh_g = h(in_gamma) @ in_gamma
        ch_b = h(in_c) @ in_b
        gh_b = h(in_gamma) @ in_b
        bh_c = h(in_b) @ in_c
        bh_g = h(in_b) @ in_gamma

        opm.qh_ch_c_q = h(q) @ ch_c @ q
        opm.qh_ch_g_q = h(q) @ ch_g @ q
        opm.qh_gh_c_q = h(q) @ gh_c @ q
        opm.qh_gh_g_q = h(q) @ gh_g @ q
        opm.qh_ch_b = h(q) @ ch_b
        opm.qh_gh_b = h(q) @ gh_b
        opm.bh_c_q = bh_c @ q
        opm.bh_g_q = bh_g @ q
        opm.bh_b = h(in_b) @ in_b

    error_in_iteration = np.empty((0, domain.size))

    time_stats.add_time("Before offline")

    while True:
        q_new, error = new_solution_for_projection_base(q, in_c, in_gamma, in_b, domain, kte1, kte2, opm, time_stats)
        error_in_iteration = np.vstack((error_in_iteration, error))
        if q_new is None:
            break

        if USE_OPM:
            q_new = orthonormalize_to_base(q_new, q)

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
        else:
            q = np.hstack((q, q_new))
            q = np.linalg.svd(q, full_matrices=False)[0]

        time_stats.add_time("Offline")

    time_stats.add_custom_time("Whole", whole_time_clock)
    time_stats.print_statistics()

    if PLOT_GREEDY_ITERATIONS:
        plt.title("Estymator błędu w iteracjach")
        plt.xlabel("Dziedzina")
        plt.ylabel("Błąd")
        for i in range(error_in_iteration.shape[0]):
            plt.semilogy(domain, error_in_iteration[i], label=f"iter {i}")
        plt.legend()
        plt.show()

    return q


def new_solution_for_projection_base(q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray, kte1, kte2, opm, time_stats):
    error = error_estimator(q, in_c, in_gamma, in_b, domain, kte1, kte2, opm, time_stats)
    idx_max = error.argmax()

    if error[idx_max] < ERROR_THRESHOLD:
        return None, error

    return solve_fem_point(domain[idx_max], in_gamma, in_c, in_b, kte1, kte2), error


def residual_norm(a_in_domain: List[csc_array], b_in_domain: np.ndarray, q: np.ndarray, domain: np.ndarray):
    residual_norm_in_domain = np.empty(b_in_domain.shape[0])
    for i in range(b_in_domain.shape[0]):
        a_reduced = q.T @ a_in_domain[i] @ q  # TODO: calculate q.T once and reuse?
        b_reduced = q.T @ b_in_domain[i]
        residual = a_in_domain[i] @ q @ solve_linear(a_reduced, b_reduced) - b_in_domain[i]
        residual_norm_in_domain[i] = norm(residual)

    return residual_norm_in_domain


def error_estimator(q: np.ndarray, in_c: csc_array, in_gamma: csc_array, in_b: csc_array, domain: np.ndarray, kte1, kte2, opm, time_stats):
    err_est_in_domain = np.empty(domain.size)

    if USE_OPM:
        qh_ch_c_q = opm.qh_ch_c_q
        qh_ch_g_q = opm.qh_ch_g_q
        qh_ch_b = opm.qh_ch_b
        qh_gh_c_q = opm.qh_gh_c_q
        qh_gh_g_q = opm.qh_gh_g_q
        qh_gh_b = opm.qh_gh_b
        bh_c_q = opm.bh_c_q
        bh_g_q = opm.bh_g_q
        bh_b = opm.bh_b
    else:
        # offline phase - calculate factors not determined by domain TODO: calculate and reuse things like qh_ch
        ch_c = h(in_c) @ in_c
        ch_g = h(in_c) @ in_gamma
        gh_c = h(in_gamma) @ in_c
        gh_g = h(in_gamma) @ in_gamma
        ch_b = h(in_c) @ in_b
        gh_b = h(in_gamma) @ in_b
        bh_c = h(in_b) @ in_c
        bh_g = h(in_b) @ in_gamma

        qh_ch_c_q = h(q) @ h(in_c) @ in_c @ q
        qh_ch_g_q = h(q) @ h(in_c) @ in_gamma @ q
        qh_ch_b = h(q) @ h(in_c) @ in_b
        qh_gh_c_q = h(q) @ h(in_gamma) @ in_c @ q
        qh_gh_g_q = h(q) @ h(in_gamma) @ in_gamma @ q
        qh_gh_b = h(q) @ h(in_gamma) @ in_b
        bh_c_q = h(in_b) @ in_c @ q
        bh_g_q = h(in_b) @ in_gamma @ q
        bh_b = h(in_b) @ in_b

    q_t = q.T
    in_c_reduced = q_t @ in_c @ q
    in_gamma_reduced = q_t @ in_gamma @ q
    in_b_reduced = q_t @ in_b

    time_stats.add_time("Offline")

    # online phase - sweep through domain
    for i in range(domain.size):
        t = domain[i]
        x = solve_fem_point(t, in_gamma_reduced, in_c_reduced, in_b_reduced, kte1, kte2)

        time_stats.add_time("Online - solve")

        err_est_in_domain[i] = norm(
            h(x) @ qh_ch_c_q @ x - t ** 2 * h(x) @ qh_ch_g_q @ x -
            factor_for_b(t) * h(x) @ qh_ch_b - t ** 2 * h(x) @ qh_gh_c_q @ x +
            t ** 4 * h(x) @ qh_gh_g_q @ x + factor_for_b(t) * t ** 2 * h(x) @ qh_gh_b -
            factor_for_b(t) * bh_c_q @ x + factor_for_b(t) * t ** 2 * bh_g_q @ x +
            factor_for_b(t) ** 2 * bh_b
        )

        time_stats.add_time("Online - add")

    if PLOT_GREEDY_ITERATIONS:
        plt.semilogy(domain, err_est_in_domain)
        plt.title("Error estimator in domain")
        plt.xlabel("Domain")
        plt.ylabel("Error estimator")
        plt.show()

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


def solve_linear(a: csc_array | np.ndarray, b: np.ndarray):
    """solves Ax = b"""
    if isspmatrix_csc(a):  # LU factorization - 3.27
        e_mat = splu(a).solve(b)
    else:
        lu = lu_factor(a)
        e_mat = lu_solve(lu, b)
    return e_mat


def solve_fem_point(t, gamma, c, b, kte1, kte2):
    return solve_linear(system_matrix(t, c, gamma), impulse_vector(t, b, kte1, kte2))


def h(array: np.ndarray | csc_array):
    """hermitian conjugate"""
    if array.ndim != 2:
        raise Exception("array has to be two-dimensional")

    return array.conj().T


def orthonormalize_to_base(vectors: np.ndarray, base: np.ndarray) -> np.ndarray:
    """expands base by new vectors, orthormalizes them to the existing base or creates a new one"""
    if vectors.ndim != 2:
        raise Exception("new_vectors has to be two-dimensional")
    if base is not None and base.ndim != 2:
        raise Exception("base has to be two-dimensional or None")

    orthonormalized_vectors = None

    for i in range(vectors.shape[1]):
        orthonormalized = orthonormalize_vector_to_base(vectors[:, i], base)
        base = np.hstack((base, orthonormalized.reshape((orthonormalized.size, 1))))
        if orthonormalized_vectors is None:
            orthonormalized_vectors = orthonormalized.reshape((orthonormalized.shape[0], 1))
        else:
            orthonormalized_vectors = np.hstack((orthonormalized_vectors, orthonormalized.reshape((orthonormalized.size, 1))))

    return orthonormalized_vectors


def orthonormalize_vector_to_base(vector: np.ndarray, base: np.ndarray) -> np.ndarray:
    """uses Gram-Schmidt process to orthonormalize input vector to the base of orthonormalized vectors"""
    if vector.ndim != 1 or base.ndim != 2:
        raise Exception("vector has to be one-dimensional and base has to be two-dimensional")

    def project(vector: np.ndarray, base_vector: np.ndarray):
        return base_vector * np.inner(vector, base_vector) # / np.inner(base_vector, base_vector) not necessary because vectors are normalized

    result = vector.copy()
    for i in range(base.shape[1]):
        result -= project(vector, base[:, i])

    return result / norm(result)


def factor_for_b(frequency: float):
    return math.sqrt(1 / wave_impedance_te(54.5976295582387, frequency))


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


class TimeStatistics:
    times = {"Whole": 0.0}
    clock = 0.0

    def start_clock(self):
        self.clock = time.time()

    def add_time(self, time_name: str):
        if time_name not in self.times:
            self.times[time_name] = 0.0

        self.times[time_name] += time.time() - self.clock
        self.clock = time.time()

    def add_custom_time(self, time_name: str, custom_clock: float):
        self.times[time_name] += time.time() - custom_clock

    def print_statistics(self):
        for time_name in self.times:
            time = self.times[time_name]
            print(f"{time_name}: {round(time, 2)} s | {round((time/self.times['Whole'])*100, 2)}%")
