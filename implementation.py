import time
import math
import numpy as np
from typing import Callable
from copy import deepcopy
from numpy.linalg import norm
from scipy.sparse import csc_array, issparse
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

ERROR_THRESHOLD = 1e-6
USE_EQUALLY_DISTRIBUTED = False
EQUALLY_DISTRIBUTED_REDUCTION_RATE = 0.97  # in range <0, 1)
PLOT_GREEDY_ITERATIONS = False
USE_OPM = False  # orthonormalize only new vectors in projection base, expand offline phase matrices


class ModelDefinition:
    # array of t that the model is to be solved for
    domain: np.ndarray = None
    # addends of a system matrix
    a0: csc_array = None
    a1: csc_array = None
    a2: csc_array = None
    # impulse vector
    b: csc_array = None
    # t-coefficients of matrices
    t_a0: Callable[[float], float]
    t_a1: Callable[[float], float]
    t_a2: Callable[[float], float]
    t_b: Callable[[float], float]

    def __init__(
            self,
            domain: np.ndarray,
            a0: csc_array,
            a1: csc_array,
            a2: csc_array,
            b: csc_array,
            t_a0: Callable[[float], float],
            t_a1: Callable[[float], float],
            t_a2: Callable[[float], float],
            t_b: Callable[[float], float]
    ):
        self.domain = domain
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.t_a0 = t_a0
        self.t_a1 = t_a1
        self.t_a2 = t_a2
        self.t_b = t_b


class OfflinePhaseMatrices:
    qh_a0h_a0_q = None
    qh_a0h_a1_q = None
    qh_a0h_a2_q = None
    qh_a0h_b = None
    qh_a1h_a0_q = None
    qh_a1h_a1_q = None
    qh_a1h_a2_q = None
    qh_a1h_b = None
    qh_a2h_a0_q = None
    qh_a2h_a1_q = None
    qh_a2h_a2_q = None
    qh_a2h_b = None
    bh_a0_q = None
    bh_a1_q = None
    bh_a2_q = None
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


def morfem(
        domain: np.ndarray,
        a0: csc_array,
        a1: csc_array,
        a2: csc_array,
        b: csc_array,
        t_a0: Callable[[float], float] = lambda t: 1.,
        t_a1: Callable[[float], float] = lambda t: t,
        t_a2: Callable[[float], float] = lambda t: t ** 2,
        t_b: Callable[[float], float] = lambda t: t,
                ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Quickly solves finite element method problem defined as ``(t_a0*a0 + t_a1*a1 + t_a2*a2)x = t_b*b``

    Parameters
    ----------
    domain : array
        Ordered set of points that the problem should be solved for.
    a0, a1, a2, b : two-dimensional arrays
        Matrices being part of definition of the problem.
    t_a0, t_a1, t_a2, t_b : (`float`) -> `float`
        Functions accepting domain point and returning specific coefficients used in the definition of the problem.

    Returns
    -------


    """

    md = ModelDefinition(domain, a0, a1, a2, b, t_a0, t_a1, t_a2, t_b)

    start = time.time()
    q = projection_base(md) if not USE_EQUALLY_DISTRIBUTED else projection_base_equally_distributed(md)
    print("Projection base: ", time.time() - start, " s")

    # reduce model order - 5.5
    md_r = md
    q_t = q.T
    md_r.a0 = q_t @ md.a0 @ q
    md_r.a1 = q_t @ md.a1 @ q
    md_r.a2 = q_t @ md.a2 @ q
    md_r.b = q_t @ md.b

    return solve_finite_element_method(md_r), q, md_r.a0, md_r.a1, md_r.a2, md_r.b


def solve_finite_element_method(md: ModelDefinition):
    x_in_domain = np.zeros((md.domain.size, md.b.shape[0], md.b.shape[1]))
    for i in range(md.domain.size):
        x_in_domain[i] = solve_fem_point(md.domain[i], md)

    return x_in_domain


def projection_base_equally_distributed(md: ModelDefinition):
    reduction_indices = np.linspace(  # indices of points to be added to projection base Q
        0,
        md.domain.size - 1,
        math.floor(md.domain.size * (1 - EQUALLY_DISTRIBUTED_REDUCTION_RATE)),
        dtype=int
    )
    vector_count = md.b.shape[1]  # how many vectors does a single Ax = b solution contain
    q = np.empty((md.b.shape[0], vector_count * reduction_indices.size))

    for i in range(reduction_indices.size):
        q[:, vector_count*i:vector_count*i+vector_count] = solve_fem_point(md.domain[reduction_indices[i]], md)

    q = np.linalg.svd(q, full_matrices=False)[0]  # orthonormalize Q

    # error_estimator(a_in_domain, b_in_domain, q, in_c, in_gamma, in_b, domain)

    return q


def projection_base(md: ModelDefinition):
    time_stats = TimeStatistics()
    time_stats.start_clock()
    whole_time_clock = time_stats.clock

    initial_vectors = np.hstack((  # starting points in projection base
        solve_fem_point(md.domain[0], md),
        solve_fem_point(md.domain[-1], md),
    ))
    q = np.linalg.svd(initial_vectors, full_matrices=False)[0]  # orthonormalize

    opm = OfflinePhaseMatrices()

    if USE_OPM:
        a0h_a0 = h(md.a0) @ md.a0
        a0h_a1 = h(md.a0) @ md.a1
        a0h_a2 = h(md.a0) @ md.a2
        a0h_b = h(md.a0) @ md.b
        a1h_a0 = h(md.a1) @ md.a0
        a1h_a1 = h(md.a1) @ md.a1
        a1h_a2 = h(md.a1) @ md.a2
        a1h_b = h(md.a1) @ md.b
        a2h_a0 = h(md.a2) @ md.a0
        a2h_a1 = h(md.a2) @ md.a1
        a2h_a2 = h(md.a2) @ md.a2
        a2h_b = h(md.a2) @ md.b
        bh_a0 = h(md.b) @ md.a0
        bh_a1 = h(md.b) @ md.a1
        bh_a2 = h(md.b) @ md.a2
        bh_b = h(md.b) @ md.b

        opm.qh_a0h_a0_q = h(q) @ a0h_a0 @ q
        opm.qh_a0h_a1_q = h(q) @ a0h_a1 @ q
        opm.qh_a0h_a2_q = h(q) @ a0h_a2 @ q
        opm.qh_a0h_b = h(q) @ a0h_b
        opm.qh_a1h_a0_q = h(q) @ a1h_a0 @ q
        opm.qh_a1h_a1_q = h(q) @ a1h_a1 @ q
        opm.qh_a1h_a2_q = h(q) @ a1h_a2 @ q
        opm.qh_a1h_b = h(q) @ a1h_b
        opm.qh_a2h_a0_q = h(q) @ a2h_a0 @ q
        opm.qh_a2h_a1_q = h(q) @ a2h_a1 @ q
        opm.qh_a2h_a2_q = h(q) @ a2h_a2 @ q
        opm.qh_a2h_b = h(q) @ a2h_b
        opm.bh_a0_q = bh_a0 @ q
        opm.bh_a1_q = bh_a1 @ q
        opm.bh_a2_q = bh_a2 @ q
        opm.bh_b = bh_b

    error_in_iteration = np.empty((0, md.domain.size))

    time_stats.add_time("Before offline")

    while True:
        q_new, error = new_solution_for_projection_base(md, q, opm, time_stats)
        error_in_iteration = np.vstack((error_in_iteration, error))
        if q_new is None:
            break

        if USE_OPM:
            q_new = orthonormalize_to_base(q_new, q)

            # expand offline phase matrices
            opm.qh_a0h_a0_q = expand_matrix(opm.qh_a0h_a0_q, q, a0h_a0, q_new)
            opm.qh_a0h_a1_q = expand_matrix(opm.qh_a0h_a1_q, q, a0h_a1, q_new)
            opm.qh_a0h_a2_q = expand_matrix(opm.qh_a0h_a2_q, q, a0h_a2, q_new)
            opm.qh_a0h_b = np.vstack((opm.qh_a0h_b, h(q_new) @ a0h_b))
            opm.qh_a1h_a0_q = expand_matrix(opm.qh_a1h_a0_q, q, a1h_a0, q_new)
            opm.qh_a1h_a1_q = expand_matrix(opm.qh_a1h_a1_q, q, a1h_a1, q_new)
            opm.qh_a1h_a2_q = expand_matrix(opm.qh_a1h_a2_q, q, a1h_a2, q_new)
            opm.qh_a1h_b = np.vstack((opm.qh_a1h_b, h(q_new) @ a1h_b))
            opm.qh_a2h_a0_q = expand_matrix(opm.qh_a2h_a0_q, q, a2h_a0, q_new)
            opm.qh_a2h_a1_q = expand_matrix(opm.qh_a2h_a1_q, q, a2h_a1, q_new)
            opm.qh_a2h_a2_q = expand_matrix(opm.qh_a2h_a2_q, q, a2h_a2, q_new)
            opm.qh_a2h_b = np.vstack((opm.qh_a2h_b, h(q_new) @ a2h_b))
            opm.bh_a0_q = np.hstack((opm.bh_a0_q , bh_a0 @ q_new))
            opm.bh_a1_q = np.hstack((opm.bh_a1_q , bh_a1 @ q_new))
            opm.bh_a2_q = np.hstack((opm.bh_a2_q , bh_a2 @ q_new))

            q = np.hstack((q, q_new))
        else:
            q = np.hstack((q, q_new))
            q = np.linalg.svd(q, full_matrices=False)[0]

        time_stats.add_time("Offline")

    time_stats.add_custom_time("Whole", whole_time_clock)
    time_stats.print_statistics()

    if PLOT_GREEDY_ITERATIONS:
        colors = ["orange", "crimson", "forestgreen", "royalblue", "black", "orchid"]
        linestyle = ["solid", "dashed", "dashdot", "dotted"]
        for i in range(error_in_iteration.shape[0]):
            plt.semilogy(md.domain, error_in_iteration[i], color=colors[i % len(colors)], linestyle=linestyle[i % len(linestyle)], label=fr"$j={i}$")
        plt.xlabel(r"$t$ [Hz]")
        plt.ylabel(r"$err_e$ [dB]")
        # plt.legend(loc="upper left")
        plt.legend()
        plt.grid()
        plt.savefig("output/greedy_iterations.png", bbox_inches="tight")
        plt.show()

    return q


def new_solution_for_projection_base(md: ModelDefinition, q: np.ndarray, opm, time_stats):
    error = error_estimator(md, q, opm, time_stats)
    idx_max = error.argmax()

    if error[idx_max] < ERROR_THRESHOLD:
        return None, error

    return solve_fem_point(md.domain[idx_max], md), error


def residual_norm(md: ModelDefinition, q: np.ndarray):
    residual_norm_in_domain = np.empty(md.domain.size)

    q_t = q.T
    md_r = deepcopy(md)
    md_r.a0 = q_t @ md.a0 @ q
    md_r.a1 = q_t @ md.a1 @ q
    md_r.a2 = q_t @ md.a2 @ q
    md_r.b = q_t @ md.b

    for i in range(md.domain.size):
        residual = system_matrix(md.domain[i], md) @ q @ solve_fem_point(md_r.domain[i], md_r) - impulse_vector(md.domain[i], md)
        residual_norm_in_domain[i] = norm(residual)

    return residual_norm_in_domain


def error_estimator(md: ModelDefinition, q: np.ndarray, opm: OfflinePhaseMatrices, time_stats: TimeStatistics):
    err_est_in_domain = np.empty(md.domain.size)

    if USE_OPM:
        qh_a0h_a0_q = opm.qh_a0h_a0_q
        qh_a0h_a1_q = opm.qh_a0h_a1_q
        qh_a0h_a2_q = opm.qh_a0h_a2_q
        qh_a0h_b = opm.qh_a0h_b
        qh_a1h_a0_q = opm.qh_a1h_a0_q
        qh_a1h_a1_q = opm.qh_a1h_a1_q
        qh_a1h_a2_q = opm.qh_a1h_a2_q
        qh_a1h_b = opm.qh_a1h_b
        qh_a2h_a0_q = opm.qh_a2h_a0_q
        qh_a2h_a1_q = opm.qh_a2h_a1_q
        qh_a2h_a2_q = opm.qh_a2h_a2_q
        qh_a2h_b = opm.qh_a2h_b
        bh_a0_q = opm.bh_a0_q
        bh_a1_q = opm.bh_a1_q
        bh_a2_q = opm.bh_a2_q
        bh_b = opm.bh_b
    else:
        # offline phase - calculate factors not determined by domain TODO: calculate and reuse things like qh_ch
        a0h_a0 = h(md.a0) @ md.a0
        a0h_a1 = h(md.a0) @ md.a1
        a0h_a2 = h(md.a0) @ md.a2
        a0h_b = h(md.a0) @ md.b
        a1h_a0 = h(md.a1) @ md.a0
        a1h_a1 = h(md.a1) @ md.a1
        a1h_a2 = h(md.a1) @ md.a2
        a1h_b = h(md.a1) @ md.b
        a2h_a0 = h(md.a2) @ md.a0
        a2h_a1 = h(md.a2) @ md.a1
        a2h_a2 = h(md.a2) @ md.a2
        a2h_b = h(md.a2) @ md.b
        bh_a0 = h(md.b) @ md.a0
        bh_a1 = h(md.b) @ md.a1
        bh_a2 = h(md.b) @ md.a2
        bh_b = h(md.b) @ md.b

        qh_a0h_a0_q = h(q) @ a0h_a0 @ q
        qh_a0h_a1_q = h(q) @ a0h_a1 @ q
        qh_a0h_a2_q = h(q) @ a0h_a2 @ q
        qh_a0h_b = h(q) @ a0h_b
        qh_a1h_a0_q = h(q) @ a1h_a0 @ q
        qh_a1h_a1_q = h(q) @ a1h_a1 @ q
        qh_a1h_a2_q = h(q) @ a1h_a2 @ q
        qh_a1h_b = h(q) @ a1h_b
        qh_a2h_a0_q = h(q) @ a2h_a0 @ q
        qh_a2h_a1_q = h(q) @ a2h_a1 @ q
        qh_a2h_a2_q = h(q) @ a2h_a2 @ q
        qh_a2h_b = h(q) @ a2h_b
        bh_a0_q = bh_a0 @ q
        bh_a1_q = bh_a1 @ q
        bh_a2_q = bh_a2 @ q
        bh_b = bh_b

    q_t = q.T
    md_r = deepcopy(md)
    md_r.a0 = q_t @ md.a0 @ q
    md_r.a1 = q_t @ md.a1 @ q
    md_r.a2 = q_t @ md.a2 @ q
    md_r.b = q_t @ md.b

    time_stats.add_time("Offline")

    # online phase - sweep through domain
    for i in range(md_r.domain.size):
        x = solve_fem_point(md_r.domain[i], md_r)
        x_h = h(x)
        t_a0 = md_r.t_a0(md_r.domain[i])
        t_a1 = md_r.t_a1(md_r.domain[i])
        t_a2 = md_r.t_a2(md_r.domain[i])
        t_b = md_r.t_b(md_r.domain[i])

        time_stats.add_time("Online - solve")

        err_est_in_domain[i] = norm(
            t_a0 * t_a0 * x_h @ qh_a0h_a0_q @ x
            + t_a0 * t_a1 * x_h @ qh_a0h_a1_q @ x
            + t_a0 * t_a2 * x_h @ qh_a0h_a2_q @ x
            - t_a0 * t_b * x_h @ qh_a0h_b
            + t_a1 * t_a0 * x_h @ qh_a1h_a0_q @ x
            + t_a1 * t_a1 * x_h @ qh_a1h_a1_q @ x
            + t_a1 * t_a2 * x_h @ qh_a1h_a2_q @ x
            - t_a1 * t_b * x_h @ qh_a1h_b
            + t_a2 * t_a0 * x_h @ qh_a2h_a0_q @ x
            + t_a2 * t_a1 * x_h @ qh_a2h_a1_q @ x
            + t_a2 * t_a2 * x_h @ qh_a2h_a2_q @ x
            - t_a2 * t_b * x_h @ qh_a2h_b
            - t_b * t_a0 * bh_a0_q @ x
            - t_b * t_a1 * bh_a1_q @ x
            - t_b * t_a2 * bh_a2_q @ x
            + t_b * t_b * bh_b
        )

        time_stats.add_time("Online - add")

    if PLOT_GREEDY_ITERATIONS:
        plt.semilogy(md_r.domain, err_est_in_domain)
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


def solve_fem_point(t: float, md: ModelDefinition):
    """solves (t0*A0 + t1*A1 + t2*A2)X = B for a specific point t in a domain"""
    a = system_matrix(t, md)
    b = impulse_vector(t, md)

    # solve by LU factorization
    if issparse(a):
        x = splu(a).solve(b)
    else:
        lu = lu_factor(a)
        x = lu_solve(lu, b)

    return x


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


def system_matrix(t: float, md: ModelDefinition):
    a = md.t_a0(t) * md.a0 + md.t_a1(t) * md.a1 + md.t_a2(t) * md.a2
    return (a + a.T) / 2  # TODO: check if symmetrization needed after ROM | should A be a sparse matrix?


def impulse_vector(t: float, md: ModelDefinition):
    b = md.t_b(t) * md.b
    return b.todense() if issparse(b) else b # TODO: allow B returned here to be sparse?
