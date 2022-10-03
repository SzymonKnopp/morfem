import math
import numpy as np
from typing import List
from numpy.linalg import norm
from scipy.sparse import csc_array, isspmatrix_csc
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


def solve_finite_element_method(a_in_domain: List[csc_array] | np.ndarray, b_in_domain: np.ndarray):
    x_in_domain = np.zeros(b_in_domain.shape)
    for i in range(x_in_domain.shape[0]):
        x = solve_linear(a_in_domain[i], b_in_domain[i])
        x_in_domain[i] = x

    return x_in_domain


def solve_finite_element_method_with_model_order_reduction(a_in_domain: List[csc_array], b_in_domain: np.ndarray, domain: np.ndarray, reduction_rate=0.95):
    q = projection_base(a_in_domain, b_in_domain, reduction_rate, domain)

    # reduce model order - 5.5
    a_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], q.shape[1]])
    b_reduced_in_domain = np.zeros([b_in_domain.shape[0], q.shape[1], b_in_domain.shape[2]])
    for i in range(b_in_domain.shape[0]):
        a_reduced_in_domain[i] = q.T @ a_in_domain[i] @ q  # TODO: calculate q_mat.T once and reuse?
        b_reduced_in_domain[i] = q.T @ b_in_domain[i]

    x_in_domain = solve_finite_element_method(a_reduced_in_domain, b_reduced_in_domain)

    return x_in_domain, b_reduced_in_domain


def projection_base(a_in_domain: List[csc_array], b_in_domain: np.ndarray, reduction_rate, domain: np.ndarray, plot_residual=True):
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

    residual_in_domain = np.empty(b_in_domain.shape[0])
    for i in range(b_in_domain.shape[0]):
        a_reduced = q.T @ a_in_domain[i] @ q  # TODO: calculate q.T once and reuse?
        b_reduced = q.T @ b_in_domain[i]
        residual = a_in_domain[i] @ q @ solve_linear(a_reduced, b_reduced) - b_in_domain[i]
        residual_in_domain[i] = norm(residual)

    if plot_residual:
        plt.semilogy(domain, residual_in_domain)
        plt.title("Residual in domain")
        plt.xlabel("Domain")
        plt.ylabel("Residual")
        plt.show()

    return q


def solve_linear(a: csc_array | np.ndarray, b: np.ndarray):
    """solves Ax = b"""
    if isspmatrix_csc(a):  # LU factorization - 3.27
        e_mat = splu(a).solve(b)
    else:
        lu = lu_factor(a)
        e_mat = lu_solve(lu, b)
    return e_mat
