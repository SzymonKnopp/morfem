import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, epsilon_0
from scipy.sparse import csc_array
from implementation import solve_finite_element_method_with_model_order_reduction


def generalized_scattering_matrix(frequency_point: float, e_mat: csc_array, b_mat_local: csc_array):
    gim = 1j * 2 * pi * frequency_point * epsilon_0 * e_mat.T @ b_mat_local  # 3.28
    gam = np.linalg.inv(gim)
    id = np.eye(gam.shape[0])
    gsm = 2 * np.linalg.inv(id + gam) - id
    return gsm


def every_nth_value(array, n: int):
    result = []
    anchor = 0
    while anchor < len(array):
        result.append(array[anchor])
        anchor += n
    return result


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2
    reduction_points_distance = 5

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    s11 = np.zeros([gate_count, frequency_points.size], dtype=complex)
    s21 = np.zeros([gate_count, frequency_points.size], dtype=complex)

    start = time.time()

    b_mat_in_frequency, e_mat_in_frequency = solve_finite_element_method_with_model_order_reduction(
        frequency_points, every_nth_value(frequency_points, reduction_points_distance),
        gate_count, c_mat, gamma_mat, b_mat, kte1, kte2
    )

    for i in range(frequency_points.size):
        gsm = generalized_scattering_matrix(frequency_points[i], e_mat_in_frequency[i], b_mat_in_frequency[i])
        s11[:, i] = gsm[0, 0]
        s21[:, i] = gsm[1, 0]

        if i % 100 == 0:
            print(i)

    print(f"Time elapsed: {time.time() - start} s")

    s11 = s11[0]
    s21 = s21[0]

    plt.plot(frequency_points, 20 * np.log10(np.abs(s11)))
    plt.plot(frequency_points, 20 * np.log10(np.abs(s21)))
    plt.legend(["S11", "S21"])
    plt.title("Dispersion characteristics")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|S11|, |S21| [dB]")
    plt.show()
    print("Done")

# w1 A + w2 * A2 + w3  A3 = w4 * B <- odwrócony interfejs