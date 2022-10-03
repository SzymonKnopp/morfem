import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from test_helpers import finite_element_method_model_order_reduction_gsm, finite_element_method_gsm


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    gsm_ref_in_frequency = finite_element_method_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    gsm_in_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    error_in_frequency = np.zeros(frequency_points.size)
    for i in range(frequency_points.size):
        error_in_frequency[i] = norm(gsm_in_frequency[i] - gsm_ref_in_frequency[i])

    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_in_frequency[:, 0, 0])))
    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_in_frequency[:, 1, 0])))
    plt.legend(["S11", "S21"])
    plt.title("Dispersion characteristics on reduced model")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|S11|, |S21| [dB]")
    plt.show()

    plt.semilogy(frequency_points, error_in_frequency)
    plt.title("Norm of difference between ref GSM and reduced model GSM")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Error [dB]")
    plt.show()

    print("Done")

# w1 A + w2 * A2 + w3  A3 = w4 * B <- odwrócony interfejs
