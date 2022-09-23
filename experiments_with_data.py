import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from test_helpers import finite_element_method_model_order_reduction_gsm, every_nth_value, finite_element_method_gsm


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    ref_gsm = finite_element_method_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    reduction_points = every_nth_value(frequency_points, 20)
    # reduction_points = [frequency_points[0], frequency_points[frequency_points.size // 2], frequency_points[-1]]
    gsm_of_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    error = np.zeros(frequency_points.size)
    for i in range(frequency_points.size):
        error[i] = norm(gsm_of_frequency[:, :, i] - ref_gsm[:, :, i])

    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_of_frequency[0, 0, :])))
    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_of_frequency[1, 0, :])))
    plt.legend(["S11", "S21"])
    plt.title("Dispersion characteristics on reduced model")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|S11|, |S21| [dB]")
    plt.show()

    plt.semilogy(frequency_points, error)
    plt.title("Norm of difference between ref GSM and reduced model GSM")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Error [dB]")
    plt.show()

    print("Done")

# w1 A + w2 * A2 + w3  A3 = w4 * B <- odwrócony interfejs
