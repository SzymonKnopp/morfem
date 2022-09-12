import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from test_helpers import finite_element_method_model_order_reduction_gsm, every_nth_value


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2
    reduction_points_distance = 5

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    start = time.time()

    reduction_points = every_nth_value(frequency_points, reduction_points_distance)
    gsm_of_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    print(f"Time elapsed: {time.time() - start} s")

    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_of_frequency[0, 0, :])))
    plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_of_frequency[1, 0, :])))
    plt.legend(["S11", "S21"])
    plt.title("Dispersion characteristics")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|S11|, |S21| [dB]")
    plt.show()
    print("Done")

# w1 A + w2 * A2 + w3  A3 = w4 * B <- odwrócony interfejs