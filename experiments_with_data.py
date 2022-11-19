import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from scipy.constants import pi, c as c_lightspeed
from test_helpers import finite_element_method_model_order_reduction_gsm, finite_element_method_gsm


TEST_REDUCED = True

if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 100)
    gate_count = 2

    in_c = csc_array(np.load("data/Ct.npy"))
    in_gamma = csc_array(np.load("data/Tt.npy"))
    in_b = csc_array(np.load("data/WP.npy"))
    in_kte1 = np.load("data/kTE1.npy")
    in_kte2 = np.load("data/kTE2.npy")

    in_gamma *= ((2 * pi) / c_lightspeed) ** 2
    in_b *= math.sqrt(1 / (8 * 1e-7 * pi ** 2))

    gsm_ref_in_frequency = finite_element_method_gsm(frequency_points, gate_count, in_c, in_gamma, in_b)

    if not TEST_REDUCED:
        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_ref_in_frequency[:, 0, 0])))
        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_ref_in_frequency[:, 1, 0])))
        plt.legend(["S11", "S21"])
        plt.title("Dispersion characteristics on reduced model")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("|S11|, |S21| [dB]")
        plt.show()

    if TEST_REDUCED:
        gsm_in_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, gate_count, in_c, in_gamma, in_b)

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
        # plt.title("Norm of difference between ref GSM and reduced model GSM")
        # plt.xlabel("Frequency [Hz]")
        # plt.ylabel("Error [dB]")
        plt.title("Różnica między charakterystyką uzyskaną z\npełnowymiarowego i zredukowanego modelu")
        plt.xlabel("Częstotliwość [Hz]")
        plt.ylabel("Błąd [dB]")
        plt.show()

        print(error_in_frequency.mean())

    print("Done")
