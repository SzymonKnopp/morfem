import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from scipy.constants import pi, c as c_lightspeed
from test_helpers import finite_element_method_model_order_reduction_gsm, finite_element_method_gsm

TEST_REDUCED = True

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (20., 10.)
    plt.rcParams["font.size"] = 24
    plt.rcParams["lines.linewidth"] = 4

    frequency_points = np.linspace(3e9, 5e9, 100)
    gate_count = 2

    in_c = csc_array(np.load("data/Ct.npy"))
    in_gamma = csc_array(np.load("data/Tt.npy"))
    in_b = csc_array(np.load("data/WP.npy"))
    # in_kte1 = np.load("data/kTE1.npy")
    # in_kte2 = np.load("data/kTE2.npy")

    in_gamma *= -((2 * pi) / c_lightspeed) ** 2
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

        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_ref_in_frequency[:, 0, 0])), color="black", label=r"$S_{11}$")
        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_ref_in_frequency[:, 1, 0])), color="orange", label=r"$S_{21}$")
        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_in_frequency[:, 0, 0])), color="crimson", linestyle="dashed",
                 dashes=(4, 4), label=r"$S_{11_{red}}$")
        plt.plot(frequency_points, 20 * np.log10(np.abs(gsm_in_frequency[:, 1, 0])), color="crimson", linestyle="dotted",
                 label=r"$S_{21_{red}}$")
        plt.xlabel(r"$t$ [Hz]")
        plt.ylabel(r"$S_{11}, S_{21}$ [dB]")
        plt.legend()
        plt.grid()
        plt.savefig("output/result.png", bbox_inches="tight")
        plt.show()

        plt.semilogy(frequency_points, error_in_frequency, color="orange")
        plt.xlabel(r"$t$ [Hz]")
        plt.ylabel(r"$\Delta S$ [dB]")
        # plt.legend(loc="upper left")
        plt.grid()
        plt.savefig("output/error.png", bbox_inches="tight")
        plt.show()

        print(error_in_frequency.mean())
        print(error_in_frequency.max())

        plt.spy(in_c, color="k")
        plt.show()
        plt.spy(in_gamma, color="k")
        plt.show()

    print("Done")
