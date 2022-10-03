import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from test_helpers import finite_element_method_gsm, finite_element_method_model_order_reduction_gsm, equally_distributed_points


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2
    tested_reduction_points_range = range(3, 30)

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    ref_gsm = finite_element_method_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    execution_time = []
    avg_error = []
    for reduction_points_amount in tested_reduction_points_range:

        start = time.time()

        reduction_points = equally_distributed_points(frequency_points, reduction_points_amount)  # TODO: change that to reduction_rate
        gsm_of_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

        error = np.zeros(frequency_points.size)
        for i in range(frequency_points.size):
            error[i] = norm(gsm_of_frequency[:, :, i] - ref_gsm[:, :, i])

        execution_time.append(time.time() - start)
        avg_error.append(error.mean())

        print("Solved for reduction_points_amount = ", reduction_points_amount)

    fig, plt1 = plt.subplots()
    fig.set_size_inches(8, 5)
    plt1.plot(tested_reduction_points_range, execution_time, "orange")
    plt2 = plt1.twinx()
    plt1.set_ylabel("Execution time [s]", color="orange")
    plt2.semilogy(tested_reduction_points_range, avg_error, "dodgerblue")
    plt2.set_ylabel("Error mean [dB]", color="dodgerblue")
    plt1.set_title("Time and error for different number of reduction points")
    plt1.set_xlabel("Number of reduction points")
    plt.show()
    print("Done")
