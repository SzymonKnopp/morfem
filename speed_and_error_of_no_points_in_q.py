import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import csc_array
from test_helpers import finite_element_method_gsm, finite_element_method_model_order_reduction_gsm, every_nth_value


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 101)
    gate_count = 2

    c_mat = csc_array(np.load("data/Ct.npy"))
    gamma_mat = csc_array(np.load("data/Tt.npy"))
    b_mat = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    ref_gsm = finite_element_method_gsm(frequency_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

    reduction_points_amount = []
    execution_time = []
    avg_error = []
    for reduction_points_distance in range(1, 41):

        start = time.time()

        reduction_points = every_nth_value(frequency_points, reduction_points_distance)
        gsm_of_frequency = finite_element_method_model_order_reduction_gsm(frequency_points, reduction_points, gate_count, c_mat, gamma_mat, b_mat, kte1, kte2)

        error = np.empty(reduction_points_distance)
        for i in range(reduction_points_distance):
            error[i] = norm(gsm_of_frequency[:, :, i] - ref_gsm[:, :, i])

        reduction_points_amount.append(len(reduction_points))
        execution_time.append(time.time() - start)
        avg_error.append(error.mean())

        print("Solved for reduction_points_distance = ", reduction_points_distance)

    plt.plot(reduction_points_amount, execution_time)
    plt.semilogy(reduction_points_amount, avg_error)
    plt.legend(["Execution time", "Average error"])
    plt.title("TITLE AAAAA")
    plt.xlabel("Number of reduction points")
    plt.ylabel("that gonna be complicated, two scales")
    plt.show()
    print("Done")
