import numpy as np
from scipy.interpolate import interp2d
from scipy.sparse import csc_array
import matplotlib.pyplot as plot

if __name__ == "__main__":
    print("Start")
    interpolation_rate = 1

    c = csc_array(np.load("data/Ct.npy"))
    gamma = csc_array(np.load("data/Tt.npy"))
    b = csc_array(np.load("data/WP.npy"))

    c = c[:c.shape[0] // 100, :c.shape[1] // 100]

    row_idx = np.zeros(c.shape)
    for i in range(row_idx.shape[0]):
        row_idx[i+1:] += 1

    column_idx = np.zeros(c.shape)
    for i in range(column_idx.shape[1]):
        column_idx[:, i+1:] += 1

    c_function = interp2d(row_idx.flatten(), column_idx.flatten(), c.todense().flatten())
    print("Interpolation function done")

    new_row_idx = np.zeros((c.shape[0] * interpolation_rate, c.shape[1] * interpolation_rate))
    for i in range(new_row_idx.shape[0]):
        row_idx[i + 1:] += 1

    new_column_idx = np.zeros((c.shape[0] * interpolation_rate, c.shape[1] * interpolation_rate))
    for i in range(new_column_idx.shape[1]):
        column_idx[:, i + 1:] += 1

    print("New stuff got")

    new_c_values = c_function(new_row_idx.flatten(), new_column_idx.flatten())
    new_c = np.reshape(new_c_values, (c.shape[0] * interpolation_rate, c.shape[1] * interpolation_rate))

    plot.spy(new_c)
