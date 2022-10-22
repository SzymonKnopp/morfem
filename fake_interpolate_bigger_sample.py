import numpy as np


def fake_interpolate(array: np.ndarray, interpolation_rate: int) -> np.ndarray:
    new_array = np.zeros((array.shape[0] * interpolation_rate, array.shape[1] * interpolation_rate))

    for i in range(interpolation_rate):
        new_array[array.shape[0]*i:array.shape[0]*(i+1), array.shape[1]*i:array.shape[1]*(i+1)] = array

    return new_array


if __name__ == "__main__":
    interpolation_rate = 10

    print("Start")

    c = np.load("data/Ct.npy")
    gamma = np.load("data/Tt.npy")
    b = np.load("data/WP.npy")

    new_c = fake_interpolate(c, interpolation_rate)
    np.save("large_data/Ct.npy", new_c)

    new_gamma = fake_interpolate(c, interpolation_rate)
    np.save("large_data/Tt.npy", new_gamma)

    new_b = np.zeros((b.shape[0] * interpolation_rate, b.shape[1]))
    for i in range(interpolation_rate):
        new_b[b.shape[0]*i:b.shape[0]*(i+1)] = b
    np.save("large_data/WP.npy", new_b)

    print("Done")

