import math
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c, epsilon_0
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu


def read_json(filename: str):
    file = open(f"data/{filename}.json", "r")
    data = json.load(file)
    file.close()
    return data


def wave_impedance_te(kcte, f0):
    mi0 = 4 * pi * 1e-7
    k0 = (2 * pi * f0) / c
    gte = 1 / math.sqrt(k0 ** 2 - kcte ** 2)
    if gte.imag != 0:  # kcte and f0 handled only as scalars, no vector values support
        gte = 1j * math.fabs(gte)
    return 2 * pi * f0 * mi0 * gte


if __name__ == "__main__":
    frequency_points = np.linspace(3e9, 5e9, 1001)
    gate_count = 2

    ct = csc_array(np.load("data/Ct.npy"))
    tt = csc_array(np.load("data/Tt.npy"))
    wp = csc_array(np.load("data/WP.npy"))
    kte1 = np.load("data/kTE1.npy")
    kte2 = np.load("data/kTE2.npy")

    s11 = np.zeros([gate_count, frequency_points.size], dtype=complex)
    s21 = np.zeros([gate_count, frequency_points.size], dtype=complex)
    no_s11f = np.zeros([gate_count, frequency_points.size], dtype=complex)
    no_s21f = np.zeros([gate_count, frequency_points.size], dtype=complex)

    start = time.time()
    for i in range(frequency_points.size):
        k0 = (2 * pi * frequency_points[i]) / c

        zte1 = wave_impedance_te(kte1, frequency_points[i])
        zte2 = wave_impedance_te(kte2, frequency_points[i])

        g = ct - k0 ** 2 * tt

        di = np.diag(np.sqrt([1/zte1, 1/zte2]))
        wp_local = wp @ di
        g = (g + g.T) / 2
        x = splu(g).solve(wp_local)  # LU factorization

        # GAM
        z2 = 1j * 2 * pi * frequency_points[i] * epsilon_0 * x.T @ wp_local
        y2 = np.linalg.inv(z2)
        id = np.eye(y2.shape[0])

        # GSM
        s = 2 * np.linalg.inv(id + y2) - id
        s11[:, i] = s[0, 0]
        s21[:, i] = s[1, 0]
        no_s11f[:, i] = s[0, 0]
        no_s21f[:, i] = s[1, 0]

        if i % 100 == 0:
            print(i)

    print(f"Time elapsed: {time.time() - start} s")

    s11 = s11[0]
    s21 = s21[0]

    plt.plot(frequency_points, 20 * np.log10(np.abs(s11)))
    plt.plot(frequency_points, 20 * np.log10(np.abs(s21)))
    plt.show()
    print("Done")
