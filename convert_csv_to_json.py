import pandas as pd
import numpy as np


def open_convert_save(filename: str):
    array = pd.read_csv(f"data_csv/{filename}.csv", header=None).values
    np.save(f"data/{filename}.npy", array)
    print(f"{filename} converted.")


if __name__ == "__main__":
    open_convert_save("Ct")
    open_convert_save("Tt")
    open_convert_save("WP")
    open_convert_save("kTE1")
    open_convert_save("kTe2")
