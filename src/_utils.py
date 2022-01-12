import numpy as np
import pandas as pd


def convert_to_numeric(data: pd.DataFrame):
    data = data.copy(True)
    for c in data:
        if data[c].dtype not in ["int", "float"]:
            codes, _ = pd.factorize(data[c])
            if np.min(codes) < 0:
                codes = codes.astype("float")
                codes[codes == -1] = np.nan
            data[c] = codes

    return data
