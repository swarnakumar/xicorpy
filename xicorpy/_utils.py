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


def validate_and_prepare_for_conditional_dependence(y, x):
    y_shape = np.shape(y)
    x_shape = np.shape(x)

    if not (np.ndim(y) == 1 and y_shape[0] > 1):
        raise ValueError("y must be a 1D array with at least one sample")
    if not (1 <= np.ndim(x) <= 2 and x_shape[0] >= 1):
        raise ValueError("x must be a 1D or 2D array with at least one sample")

    if x_shape[0] != y_shape[0]:
        raise ValueError("y and x MUST have the same number of samples")

    y_: pd.Series = convert_to_numeric(pd.DataFrame({"y": y}))["y"]

    if y_.count() <= 2:
        raise ValueError("y must have at least 3 non-null samples")

    if y_.hasnans is not None:
        y_ = y_.dropna()

    x_df = convert_to_numeric(pd.DataFrame(x)).loc[y_.index]
    return y_, x_df
