from typing import Dict, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from ._utils import convert_to_numeric, validate_and_prepare_for_conditional_dependence


class ConditionalDependence:
    """Class containing methods for calculating conditional dependence."""

    def __init__(self, y: npt.ArrayLike, z: npt.ArrayLike):
        """

        Initialize and Validate a ConditionalDependence object.

        You can then pass any X values to `compute_conditional_dependence` and `compute_conditional_dependence_1d`

        Args:
            y (npt.ArrayLike): A single list or 1D array or a pandas Series.
            z (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

        Raises:
            ValueError: If y is not 1d.
            ValueError: If z is not 1d or 2d.
            ValueError: If y and z have different lengths.
            ValueError: If there are <= 2 valid y values.
        """
        self.y_, self.z_df = validate_and_prepare_for_conditional_dependence(y, z)
        self.y = y
        self.z = z

    def _validate_x(self, x: npt.ArrayLike):
        if not 1 <= np.ndim(x) <= 2:
            raise ValueError("x must be a 1D or 2D array")

        x_shape = np.shape(x)
        y_shape = np.shape(self.y)
        if x_shape[0] != y_shape[0]:
            raise ValueError("x must have the same number of samples as y")

    def compute_conditional_dependence(self, x: npt.ArrayLike = None):
        """
        Compute conditional dependence coefficient based on:
            [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics](https://arxiv.org/abs/1910.12327)

        If X is passed, computes `T(Y, Z|X)` where `T` is the conditional dependence coefficient. Otherwise, computes `T(Y, Z)`.

        Conditional Dependence Coefficient lies between 0 and 1, and is

            0 if Y is completely independent of Z|X
            1 if Y is a measurable function of Z|X

        Args:
            x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

        Returns:
            float: Conditional Dependence Coefficient.

        Raises:
            ValueError: If x is passed, and not same number of rows as y.

        """
        if x is None:
            return _conditional_dependence_no_x(self.y_, self.z_df)
        else:
            self._validate_x(x)
            x_ = convert_to_numeric(pd.DataFrame(x)).loc[self.y_.index]
            return _conditional_dependence_with_x(self.y_, self.z_df, x_)

    def compute_conditional_dependence_1d(self, x: npt.ArrayLike = None):
        """
        Computes conditional dependence of y on **each column** of z individually.

        Use when you want to compute `T(Y, Z_j|X)` for each column of Z.

        Args:
            x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

        Returns:
            dict: Keys are column names (or indices if x is not a pandas object), and values are conditional dependence coefficients.

        Raises:
            ValueError: If x is passed, and does not have same number of rows as y.

        """
        if x is None:
            return _conditional_dependence_each_z_no_x(self.y_, self.z_df)
        else:
            self._validate_x(x)
            x_ = convert_to_numeric(pd.DataFrame(x)).loc[self.y_.index]
            return _conditional_dependence_with_each_z(self.y_, self.z_df, x_)


def compute_conditional_dependence(
    y: npt.ArrayLike, z: npt.ArrayLike, x: npt.ArrayLike = None
) -> float:
    """
    Compute conditional dependence coefficient based on:
        [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics](https://arxiv.org/abs/1910.12327)

    If X is passed, computes `T(Y, Z|X)` where `T` is the conditional dependence coefficient. Otherwise, computes `T(Y, Z)`.

    Conditional Dependence Coefficient lies between 0 and 1, and is

        0 if Y is completely independent of Z|X
        1 if Y is a measurable function of Z|X

    Args:
        y (npt.ArrayLike): A single list or 1D array or a pandas Series.
        z (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
        x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

    Returns:
        float: Conditional Dependence Coefficient.

    Raises:
        ValueError: If y is not 1d.
        ValueError: If z is not 1d or 2d.
        ValueError: If y and z have different lengths.
        ValueError: If there are <= 2 valid y values.
        ValueError: If x is passed, and not same number of rows as y.

    """
    return ConditionalDependence(y, z).compute_conditional_dependence(x)


def compute_conditional_dependence_1d(
    y: npt.ArrayLike, z: npt.ArrayLike, x: npt.ArrayLike = None
) -> Dict[Union[str, int], float]:
    """
    Computes conditional dependence of y on **each column** of z individually.

    Use when you want to compute `T(Y, Z_j|X)` for each column of Z.

    Args:
        y (npt.ArrayLike): A single list or 1D array or a pandas Series.
        z (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
        x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

    Returns:
        dict: Keys are column names (or indices if x is not a pandas object), and values are conditional dependence coefficients.

    Raises:
        ValueError: If y is not 1d.
        ValueError: If z is not 1d or 2d.
        ValueError: If y and z have different lengths.
        ValueError: If there are <= 2 valid y values.
        ValueError: If x is passed, and does not have the same number of rows as y.

    """
    return ConditionalDependence(y, z).compute_conditional_dependence_1d(x)


def _conditional_dependence_no_x(y: pd.Series, z: pd.DataFrame) -> float:
    if y.min() == y.max():
        return 0

    n = y.count()
    r_y = y.rank(method="max", ascending=True).values

    l_y = y.rank(method="max", ascending=False).values
    s = (l_y * (n - l_y)).sum() / (n ** 3)

    z_neighbors = NearestNeighbors(n_neighbors=2).fit(z.values)
    m = z_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
    r_m = r_y[m]

    q = (np.minimum(r_y, r_m) - (l_y * l_y) / n).sum() / (n * n)
    return q / s


def _conditional_dependence_each_z_no_x(
    y: pd.Series, z: pd.DataFrame
) -> Dict[Union[str, int], float]:
    ret: Dict[Union[str, int], float] = {c: 0 for c in z.columns}

    if y.min() == y.max():
        return ret

    n = y.count()
    r_y = y.rank(method="max", ascending=True).values

    l_y = y.rank(method="max", ascending=False).values
    s = (l_y * (n - l_y)).sum() / (n ** 3)

    for c in z.columns:
        z_neighbors = NearestNeighbors(n_neighbors=2).fit(z[[c]].values)
        m = z_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
        r_m = r_y[m]

        q = (np.minimum(r_y, r_m) - (l_y * l_y) / n).sum() / (n * n)
        ret[c] = q / s
    return ret


def _conditional_dependence_with_x(
    y: pd.Series, z: pd.DataFrame, x: pd.DataFrame
) -> float:
    if y.min() == y.max():
        return 0

    r_y = y.rank(method="max", ascending=True).values
    x_z = pd.concat([x, z], axis=1)

    x_z_neighbors = NearestNeighbors(n_neighbors=2).fit(x_z.values)
    m = x_z_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
    r_m = r_y[m]

    x_neighbors = NearestNeighbors(n_neighbors=2).fit(x.values)
    n_i = x_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
    r_n = r_y[n_i]

    num = (np.minimum(r_y, r_m) - np.minimum(r_y, r_n)).sum()
    den = (r_y - np.minimum(r_y, r_n)).sum()
    return num / den


def _conditional_dependence_with_each_z(
    y: pd.Series, z: pd.DataFrame, x: pd.DataFrame
) -> Dict[Union[str, int], float]:
    ret: Dict[Union[str, int], float] = {c: 0 for c in z.columns}
    if y.min() == y.max():
        return ret

    r_y = y.rank(method="max", ascending=True).values

    x_neighbors = NearestNeighbors(n_neighbors=2).fit(x.values)
    n_i = x_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
    r_n = r_y[n_i]

    for c in z.columns:
        x_z = pd.concat([x, z[[c]]], axis=1)

        x_z_neighbors = NearestNeighbors(n_neighbors=2).fit(x_z.values)
        m = x_z_neighbors.kneighbors(n_neighbors=1, return_distance=False).ravel()
        r_m = r_y[m]

        num = (np.minimum(r_y, r_m) - np.minimum(r_y, r_n)).sum()
        den = (r_y - np.minimum(r_y, r_n)).sum()

        ret[c] = num / den

    return ret
