from typing import Tuple, Union
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss

from ._utils import convert_to_numeric

_RetType = Union[float, npt.NDArray, pd.DataFrame]


class XiCorrelation:
    """Class containing Xi Correlation computation components"""

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike = None):
        """
        If only `x` is passed, computes correlation between each column of `x`.
        If `y` is also passed, computes correlation between each column of `x` vs each column of `y`.

        If only `x` is passed, `x` MUST be 2-d. Otherwise, both `x` and `y` can be 1-d

        Args:
            x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
            y (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

        Raises:
            ValueError: If x and y are not of the same shape.
            ValueError: If there's less than 2 columns to compute correlation.

        """
        if not (1 <= np.ndim(x) <= 2 and np.shape(x)[0] >= 1):
            raise ValueError("x must be a 1D/2D array/list")

        x_df = pd.DataFrame(x)
        x_shape = np.shape(x)

        if y is not None:
            if not (1 <= np.ndim(y) <= 2 and np.shape(y)[0] >= 1):
                raise ValueError("y must be a 1D/2D array/list")
            y_shape = np.shape(y)
            if x_shape[0] != y_shape[0]:
                raise ValueError(
                    f"x: {x_shape[0]} samples, y: {y_shape[0]} samples. "
                    f"x and y MUST HAVE the same number of samples"
                )
            y_df = pd.DataFrame(y)
        else:
            if not (np.ndim(x) == 2 and np.shape(x)[0] >= 2 and np.shape(x)[1] >= 2):
                raise ValueError("x must be 2D if y is not provided")
            y_df = pd.DataFrame(x)

        self.x_df = convert_to_numeric(x_df)
        self.y_df = convert_to_numeric(y_df)

        self._x = x
        self._y = y

    def compute_xi(
        self,
        get_modified_xi: bool = None,
        m_nearest_neighbours: int = None,
        get_p_values: bool = False,
    ) -> Union[_RetType, Tuple[_RetType, _RetType]]:
        """
        Compute the Xi Coefficient (Chatterjee's Rank Correlation) between columns in X and Y.

        Xi Coefficient based on:
            [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)


        Modified Xi Coefficient based on:
            [Lin and Han (2021). "On boosting the power of Chatterjee's rank correlation"](https://arxiv.org/abs/2108.06828)

        The modified Xi Coefficient looks at M nearest neighbours to compute the correlation.
        This allows the coefficient to converge much faster. However, it is computationally slightly more intensive.
        For very large data, the two are likely to be very similar. We recommend using the modified Xi Coefficient.

        Args:
            get_modified_xi: Should the modified xi be computed? By default this is True when there are no ties and False when ties are present
            m_nearest_neighbours: Only used if get_modified_xi is True.
            get_p_values: Should the p-values be computed?
                            The null hypothesis is that Y is completely independent of X (i.e., xi = 0).

        Returns:
            float/np.ndarray/pd.DataFrame:
            - Xi Coefficient Values.
                - If both X and Y are 1-d, returns a single float.
                - If X is numpy object, returns a 2-D numpy array.
                - Otherwise returns a pd.DataFrame.
            - P-Values (only if get_p_values are true):
                - Same format at Xi

        """
        if _check_ties(self.x_df, self.y_df):
            if get_modified_xi:
                raise warnings.warn(
                    "Cannot use modified xi when there are ties present. Either explicitly set"
                    "`get_modified_xi=False` or leave as `None` to accept automatic decision.",
                    RuntimeWarning,
                )
            elif not get_modified_xi:  # handles None and False
                get_modified_xi = False
        elif get_modified_xi is None:
            get_modified_xi = True

        ret = pd.DataFrame(0, index=self.x_df.columns, columns=self.y_df.columns)
        _, p = _get_p_no_ties(0, self.x_df.shape[0])
        p_values = pd.DataFrame(p, index=self.x_df.columns, columns=self.y_df.columns)

        for i in self.x_df.columns:
            i_col: pd.Series = self.x_df[i]
            if i_col.min() == i_col.max():  # pragma: no cover
                # Constant column. Correlation will anyway be 0.
                ret.loc[i] = 0
            else:
                if i_col.hasnans:
                    i_col = i_col.dropna()

                if i_col.shape[0] <= 2:  # pragma: no cover
                    # Not enough samples to compute correlation.
                    ret.loc[i] = 0
                else:
                    # Sort once to avoid sorting each time we compute correlation.
                    i_col = i_col.sort_values()
                    for j in self.y_df.columns:
                        j_col: pd.Series = self.y_df.loc[i_col.index, j]
                        if j_col.hasnans:
                            j_col = j_col.dropna()

                        if get_p_values:
                            xi, sd, p = _single_pair(  # type: ignore
                                i_col,
                                j_col,
                                get_modified_xi,
                                m_nearest_neighbours,
                                True,
                            )
                            ret.loc[i, j] = xi
                            p_values.loc[i, j] = p
                        else:
                            ret.loc[i, j] = _single_pair(
                                i_col,
                                j_col,
                                get_modified_xi,
                                m_nearest_neighbours,
                                False,
                            )

        if (
            isinstance(self._x, list)
            and isinstance(self._y, list)
            and np.ndim(self._x) == 1
            and np.ndim(self._y) == 1
        ):
            if get_p_values:
                return ret.values[0, 0], p_values.values[0, 0]
            else:
                return ret.values[0, 0]

        if isinstance(self._x, np.ndarray):
            ret = ret.values
            p_values = p_values.values

        if get_p_values:
            return ret, p_values

        return ret


def compute_xi_correlation(
    x: npt.ArrayLike,
    y: npt.ArrayLike = None,
    get_modified_xi: bool = None,
    m_nearest_neighbours: int = None,
    get_p_values: bool = False,
) -> Union[_RetType, Tuple[_RetType, _RetType]]:
    """
    Helper function to compute the Xi Coefficient - uses the class machinery from `XiCorrelation`.

    Compute the Xi Coefficient (Chatterjee's Rank Correlation) between columns in X and Y.

    Xi Coefficient based on:
        [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)


    Modified Xi Coefficient based on:
        [Lin and Han (2021). "On boosting the power of Chatterjee's rank correlation"](https://arxiv.org/abs/2108.06828)

    The modified Xi Coefficient looks at M nearest neighbours to compute the correlation.
    This allows the coefficient to converge much faster. However, it is computationally slightly more intensive.
    For very large data, the two are likely to be very similar. We recommend using the modified Xi Coefficient.

    If only X is passed, computes correlation between each column of X.
    If Y is also passed, computes correlation between each column of X vs each column of Y.

    If only X is passed, X MUST be 2-d. Otherwise, both X and Y can be 1-d

    Args:
        x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
        y (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
        get_modified_xi: Should the modified xi be computed? By default this is True when there are no ties and False when ties are present
        m_nearest_neighbours: Only used if get_modified_xi is True.
        get_p_values: Should the p-values be computed?
                        The null hypothesis is that Y is completely independent of X (i.e., xi = 0).

    Returns:
        float/np.ndarray/pd.DataFrame:
        - Xi Coefficient Values.
            - If both X and Y are 1-d, returns a single float.
            - If X is numpy object, returns a 2-D numpy array.
            - Otherwise returns a pd.DataFrame.
        - P-Values (only if get_p_values are true):
            - Same format at Xi


    """
    return XiCorrelation(x, y).compute_xi(
        get_modified_xi, m_nearest_neighbours, get_p_values
    )


def _single_pair(
    x: pd.Series,
    y: pd.Series,
    get_modified_xi: bool = True,
    m: int = None,
    get_p_value: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    if x.equals(y):
        if get_p_value:
            sd, p = _get_p_no_ties(1, x.shape[0])
            return 1, sd, p
        else:
            return 1
    if x.max() == x.min() or y.max() == y.min():  # pragma: no cover
        if get_p_value:
            sd, p = _get_p_no_ties(0, x.shape[0])
            return 0, sd, p
        else:
            return 0

    if not x.index.equals(y.index):
        common = x.index.intersection(y.index)
        return _single_pair(
            x.loc[common], y.loc[common], get_modified_xi, m, get_p_value
        )

    n = x.shape[0]
    if n <= 2:  # pragma: no cover
        if get_p_value:
            sd, p = _get_p_no_ties(0, n)
            return 0, sd, p
        else:
            return 0

    if get_modified_xi:
        return _modified_xi(y, m, get_p_value)
    else:
        return _xi(y, get_p_value)


def _xi(
    y: pd.Series, get_p_value: bool = False
) -> Union[float, Tuple[float, float, float]]:
    n = y.shape[0]
    r = y.rank(method="max", ascending=True).values
    l = y.rank(method="max", ascending=False).values
    num = n * np.abs(r[1:] - r[:-1]).sum()
    den = 2 * (l * (n - l)).sum()
    xi = 1 - num / den
    if get_p_value:
        sd, p = _get_p_value(xi, r, l, n)
        return xi, sd, p
    else:
        return xi


def _modified_xi(
    y: pd.Series, m: int = None, get_p_value: bool = False
) -> Union[float, Tuple[float, float, float]]:
    n = y.shape[0]
    r = y.rank(method="first", ascending=True).values
    if m is None:
        m = min(int(np.sqrt(n)), 10)
        if m < 1:  # pragma: no cover
            m = 1
    num = 0
    for mm in range(1, m + 1):
        m_sum = np.minimum(r[: (n - mm)], r[mm:]).sum()
        m_sum += r[(n - mm) :].sum()

        num += m_sum

    den = (n + 1) * (n * m + m * (m + 1) / 4)
    xi = -2 + 6 * num / den

    if get_p_value:
        v = (2 / 5 * 1 / (n * m) + 8 / 15 * m / n**2) * 2
        p = 1 - ss.norm.cdf(np.sqrt(n) * xi / np.sqrt(v))
        return xi, np.sqrt(v), p

    return xi


def _get_p_no_ties(xi: float, n: int) -> Tuple[float, float]:
    sd = np.sqrt(2 / 5)
    if xi == 0:
        p = 0.5
    else:
        p = 1 - ss.norm.cdf(np.sqrt(n) * xi / np.sqrt(2 / 5))
    return sd, p


def _get_p_value(
    xi: float, r: np.ndarray, l: np.ndarray, n: int
) -> Tuple[float, float]:
    if xi == 0:  # pragma: no cover
        return 0.5, np.sqrt(2 / 5)

    qfr = sorted(r / n)
    gr = l / n
    cu = (gr * (1 - gr)).mean()

    ind = np.arange(1, n + 1)
    ind2 = 2 * n - 2 * ind + 1

    ai = (ind2 * qfr * qfr).mean() / n
    ci = (ind2 * qfr).mean() / n

    cq = np.cumsum(qfr)
    m = (cq + (n - ind) * qfr) / n
    b = (m * m).mean()
    v = (ai - 2 * b + ci**2) / cu**2

    p = 1 - ss.norm.cdf(np.sqrt(n) * xi / np.sqrt(v))
    return np.sqrt(v), p


def _check_ties(*dfs: pd.DataFrame) -> bool:
    df = pd.concat(dfs, ignore_index=True, axis="columns")
    if len(df.drop_duplicates()) == len(df):
        return False
    return True
