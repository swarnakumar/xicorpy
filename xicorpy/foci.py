import warnings
from typing import List, Union, Tuple

import numpy.typing as npt

from ._utils import validate_and_prepare_for_conditional_dependence
from .conditional_dependence import (
    compute_conditional_dependence_1d,
    compute_conditional_dependence,
)


class FOCI:
    """Class for computing FOCI."""

    def __init__(self, y: npt.ArrayLike, x: npt.ArrayLike):
        """
        Initialize and validate the FOCI object.

        You can then use the `select_features` method to select features.

        Args:
            y (npt.ArrayLike): A single list or 1D array or a pandas Series.
            x (npt.ArrayLike): A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.

        Raises:
            ValueError: If y is not 1d.
            ValueError: If x is not 1d or 2d.
            ValueError: If y and x have different lengths.
            ValueError: If there are <= 2 valid y values.
        """
        self.y_, self.x_df = validate_and_prepare_for_conditional_dependence(y, x)

    def _get_next_p(self, current_selection: List[Union[int, str]]):
        if current_selection:
            to_be_evaluated = [k for k in self.x_df if k not in current_selection]
            codec = compute_conditional_dependence_1d(
                self.y_, self.x_df[to_be_evaluated], self.x_df[current_selection]
            )
        else:
            codec = compute_conditional_dependence_1d(self.y_, self.x_df)

        next_p = max(codec, key=lambda k: codec[k])
        return next_p, codec[next_p]

    def select_features(
        self,
        num_features: int = None,
        init_selection: List[Union[int, str]] = None,
        get_conditional_dependency: bool = False,
    ) -> Union[List[Union[int, str]], Tuple[List[Union[int, str]], List[float]]]:
        """
        Selects features based on the Feature Ordering based on Conditional Independence (FOCI) algorithm in:
            [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics](https://arxiv.org/abs/1910.12327)

        Args:
            num_features: Maximum number of features to select. Defaults to the number of features in x.
            init_selection (list): Initial selection of features.
            get_conditional_dependency (bool): If True, returns conditional dependency. Defaults to False

        Returns:
            list: List of selected features.
                If x was `pd.DataFrame`, this will be column names.
                Otherwise, this will be indices.
            list: Conditional Dependency measure as each feature got selected
                Only when get_conditional_dependency is True

        """
        if num_features is None:  # pragma: no cover
            num_features = self.x_df.shape[1]

        current_selection = [i for i in (init_selection or [])]
        if len(current_selection) >= num_features:
            warnings.warn("Initial selection is already complete")
            if get_conditional_dependency:  # pragma: no cover
                return current_selection, []
            return current_selection

        codec = []
        stop = False
        while not stop and len(current_selection) < num_features:
            next_p, next_t = self._get_next_p(current_selection)

            if next_t <= 0:  # pragma: no cover
                stop = True
            else:
                current_selection.append(next_p)
                if get_conditional_dependency:
                    codec.append(
                        compute_conditional_dependence(
                            self.y_, self.x_df[current_selection]
                        )
                    )

        if get_conditional_dependency:
            return current_selection, codec

        return current_selection


def select_features_using_foci(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    num_features: int = None,
    init_selection: List[Union[int, str]] = None,
    get_conditional_dependency: bool = False,
) -> Union[List[Union[int, str]], Tuple[List[Union[int, str]], List[float]]]:
    """
    Implements the FOCI algorithm for feature selection.

    Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics.
    https://arxiv.org/abs/1910.12327.

    Args:
        y (npt.ArrayLike): The dependent variable. A single list or 1D array or a pandas Series.
        x (npt.ArrayLike): The independent variables. A single list or list of lists or 1D/2D numpy array or pd.Series or pd.DataFrame.
        num_features: Max number of features to select. Defaults to ALL features.
        init_selection (list): Initial selection of features.
            If `x` is a `pd.DataFrame`, this is expected to be a list of column names.
            Otherwise, this is expected to be a list of indices.
        get_conditional_dependency (bool): If True, returns conditional dependency

    Returns:
        list: List of selected features.
            If x was `pd.DataFrame`, this will be column names.
            Otherwise, this will be indices.
        list: Conditional Dependency measure as each feature got selected
            Only when get_conditional_dependency is True

    Raises:
        ValueError: If y is not 1d.
        ValueError: If x is not 1d or 2d.
        ValueError: If y and x have different lengths.
        ValueError: If there are <= 2 valid y values.

    """
    return FOCI(y, x).select_features(
        num_features,
        init_selection,
        get_conditional_dependency=get_conditional_dependency,
    )
