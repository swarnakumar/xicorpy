import numpy as np
import pandas as pd
import pytest

from xicorpy.correlation import compute_xi_correlation


@pytest.fixture
def anscombes_quartet():
    anscombes_quartet = {
        "x_1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "x_2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "x_3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "x_4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        "y_1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "y_2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74],
        "y_3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        "y_4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89],
    }

    return anscombes_quartet


def test_compute_xi_anscombes_quartet_1_xy(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["x_1"], anscombes_quartet["y_1"], get_modified_xi=False
    )
    assert isinstance(xi, float)
    assert np.allclose(xi, 0.275), f"xi = {xi}. Expected: 0.275"

    xi = compute_xi_correlation(
        anscombes_quartet["x_1"], anscombes_quartet["y_1"], get_modified_xi=True
    )
    assert isinstance(xi, float)
    assert np.allclose(xi, 0.513888), f"xi = {xi}. Expected: 0.513888"


def test_compute_xi_anscombes_quartet_1_p_values(anscombes_quartet):
    xi, p = compute_xi_correlation(
        anscombes_quartet["x_1"],
        anscombes_quartet["y_1"],
        get_modified_xi=False,
        get_p_values=True,
    )
    assert isinstance(xi, float)
    assert isinstance(p, float)
    assert np.allclose(xi, 0.275), f"xi = {xi}. Expected: 0.275"
    assert np.allclose(p, 0.07841556), f"p = {p}. Expected: 0.07841556"

    xi, p = compute_xi_correlation(
        anscombes_quartet["x_1"],
        anscombes_quartet["y_1"],
        get_modified_xi=True,
        get_p_values=True,
    )
    assert isinstance(xi, float)
    assert isinstance(p, float)
    assert np.allclose(xi, 0.513888), f"xi = {xi}. Expected: 0.513888"
    assert np.allclose(p, 0.0), f"p = {p}. Expected: 0.0"


def test_compute_xi_anscombes_quartet_1_yx(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["y_1"], anscombes_quartet["x_1"], get_modified_xi=False
    )
    assert isinstance(xi, float)
    assert np.allclose(xi, 0.25), f"xi = {xi}. Expected: 0.25"

    xi = compute_xi_correlation(
        anscombes_quartet["y_1"], anscombes_quartet["x_1"], get_modified_xi=True
    )
    assert isinstance(xi, float)
    assert np.allclose(xi, 0.48611), f"xi = {xi}. Expected: 0.48611"


def test_compute_xi_anscombes_quartet_3_xy(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["x_3"], anscombes_quartet["y_3"], get_modified_xi=False
    )
    assert np.allclose(xi, 0.725), f"xi = {xi}. Expected: 0.725"

    xi = compute_xi_correlation(
        anscombes_quartet["x_3"], anscombes_quartet["y_3"], get_modified_xi=True
    )
    assert np.allclose(xi, 0.736111, 1e-4), f"xi = {xi}. Expected: 0.736111"


def test_compute_xi_anscombes_quartet_3_yx(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["y_3"], anscombes_quartet["x_3"], get_modified_xi=False
    )
    assert np.allclose(xi, 0.725, 1e-4), f"xi = {xi}. Expected: 0.725"

    xi = compute_xi_correlation(
        anscombes_quartet["y_3"], anscombes_quartet["x_3"], get_modified_xi=True
    )
    assert np.allclose(xi, 0.736111, 1e-4), f"xi = {xi}. Expected: 0.736111"


def test_compute_xi_anscombes_quartet_4_xy(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["x_4"], anscombes_quartet["y_4"], get_modified_xi=False
    )
    assert np.allclose(xi, 0.175), f"xi = {xi}. Expected: 0.175"

    xi = compute_xi_correlation(
        anscombes_quartet["x_4"], anscombes_quartet["y_4"], get_modified_xi=True
    )
    assert np.allclose(xi, 0.111111, 1e-4), f"xi = {xi}. Expected: 0.111111"


def test_compute_xi_anscombes_quartet_4_yx(anscombes_quartet):
    xi = compute_xi_correlation(
        anscombes_quartet["y_4"], anscombes_quartet["x_4"], get_modified_xi=False
    )
    assert np.allclose(xi, 0.45, 1e-4), f"xi = {xi}. Expected: 0.45"

    xi = compute_xi_correlation(
        anscombes_quartet["y_4"], anscombes_quartet["x_4"], get_modified_xi=True
    )
    assert np.allclose(xi, 0.75, 1e-4), f"xi = {xi}. Expected: 0.75"


def test_compute_xi_df_df_xy(anscombes_quartet):
    x = pd.DataFrame({k: v for k, v in anscombes_quartet.items() if "x" in k})
    y = pd.DataFrame({k: v for k, v in anscombes_quartet.items() if "y" in k})

    xi = compute_xi_correlation(x, y, get_modified_xi=False)
    assert isinstance(xi, pd.DataFrame)
    assert xi.shape == (4, 4)

    assert sorted(xi.index) == ["x_1", "x_2", "x_3", "x_4"]
    assert sorted(xi.columns) == ["y_1", "y_2", "y_3", "y_4"]

    assert xi.loc["x_1", "y_1"] == pytest.approx(0.275, 1e-4)
    assert xi.loc["x_2", "y_2"] == pytest.approx(0.6, 1e-4)


def test_compute_xi_df_df_yx(anscombes_quartet):
    x = pd.DataFrame({k: v for k, v in anscombes_quartet.items() if "x" in k})
    y = pd.DataFrame({k: v for k, v in anscombes_quartet.items() if "y" in k})

    xi, p = compute_xi_correlation(y, x, get_modified_xi=False, get_p_values=True)
    assert isinstance(xi, pd.DataFrame)
    assert xi.shape == (4, 4)

    assert sorted(xi.columns) == ["x_1", "x_2", "x_3", "x_4"]
    assert sorted(xi.index) == ["y_1", "y_2", "y_3", "y_4"]

    assert xi.loc["y_1", "x_1"] == pytest.approx(0.25, 1e-4)
    assert xi.loc["y_3", "x_3"] == pytest.approx(0.725, 1e-4)

    assert isinstance(p, pd.DataFrame)
    assert p.columns.equals(xi.columns)
    assert p.index.equals(xi.index)


def test_compute_xi_non_numeric_xy():
    x = np.array(["abcd", "abcd", "xsdf", "abcd", "xsdf", "wert", None] * 10)
    y = np.array([1, 2, 3, 4, 5, 6, 1] * 10)

    xi = compute_xi_correlation(x, y)

    x_valid = np.array(["abcd", "abcd", "xsdf", "abcd", "xsdf", "wert"] * 10)
    y_valid = np.array([1, 2, 3, 4, 5, 6] * 10)
    xi_valid = compute_xi_correlation(x_valid, y_valid)
    assert np.allclose(xi, xi_valid), f"xi = {xi}; xi_valid: {xi_valid}."


def test_compute_xi_non_numeric_yx():
    x = ["abcd", "abcd", "xsdf", "abcd", "xsdf", "wert", None] * 10
    y = [1, 2, 3, 4, 5, 6, 1] * 10

    xi = compute_xi_correlation(y, x, False)
    assert np.allclose(xi, 0.908695), f"xi = {xi}. Expected: 0.908695."


def test_all_constant():
    x = [1] * 10
    y = list(range(10))

    xi, p = compute_xi_correlation(x, y, get_p_values=True)
    assert xi == 0
    assert p == 0.5


def test_all_equality():
    x = list(range(10))
    y = list(range(10))

    xi, p = compute_xi_correlation(x, y, get_p_values=True)
    assert xi == 1
    assert np.allclose(p, 0, atol=1e-4)


def test_internal_xi(anscombes_quartet):
    df = pd.DataFrame(anscombes_quartet)

    xi = compute_xi_correlation(df)
    assert sorted(xi.columns) == sorted(anscombes_quartet.keys())
    assert sorted(xi.index) == sorted(anscombes_quartet.keys())

    for k in anscombes_quartet:
        assert xi.loc[k, k] == 1


def test_error_checks():
    with pytest.raises(ValueError):
        # No x passed. Value Error
        compute_xi_correlation([])

    with pytest.raises(ValueError):
        # Blank Y passed
        compute_xi_correlation([1, 2, 3], [])

    with pytest.raises(ValueError):
        # No Y, but 1-d X
        compute_xi_correlation([1, 2, 3])

    with pytest.raises(ValueError):
        # No Y, 2-d X, but single column
        compute_xi_correlation([[1, 2, 3]])

    with pytest.raises(ValueError):
        # Unequal shapes
        compute_xi_correlation([1, 2, 3], [3, 4])
