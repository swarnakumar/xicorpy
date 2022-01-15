import pytest
import numpy as np
import pandas as pd

from src.conditional_dependence import (
    compute_conditional_dependence,
    compute_conditional_dependence_1d,
)


@pytest.fixture
def data():
    n = 100000
    p = 3

    x = np.random.uniform(0, 1, (n, p))
    # y is independent of each x, but is a function of all three put together
    y = np.mod(x.sum(axis=1), 1)

    return y, x


# 1 column at a time. Independent
def test_codec_independent_1d(data):
    y, x = data

    tol = 0.1

    c1 = compute_conditional_dependence(y, x[:, 0])
    assert np.allclose(
        c1, 0, atol=tol
    ), f"Expected c1 to be very close to 0. c1: {c1}. tol: {tol}"

    c2 = compute_conditional_dependence(y, x[:, 1])
    assert np.allclose(
        c2, 0, atol=tol
    ), f"Expected c2 to be very close to 0. c2: {c2}. tol: {tol}"

    c3 = compute_conditional_dependence(y, x[:, 2])
    assert np.allclose(
        c3, 0, atol=tol
    ), f"Expected c3 to be very close to 0. c3: {c3}. tol: {tol}"


# 2 columns at a time. Must still be independent.
def test_codec_independent_2d(data):
    y, x = data

    tol = 0.1

    c1 = compute_conditional_dependence(y, x[:, :2])
    assert np.allclose(c1, 0, atol=tol), f"Expected c1 to be very close to 0. c1: {c1}"

    c2 = compute_conditional_dependence(y, x[:, 1:])
    assert np.allclose(c2, 0, atol=tol), f"Expected c2 to be very close to 0. c2: {c2}"


# y is a function of x.
def test_codec_dependent(data):
    y, x = data
    tol = 0.1
    c = compute_conditional_dependence(y, x)
    assert np.allclose(c, 1, atol=tol), f"Expected c to be very close to 1. c: {c}"


# adjusting for x[2], y is still independent of x[1].
def test_codec_conditional_independent(data):
    y, x = data
    tol = 0.1
    c = compute_conditional_dependence(y, x[:, 1], x[:, 2])
    assert np.allclose(c, 0, atol=tol), f"Expected c to be very close to 1. c: {c}"


# adjusting for x[2], y is a function of x[0,1].
def test_codec_conditional_dependent(data):
    y, x = data
    tol = 0.1
    c = compute_conditional_dependence(y, x[:, :2], x[:, 2])
    assert np.allclose(c, 1, atol=tol), f"Expected c to be very close to 1. c: {c}"


def test_constant_y(data):
    _, x = data

    c = compute_conditional_dependence(np.ones(x.shape[0]), x[:, :2], x[:, 2])
    assert c == 0

    c = compute_conditional_dependence(np.ones(x.shape[0]), x[:, :2])
    assert c == 0

    c = compute_conditional_dependence_1d(np.ones(x.shape[0]), x[:, 0], x[:, 1:])
    assert np.allclose(list(c.values()), 0)

    c = compute_conditional_dependence_1d(np.ones(x.shape[0]), x)
    assert np.allclose(list(c.values()), 0)


def test_codec_column_wise_with_x(data):
    y, x = data
    tol = 0.1
    c = compute_conditional_dependence_1d(y, x[:, 1:], x[:, 0])
    assert sorted(c.keys()) == [
        0,
        1,
    ], f"Expected c to be a dict with keys 0 and 1. c: {c}"

    assert np.allclose(
        list(c.values()), 0, atol=tol
    ), f"Expected c to be very close to 0. c: {c}"


def test_codec_column_wise_no_x(data):
    y, x = data
    tol = 0.1
    c = compute_conditional_dependence_1d(y, x)
    assert sorted(c.keys()) == [
        0,
        1,
        2,
    ], f"Expected c to be a dict with keys 0, 1 and 2. c: {c}"

    assert np.allclose(
        list(c.values()), 0, atol=tol
    ), f"Expected c to be very close to 0. c: {c}"


def test_codec_with_pandas(data):
    y, x = data
    x_df = pd.DataFrame(x, columns=["a", "b", "c"])

    c = compute_conditional_dependence(y, x_df)
    assert np.allclose(c, 1, atol=0.1), f"Expected c to be very close to 1. c: {c}"

    c = compute_conditional_dependence_1d(y, x_df)
    assert sorted(c.keys()) == [
        "a",
        "b",
        "c",
    ], f"Expected c to be a dict with keys a, b and c. c: {c}"
    assert np.allclose(
        list(c.values()), 0, atol=0.1
    ), f"Expected c to be very close to 0. c: {c}"

    c = compute_conditional_dependence_1d(y, x_df[["b", "c"]], x_df["a"])
    assert sorted(c.keys()) == [
        "b",
        "c",
    ], f"Expected c to be a dict with keys b and c. c: {c}"
    assert np.allclose(
        list(c.values()), 0, atol=0.1
    ), f"Expected c to be very close to 0. c: {c}"

    c = compute_conditional_dependence_1d(y, x_df["c"], x_df[["a", "b"]])
    assert sorted(c.keys()) == ["c"], f"Expected c to be a dict with keys c. c: {c}"
    assert np.allclose(
        list(c.values()), 1, atol=0.1
    ), f"Expected c to be very close to 1. c: {c}"


def test_error_checks():
    # Y is blank
    with pytest.raises(ValueError):
        compute_conditional_dependence([], np.ones(10), np.ones(10))

    # Z is blank
    with pytest.raises(ValueError):
        compute_conditional_dependence(np.ones(10), [])

    # Y and X arent the same length
    with pytest.raises(ValueError):
        compute_conditional_dependence(np.ones(10), np.ones(20))

    # Y and X arent the same length
    with pytest.raises(ValueError):
        compute_conditional_dependence(np.ones(10), np.ones(20).reshape(2, 10))

    # n is too small
    with pytest.raises(ValueError):
        compute_conditional_dependence(np.ones(2), np.ones(2))

    # x is blank
    with pytest.raises(ValueError):
        compute_conditional_dependence(np.ones(10), np.ones(10), [])

    with pytest.raises(ValueError):
        compute_conditional_dependence_1d(np.ones(10), np.ones(10), [])

    # x isnt 1d/2d
    with pytest.raises(ValueError):
        compute_conditional_dependence(
            np.ones(10), np.ones(10), np.ones(30).reshape((10, 3, 1))
        )
