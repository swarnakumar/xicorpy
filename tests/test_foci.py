import pytest
import numpy as np

from src import select_features_using_foci


@pytest.fixture
def data():
    n = 2000
    p = 100

    x = np.random.normal(0, 1, (n, p))
    return x


def test_constant_y(data):
    y = np.ones(data.shape[0])
    selected = select_features_using_foci(y, data, num_features=10)
    expected = []

    assert selected == expected, f"Expected NO features. Selected: {selected}"


# Init selection is already full. Should just return as is
def test_init_selection_is_full(data):
    y = np.random.rand(data.shape[0])
    selected = select_features_using_foci(
        y, data, init_selection=[1, 2], num_features=2
    )
    expected = [1, 2]

    assert selected == expected, f"Expected [1, 2]. Selected: {selected}"


def test_add_83(data):
    y = data[:, 0] * data[:, 1] + np.sin(data[:, 2] * data[:, 0])

    selected = select_features_using_foci(y, data, num_features=10)
    expected = [0, 1, 2]
    assert [i for i in expected if i not in selected] == [], f"Selected: {selected}"


def test_add_83_wo_autostop(data):
    y = data[:, 0] * data[:, 1] + np.sin(data[:, 2] * data[:, 0])

    selected = select_features_using_foci(y, data, num_features=3, auto_stop=False)
    expected = [0, 1, 2]
    assert [i for i in expected if i not in selected] == [], f"Selected: {selected}"


def test_add_84(data):
    y = (
        data[:, 0] * data[:, 1]
        + data[:, 0]
        - data[:, 2]
        + np.random.normal(0, 1, data.shape[0])
    )

    selected = select_features_using_foci(y, data, num_features=10)
    expected = [0, 1, 2]
    assert [i for i in expected if i not in selected] == [], f"Selected: {selected}"


def test_add_84_with_init(data):
    y = (
        data[:, 0] * data[:, 1]
        + data[:, 0]
        - data[:, 2]
        + np.random.normal(0, 1, data.shape[0])
    )

    selected = select_features_using_foci(y, data, num_features=10, init_selection=[0])
    expected = [0, 1, 2]
    assert [i for i in expected if i not in selected] == [], f"Selected: {selected}"
