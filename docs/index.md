# Chatterjee's Xi, its Applications, and Offshoots

XicorPy is a Python package implementing Chatterjee's Xi, and its various offshoots.

The package currently implements:   

1. Chatterjee's Xi[^1]
2. Modified Xi[^2]
3. Conditional Dependence Coefficient[^3]
4. Feature Selection Algorithm (FOCI)[^3]



## Installation

The package is available on PyPI. You can install using pip: `pip install xicorpy`.

## Usage

Compute Chatterjee's Xi:

```python
import xicorpy

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
xi = xicorpy.compute_xi_correlation(x, y)

xi, p_value = xicorpy.compute_xi_correlation(x, y, get_p_values=True)

```

Compute Conditional Dependence Coefficient:

```python
import numpy as np
import xicorpy

n = 10000
p = 3
x = np.random.uniform(0, 1, (n, p))

# y is independent of each x, but is a function of all three put together
y = np.mod(x.sum(axis=1), 1)

# y is independent of x, so coeff is 0. 
c = xicorpy.compute_conditional_dependence(y, x[:, 0])
assert np.allclose(c, 0, atol=1e-2)

# y is a function of all three put together. Now coeff is close to 1.
c = xicorpy.compute_conditional_dependence(y, x)
assert np.allclose(c, 1, atol=1e-1)

# Given col1, and col2, y is a function of col3.
c = xicorpy.compute_conditional_dependence(y, x[:, 0], x[:, 1:])
assert np.allclose(c, 1, atol=1e-2)
```

Select Features using FOCI:

```python
import numpy as np
import xicorpy


n = 2000
p = 100
x = np.random.normal(0, 1, (n, p))
y = x[:, 0] * x[:, 1] + np.sin(x[:, 2] * x[:, 0])

# Select Features
selected = xicorpy.select_features_using_foci(y, x, num_features=10)

# Select Features with Initial Feature Set
selected = xicorpy.select_features_using_foci(y, x, num_features=10, init_selection=[0])
```


## Citations

[^1]: [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)
[^2]: [Lin and Han (2021). "On boosting the power of Chatterjee's rank correlation"](https://arxiv.org/abs/2108.06828)
[^3]: [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence"](https://arxiv.org/abs/1910.12327)

