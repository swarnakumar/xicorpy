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

```python
import xicorpy

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
xi = xicorpy.compute_xi_correlation(x, y)

xi, p_value = xicorpy.compute_xi_correlation(x, y, get_p_values=True)

```

Refer to Documentation for more details.


## Citations

[^1]: [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)
[^2]: [Lin and Han (2021). "On boosting the power of Chatterjee's rank correlation"](https://arxiv.org/abs/2108.06828)
[^3]: [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence"](https://arxiv.org/abs/1910.12327)

