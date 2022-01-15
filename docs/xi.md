# Chatterjee's Xi Coefficient

Chatterjee's Xi[^1] measures if `Y is a function of X`. 
The coefficient is 0 if X and Y are independent, and 1 if Y is a measurable function of X. 
Xi is computed by comparing ranks of consecutive values of X when Y is sorted. 

Lin and Han's [^2] modification makes the original formulation more robust by comparing M right-neighbours. 
When M == 1, it reduces to the original formulation.

Do note that this formulation is `asymmetric`:

```
Xi(X, Y): Measures Y as a function of X
Xi(Y, X): Measures X as a function of Y 

Xi(X, Y) != Xi(Y, X)
```

To illustrate this better, consider the following example:

| X | Y |
|---|---|
| 8 | 6.58 |
| 8 | 5.76 |
| 8 | 7.71 |
| 8 | 8.84 |
| 8 | 8.47 |
| 8 | 7.04 |
| 8 | 5.25 |
| 19 | 12.5 |
| 8 |5.56
| 8 | 7.91 |
| 8 | 6.89 |

This is the 4th dataset of [Anscombe’s quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)

While we cannot have an estimate of Y given X, we can estimate X given Y - `If Y < 10: X = 8, else X = 19`.

| Direction | Chatterjee's Xi | Modified Xi |
|-----------|----------------|--------------|
| Xi(X, Y)     | 0.175           | 0.111         |
| Xi(Y, X)     | 0.45           | 0.75         |

The above table also illustrates the impact of Lin and Han's modification. 
For very large data, the two are likely to be very similar, and for smaller data, Lin and Han's formulation tends to be appropriate.

## Usage

```python
import xicorpy

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
xi = xicorpy.compute_xi_correlation(x, y)

## Get p-values:
xi, p_value = xicorpy.compute_xi_correlation(x, y, get_p_values=True)

## Explicitly specify m-nearest-neighbours:
xi = xicorpy.compute_xi_correlation(x, y, m_nearest_neighbours=5)

## Compute original formulation without Lin and Han's Modification:
xi = xicorpy.compute_xi_correlation(x, y, get_modified_xi=False)

```

Compute correlations between all columns in X vs all columns in Y:

```python

import pandas as pd
import xicorpy

x = pd.DataFrame({
    "x_1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
})
y = pd.DataFrame({
    "y_1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "y_2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74],
    "y_3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    "y_4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89],
})

xi = xicorpy.compute_xi_correlation(x, y)
```

Compute correlations between all columns in X:
```python

import pandas as pd
import xicorpy

x = pd.DataFrame({
    "x_1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y_1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "y_2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74],
})

xi = xicorpy.compute_xi_correlation(x)
```


## Citations

[^1]: [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)
[^2]: [Lin and Han (2021). "On boosting the power of Chatterjee's rank correlation"](https://arxiv.org/abs/2108.06828)

