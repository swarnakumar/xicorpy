# Feature Ordering by Conditional Dependence (FOCI)

FOCI[^1] is a forward stepwise feature selection algorithm 
for multivariate regression based on Conditional Dependence measure.

## Usage

Make a random `2000x100` array of independent variables, and simulate a complex `Y`

```python
import numpy as np


n = 2000
p = 100
x = np.random.normal(0, 1, (n, p))
y = x[:, 0] * x[:, 1] + np.sin(x[:, 2] * x[:, 0])
```

Select Features
```python
import xicorpy

selected = xicorpy.select_features_using_foci(y, x, num_features=10)
```

Select Features with Initial Feature Set
```python
import xicorpy

selected = xicorpy.select_features_using_foci(y, x, num_features=10, init_selection=[0])
```

## Citations

[^1]: [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics](https://arxiv.org/abs/1910.12327)
