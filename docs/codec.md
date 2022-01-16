# Conditional Dependence Coefficient

Conditional Dependence Coefficient[^1] measures if `Y is a function of Z|X` (represented at `T(Y, Z|X)`), where Z and X can both be multi-variate.
When `X` is not present, it reduces to `T(Y, Z)`.

The coefficient is computed by comparing the nearest neighbour of consecutive rows of Z when Y is sorted.
It is 0 if `Y` is completely independent of `Z|X`, and 1 if `Y` is a measurable function of `Z|X`.

## Usage

Make a random 10000 x 3 array: 

Y is independent of each column. But it is a function of all three put together.

```python
import numpy as np

n = 10000
p = 3
x = np.random.uniform(0, 1, (n, p))

# y is independent of each x, but is a function of all three put together
y = np.mod(x.sum(axis=1), 1)

```

Compute `T(Y, Z)`:

```python
import xicorpy

# y is independent of x, so coeff is 0. 
c = xicorpy.compute_conditional_dependence(y, x[:, 0])
assert np.allclose(c, 0, atol=1e-2)

# y is independent of any two x columns too, so coeff is again 0.
c = xicorpy.compute_conditional_dependence(y, x[:, :1])
assert np.allclose(c, 0, atol=1e-2)
c = xicorpy.compute_conditional_dependence(y, x[:, 1:])
assert np.allclose(c, 0, atol=1e-2)

# y is a function of all three put together. Now coeff is close to 1.
c = xicorpy.compute_conditional_dependence(y, x)
assert np.allclose(c, 1, atol=1e-1)
```

Compute `T(Y, Z | X)`:

```python
# Given col1, and col2, y is a function of col3.
c = xicorpy.compute_conditional_dependence(y, x[:, 0], x[:, 1:])
assert np.allclose(c, 1, atol=1e-2)
```

Compute coefficient for each column of X:
```python
c = xicorpy.compute_conditional_dependence_1d(y, x)
assert isinstance(c, dict)
assert sorted(c.keys()) == [0, 1, 2]

c = xicorpy.compute_conditional_dependence_1d(y, x[:, :2], x[:, 2])
assert isinstance(c, dict)
assert sorted(c.keys()) == [0, 1]
```

## Citations

[^1]: [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence", Annals of Statistics](https://arxiv.org/abs/1910.12327)

