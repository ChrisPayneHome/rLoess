# rLoess

A from-scratch Python implementation of **LOESS** (LOcally Estimated
Scatterplot Smoothing), built to match the behavior of R's built-in
`stats::loess()` as closely as possible — same core algorithm (Cleveland,
Devlin & Grosse, 1988), same parameters, same options.

Verified to match `statsmodels.nonparametric.smoothers_lowess.lowess`
(degree=1, non-robust case) to within `1e-8`, and includes its own test
suite covering signal recovery, span/degree effects, robust fitting, and
multivariate predictors.

## Features

- **`span`** — fraction of data used in each local neighborhood (like R's `span`, default `0.75`)
- **`degree`** — 0 (local mean), 1 (local linear), or 2 (local quadratic, default) — same as R's `degree`
- **`robust=True`** — iterative Tukey biweight reweighting, equivalent to R's `family="symmetric"` (default is R's `family="gaussian"`, i.e. `robust=False`)
- **Multivariate predictors** — degree-2 fits include all cross terms, exactly as R does for `loess(y ~ x1 + x2, degree = 2)`
- **`normalize=True`** — rescales multiple predictors so distances aren't dominated by whichever variable has the largest scale, matching R's default
- **Standard errors** — optional `compute_se=True` + `predict(..., return_se=True)`, analogous to R's `predict(fit, newdata, se = TRUE)`
- **`summary()`** — prints span, degree, family, equivalent number of parameters, and residual standard error, similar to `summary(loess_fit)` in R

## Install

```bash
pip install numpy
# then just copy the rloess/ folder into your project, or:
pip install -e .
```

## Usage

```python
import numpy as np
from rloess import Loess

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.2, size=100)

# Equivalent to R:  loess(y ~ x, span = 0.3, degree = 2)
model = Loess(span=0.3, degree=2)
model.fit(x, y)

pred = model.predict(x)                 # fitted values
new_pred = model.predict(np.linspace(0, 10, 500))  # predict on a fine grid

model.summary()
```

### Robust fitting (outlier resistant)

```python
# Equivalent to R:  loess(y ~ x, span = 0.3, degree = 2, family = "symmetric")
model = Loess(span=0.3, degree=2, robust=True)
model.fit(x, y)

model.robustness_weights_   # per-observation weight, near 0 for outliers
```

### Multivariate predictors

```python
X = np.column_stack([x1, x2])   # shape (n, 2)
model = Loess(span=0.5, degree=2, normalize=True)
model.fit(X, y)
model.predict(X_new)             # X_new shape (m, 2)
```

### Standard errors / confidence intervals

```python
model = Loess(span=0.3, degree=2, compute_se=True)
model.fit(x, y)
pred, se = model.predict(x_grid, return_se=True)

lower = pred - 1.96 * se
upper = pred + 1.96 * se
```

## How closely does this match R?

- The **core math is identical**: tricube-weighted local polynomial
  regression, centered at the query point, with the neighborhood defined
  by the `span` nearest points (by Euclidean distance after optional
  predictor scaling).
- **Robustness iterations** use the same bisquare reweighting scheme
  R uses for `family="symmetric"`, with the same default of 4 iterations.
- This package always computes the **exact** local fit at every query
  point — equivalent to R's `surface="direct"`. R's *default* is
  `surface="interpolate"`, which fits on a kd-tree of "vertices" and
  interpolates between them purely as a speed optimization for large
  datasets. Results from the two should be very close but not bit-for-bit
  identical; if you need to match R's default output exactly, refit in R
  with `control = loess.control(surface = "direct")` for a fair
  comparison.
- Standard errors are computed via the usual local-regression variance
  formula (`Var = sigma^2 * ||l(x0)||^2`, where `l(x0)` is the vector of
  linear weights producing the local fit), with `sigma^2` estimated from
  the residual sum of squares and the trace of the smoother (hat) matrix
  as the equivalent number of parameters. This mirrors what
  `summary(loess_fit)` reports in R, though R's exact degrees-of-freedom
  bookkeeping (`one.delta`/`two.delta`) differs slightly in higher-order
  terms.
- `span > 1` (R allows this to further smooth beyond using every point)
  is supported as an approximation — the neighborhood is enlarged by a
  factor of `span` rather than following R's exact internal windowing
  rule for this edge case.

## Performance notes

Because every prediction (and every robustness iteration during fitting)
performs an exact local regression using **all** training points, cost is
O(n) per prediction and O(n²) per fitting pass. This is fine for the
typical scatterplot-smoothing use case (up to a few thousand points) but
will be slow for very large datasets — R's default `surface="interpolate"`
exists specifically to avoid this cost via approximation, which is a
possible future extension here (e.g. a kd-tree + local blending).

## Files

- `rloess/loess.py` — the `Loess` class and `loess()` convenience function
- `tests/test_loess.py` — test suite (`python tests/test_loess.py`)
- `example.py` — runnable example generating a comparison plot
