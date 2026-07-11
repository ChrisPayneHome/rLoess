"""
Basic sanity/regression tests for pyloess.

Run with:  python -m pytest tests/  (or) python tests/test_loess.py
"""
import numpy as np
import sys
import os

from rloess import Loess


def test_1d_recovers_signal():
    np.random.seed(0)
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    y_true = np.sin(x)
    y = y_true + np.random.normal(scale=0.2, size=n)

    m = Loess(span=0.3, degree=2).fit(x, y)
    pred = m.predict(x)
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    assert rmse < 0.15


def test_span_controls_smoothness():
    np.random.seed(0)
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) + np.random.normal(scale=0.2, size=n)

    small = Loess(span=0.15, degree=2).fit(x, y)
    large = Loess(span=0.9, degree=2).fit(x, y)
    assert np.sum(small.residuals_ ** 2) < np.sum(large.residuals_ ** 2)


def test_robust_resists_outliers():
    np.random.seed(1)
    n = 80
    x = np.linspace(-3, 3, n)
    y_true = x ** 2
    y = y_true + np.random.normal(scale=1.0, size=n)
    y[10] += 40
    y[50] -= 40

    plain = Loess(span=0.5, degree=2, robust=False).fit(x, y)
    robust = Loess(span=0.5, degree=2, robust=True).fit(x, y)

    err_plain = np.sqrt(np.mean((plain.predict(x) - y_true) ** 2))
    err_robust = np.sqrt(np.mean((robust.predict(x) - y_true) ** 2))
    assert err_robust < err_plain
    # outlier points should end up strongly downweighted
    assert robust.robustness_weights_[10] < 0.1
    assert robust.robustness_weights_[50] < 0.1


def test_multivariate_fit():
    np.random.seed(2)
    n = 150
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    X = np.column_stack([x1, x2])
    y_true = x1 ** 2 + 0.5 * x1 * x2 - x2 ** 2
    y = y_true + np.random.normal(scale=0.3, size=n)

    m = Loess(span=0.4, degree=2).fit(X, y)
    rmse = np.sqrt(np.mean((m.predict(X) - y_true) ** 2))
    assert rmse < 0.5


def test_standard_errors_are_positive_and_finite():
    np.random.seed(3)
    n = 60
    x = np.linspace(0, 10, n)
    y = np.sin(x) + np.random.normal(scale=0.2, size=n)

    m = Loess(span=0.4, degree=2, compute_se=True).fit(x, y)
    preds, se = m.predict(x[:10], return_se=True)
    assert np.all(se > 0)
    assert np.all(np.isfinite(se))


def test_degree_zero_is_local_weighted_mean():
    np.random.seed(4)
    n = 50
    x = np.linspace(-1, 1, n)
    y = np.zeros(n) + np.random.normal(scale=0.01, size=n)  # ~ constant
    m = Loess(span=0.5, degree=0).fit(x, y)
    pred = m.predict(np.array([0.0]))
    assert abs(pred[0]) < 0.05


def test_matches_statsmodels_lowess_local_linear():
    """Cross-check against statsmodels' independent lowess implementation
    for the degree=1, non-robust case (statsmodels' default settings)."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        return  # optional dependency; skip if unavailable

    np.random.seed(3)
    n = 200
    x = np.linspace(0, 10, n)
    y = np.sin(x) + np.random.normal(scale=0.15, size=n)

    sm = lowess(y, x, frac=0.3, it=0, return_sorted=True)
    m = Loess(span=0.3, degree=1, robust=False).fit(x, y)
    mine = m.predict(sm[:, 0])

    assert np.max(np.abs(mine - sm[:, 1])) < 1e-8


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
