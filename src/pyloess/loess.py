"""
pyloess.loess
=============

A from-scratch Python implementation of LOESS / LOWESS (LOcally Estimated
Scatterplot Smoothing) that mirrors the behaviour of R's built-in
``loess()`` function.

Algorithm (Cleveland, Devlin & Grosse, 1988 -- the same method R's
``stats::loess`` is based on):

1. For a query point x0, compute distances from x0 to every training
   point (after optionally rescaling predictors so multi-dimensional
   distances are meaningful -- R's ``normalize=TRUE``).
2. Pick a neighbourhood: the ``span`` fraction of the n closest points
   (span in (0, 1]), or all n points with an enlarged window if
   span > 1.
3. Weight neighbours with the tricube kernel based on distance relative
   to the farthest point kept in the neighbourhood.
4. Fit a weighted polynomial regression (degree 0, 1 or 2) of y on the
   predictors, *centered at x0*, so the intercept of the local fit is
   directly the smoothed value at x0.
5. Optionally repeat with robustness (bisquare) reweighting based on the
   residuals of the previous pass -- R's ``family="symmetric"``.

This mirrors R's ``surface="direct"`` mode (exact computation at every
point) rather than the default ``surface="interpolate"`` (a kd-tree +
blending approximation used purely for speed). Numerically the two
should agree closely; ``direct`` is simply slower for very large n.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations_with_replacement
from typing import Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Low level numerical helpers
# --------------------------------------------------------------------------

def _tricube(u: np.ndarray) -> np.ndarray:
    """Tricube kernel: (1 - |u|^3)^3 for |u| < 1, else 0."""
    w = np.zeros_like(u, dtype=float)
    mask = u < 1
    w[mask] = (1.0 - u[mask] ** 3) ** 3
    return w


def _bisquare(u: np.ndarray) -> np.ndarray:
    """Tukey's bisquare weight: (1 - u^2)^2 for |u| < 1, else 0."""
    w = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1
    w[mask] = (1.0 - u[mask] ** 2) ** 2
    return w


def _generate_monomial_exponents(p: int, degree: int) -> list:
    """
    All exponent tuples (e_1, ..., e_p) with sum(e) <= degree, ordered by
    total degree. The first entry is always the all-zero (intercept) tuple.
    Matches the terms R's loess() includes for degree=1 (linear) and
    degree=2 (full quadratic incl. cross terms), for any number p of
    numeric predictors.
    """
    exps = [tuple([0] * p)]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(p), d):
            e = [0] * p
            for idx in combo:
                e[idx] += 1
            t = tuple(e)
            if t not in exps:
                exps.append(t)
    return exps


def _design_matrix(Xc: np.ndarray, exponents: Sequence[Tuple[int, ...]]) -> np.ndarray:
    """Build a design matrix from centered coordinates Xc (n x p) and a list
    of exponent tuples describing each monomial column."""
    n = Xc.shape[0]
    k = len(exponents)
    B = np.ones((n, k))
    for j, e in enumerate(exponents):
        if not any(e):
            continue  # intercept column already 1
        col = np.ones(n)
        for var_idx, power in enumerate(e):
            if power:
                col = col * Xc[:, var_idx] ** power
        B[:, j] = col
    return B


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------

class Loess:
    """
    Local polynomial regression smoother, matching R's ``loess()``.

    Parameters
    ----------
    span : float, default 0.75
        Equivalent to R's ``span``. Fraction of the data used in each
        local neighbourhood when in (0, 1]. Values > 1 enlarge the window
        beyond the closest point (all points are used, kernel widened by
        the given factor) -- an approximation of R's behaviour for
        span > 1.
    degree : {0, 1, 2}, default 2
        Degree of the local polynomial. Same meaning as R's ``degree``.
        0 = locally weighted mean, 1 = local linear, 2 = local quadratic
        (including cross terms for multivariate predictors).
    robust : bool, default False
        If True, use iterative bisquare robustness reweighting, matching
        R's ``family="symmetric"`` (as opposed to the default
        ``family="gaussian"``, which is ``robust=False`` here).
    iterations : int, optional
        Number of robustness iterations. Defaults to 4 when
        ``robust=True`` (R's default control) and 0 when ``robust=False``.
    normalize : bool, default True
        Rescale multivariate predictors before computing distances, so
        that predictors on very different scales don't dominate the
        neighbourhood search. Matches R's ``normalize=TRUE`` (only
        relevant when there is more than one predictor).
    compute_se : bool, default False
        If True, also compute the equivalent number of parameters and
        residual scale needed to report standard errors on prediction
        (mirrors R's ``predict(..., se=TRUE)``). This adds an extra
        O(n^2) pass, so it's off by default.

    Attributes set after ``fit``
    -----------------------------
    fitted_ : ndarray of shape (n,)
        Smoothed values at the training points (final robustness pass).
    residuals_ : ndarray of shape (n,)
        y - fitted_.
    robustness_weights_ : ndarray of shape (n,)
        Final per-observation robustness weights (all ones if
        ``robust=False``).
    enp_ : float
        Equivalent number of parameters (trace of the smoother/hat
        matrix), only computed if ``compute_se=True``.
    residual_scale_ : float
        Estimated residual standard deviation, only computed if
        ``compute_se=True``.
    """

    def __init__(
        self,
        span: float = 0.75,
        degree: int = 2,
        robust: bool = False,
        iterations: Optional[int] = None,
        normalize: bool = True,
        compute_se: bool = False,
    ):
        if span <= 0:
            raise ValueError("span must be > 0")
        if degree not in (0, 1, 2):
            raise ValueError("degree must be 0, 1, or 2")

        self.span = span
        self.degree = degree
        self.robust = robust
        self.iterations = iterations if iterations is not None else (4 if robust else 0)
        self.normalize = normalize
        self.compute_se = compute_se

        # populated by fit()
        self.X_train_ = None
        self.y_train_ = None
        self.Xs_train_ = None  # scaled training predictors
        self.scale_ = None
        self.exponents_ = None
        self.fitted_ = None
        self.residuals_ = None
        self.robustness_weights_ = None
        self.enp_ = None
        self.residual_scale_ = None

    # ---------------------------------------------------------------- #
    # public API
    # ---------------------------------------------------------------- #

    def fit(self, X, y) -> "Loess":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")
        if n < self.degree + 1:
            raise ValueError("need at least degree+1 observations")

        self.X_train_ = X
        self.y_train_ = y
        self.n_, self.p_ = n, p
        self.exponents_ = _generate_monomial_exponents(p, self.degree)

        if self.normalize and p > 1:
            med = np.median(X, axis=0)
            mad = np.median(np.abs(X - med), axis=0)
            mad[mad == 0] = 1.0
            self.scale_ = mad
        else:
            self.scale_ = np.ones(p)

        Xs = X / self.scale_
        self.Xs_train_ = Xs

        rob_weights = np.ones(n)
        fitted = np.zeros(n)
        n_passes = self.iterations + 1

        for it in range(n_passes):
            fitted = np.array([
                self._local_fit(Xs[i], Xs, y, rob_weights, want_l=False)[0]
                for i in range(n)
            ])
            residuals = y - fitted
            if it < n_passes - 1:
                s = np.median(np.abs(residuals))
                if s <= 0 or not np.isfinite(s):
                    break
                rob_weights = _bisquare(residuals / (6.0 * s))

        self.fitted_ = fitted
        self.residuals_ = y - fitted
        self.robustness_weights_ = rob_weights

        if self.compute_se:
            self._compute_enp_and_scale()

        return self

    def predict(self, X_new, return_se: bool = False):
        """
        Predict smoothed values at new points.

        Parameters
        ----------
        X_new : array-like, shape (m,) or (m, p)
        return_se : bool
            If True, also return the standard error of each prediction.
            Requires the model to have been fit with ``compute_se=True``.

        Returns
        -------
        preds : ndarray, shape (m,)
        se : ndarray, shape (m,)   (only if return_se=True)
        """
        self._check_fitted()
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_new.shape[1] != self.p_:
            raise ValueError(f"expected {self.p_} predictor column(s), got {X_new.shape[1]}")

        Xs_new = X_new / self.scale_
        m = Xs_new.shape[0]
        preds = np.empty(m)
        ses = np.empty(m) if return_se else None

        for i in range(m):
            val, l_vec = self._local_fit(
                Xs_new[i], self.Xs_train_, self.y_train_,
                self.robustness_weights_, want_l=return_se
            )
            preds[i] = val
            if return_se:
                if self.residual_scale_ is None:
                    raise RuntimeError(
                        "Standard errors requested but compute_se=False at fit time. "
                        "Refit with Loess(..., compute_se=True)."
                    )
                ses[i] = self.residual_scale_ * np.sqrt(np.sum(l_vec ** 2))

        if return_se:
            return preds, ses
        return preds

    def summary(self) -> str:
        self._check_fitted()
        lines = [
            "LOESS fit summary",
            "-----------------",
            f"Number of Observations: {self.n_}",
            f"Number of predictors:   {self.p_}",
            f"Span:                   {self.span}",
            f"Degree:                 {self.degree}",
            f"Family:                 {'symmetric (robust)' if self.robust else 'gaussian'}",
            f"Robustness iterations:  {self.iterations}",
        ]
        if self.enp_ is not None:
            lines.append(f"Equivalent Number of Parameters (enp): {self.enp_:.3f}")
        if self.residual_scale_ is not None:
            lines.append(f"Residual Standard Error:               {self.residual_scale_:.5f}")
        rss = float(np.sum(self.residuals_ ** 2))
        lines.append(f"Residual sum of squares:                {rss:.5f}")
        text = "\n".join(lines)
        print(text)
        return text

    # ---------------------------------------------------------------- #
    # internals
    # ---------------------------------------------------------------- #

    def _check_fitted(self):
        if self.X_train_ is None:
            raise RuntimeError("Loess instance is not fitted yet. Call .fit(X, y) first.")

    def _n_local_points(self, n: int) -> int:
        if self.span <= 1:
            q = int(np.ceil(self.span * n))
            q = max(q, self.degree + 1)
            q = min(q, n)
            return q
        return n

    def _local_fit(self, x0, Xtrain, ytrain, rob_weights, want_l=False):
        """
        Fit the local weighted polynomial centered at x0 using all of
        Xtrain/ytrain and combined (tricube * robustness) weights.
        Returns (fitted_value, l_vector_or_None).

        l_vector is the vector of weights l such that fitted_value = l @
        ytrain -- used for standard-error / hat-matrix-trace computations.
        """
        n = Xtrain.shape[0]
        diff = Xtrain - x0
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        q = self._n_local_points(n)
        if self.span <= 1:
            d_max = np.partition(dist, q - 1)[q - 1]
            if d_max <= 0:
                d_max = np.max(dist) if np.max(dist) > 0 else 1e-12
        else:
            d_max = np.max(dist) * self.span
            if d_max <= 0:
                d_max = 1e-12

        u = dist / d_max
        w = _tricube(u) * rob_weights

        # guard against a degenerate (all-zero) weight vector
        if not np.any(w > 0):
            w = np.ones(n) * 1e-8

        B = _design_matrix(diff, self.exponents_)
        sqrtw = np.sqrt(w)
        Bw = B * sqrtw[:, None]
        yw = ytrain * sqrtw

        # Solve the weighted least squares problem via lstsq for stability.
        beta, *_ = np.linalg.lstsq(Bw, yw, rcond=None)
        fitted_value = float(beta[0])

        l_vec = None
        if want_l:
            # l s.t. fitted = l @ ytrain.
            # beta = pinv(Bw^T Bw) @ Bw^T @ yw = pinv(Bw^T Bw) @ B^T @ (w * ytrain)
            # fitted = beta[0] = row0(pinv(Bw^T Bw) @ B^T) @ (w * ytrain)
            pinv_BtB = np.linalg.pinv(Bw.T @ Bw)
            row0 = pinv_BtB[0, :] @ B.T   # shape (n,)
            l_vec = row0 * w

        return fitted_value, l_vec

    def _compute_enp_and_scale(self):
        """Compute the trace of the smoother (hat) matrix and the residual
        scale estimate, needed for standard errors -- analogous to what
        R reports as 'Equivalent Number of Parameters' and 'Residual
        Standard Error' in summary(loess_fit)."""
        n = self.n_
        diag_L = np.empty(n)
        for i in range(n):
            _, l_vec = self._local_fit(
                self.Xs_train_[i], self.Xs_train_, self.y_train_,
                self.robustness_weights_, want_l=True
            )
            diag_L[i] = l_vec[i]

        trace_L = float(np.sum(diag_L))
        self.enp_ = trace_L
        rss = float(np.sum(self.residuals_ ** 2))
        df = max(n - trace_L, 1e-6)
        self.residual_scale_ = float(np.sqrt(rss / df))


# --------------------------------------------------------------------------
# Convenience functional API (closer to calling loess()/predict() in R)
# --------------------------------------------------------------------------

def loess(x, y, span: float = 0.75, degree: int = 2, robust: bool = False,
          normalize: bool = True) -> Loess:
    """Functional convenience wrapper: fit and return a Loess model."""
    model = Loess(span=span, degree=degree, robust=robust, normalize=normalize)
    model.fit(x, y)
    return model
