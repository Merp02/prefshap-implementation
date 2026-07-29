"""
Kernel-agnostic fitting for preference models.

The design separates three things that were previously tangled together:

    layer 1  fit_nystrom_logistic(K_mM, K_MM, y, lam)
             The solver. Knows nothing about kernels, items, duels or
             covariates -- it takes two Gram matrices and labels. Any PSD
             kernel works: RBF, Matern, linear, string kernels, graph
             kernels, a matrix you computed in another language and loaded
             from disk.

    layer 2  preferential(k) / with_context(kE, k_u)
             Combinators that lift a base kernel on items to a skew-symmetric
             kernel on duels, and optionally multiply in a context kernel.
             Pure function composition; use them or ignore them.

    layer 3  fit_preference_model(Xl, Xr, y, kernel=...)
             Convenience wrapper for the common case. Every default it picks
             is overridable, and you can bypass it entirely.

If you already have your kernel matrices, layer 1 is all you need:

    fit = fit_nystrom_logistic(K_mM, K_MM, y, lam=1e-6)
    scores = K_starM @ fit.alpha

Why Nystrom + whitening rather than a dual solver: with K_MM + eps I = L L^T
and Phi = K_mM L^{-T}, the objective becomes plain L2 logistic regression on
Phi, so a standard linear solver handles it. The whitening is what makes it
well conditioned; nothing about it is specific to a choice of kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

__all__ = [
    "fit_nystrom_logistic", "NystromFit",
    "preferential", "with_context",
    "rbf", "matern52", "laplacian", "linear", "polynomial",
    "median_lengthscale",
    "fit_preference_model", "PreferenceModel",
]


# =========================================================================== #
# Layer 1 -- the solver. No kernel assumptions whatsoever.
# =========================================================================== #

@dataclass
class NystromFit:
    """Result of a Nystrom-whitened logistic fit."""
    alpha: np.ndarray            # (M,) dual coefficients: g(.) = sum_i alpha_i k(z_i, .)
    beta: np.ndarray             # (M,) primal coefficients in the whitened basis
    L: np.ndarray                # (M, M) lower Cholesky factor of K_MM + eps I
    lam: float
    jitter_used: float
    n_features: int

    def scores_from_kernel(self, K_starM: np.ndarray) -> np.ndarray:
        """g at new points, given their kernel against the M centres."""
        return np.asarray(K_starM, dtype=float) @ self.alpha


def fit_nystrom_logistic(
    K_mM: np.ndarray,
    K_MM: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-6,
    jitter: float = 1e-6,
    fit_intercept: bool = False,
    solver: str = "auto",
    max_iter: int = 1000,
    tol: float = 1e-8,
    dtype=None,
    chunk_size: int = 20000,
) -> NystromFit:
    """
    Solve  min_a  sum_i log(1 + exp(-y_i (K_mM a)_i)) + lam * m * a^T K_MM a.

    Parameters
    ----------
    K_mM : (m, M) kernel between the m training points and the M centres.
    K_MM : (M, M) kernel among the centres. Must be symmetric PSD.
    y : (m,) labels, {-1,+1} or {0,1}.
    lam : ridge strength.
    jitter : initial diagonal ridge added to K_MM before factorisation;
        escalated by 10x until Cholesky succeeds. Needed for kernels that are
        rank-deficient (the preferential kernel badly so).
    fit_intercept : leave False for skew-symmetric kernels, where a bias term
        would destroy the antisymmetry. Exposed because the solver itself is
        general and other kernels may want one.
    solver : "auto" picks newton-cholesky when m > 5M, else lbfgs.
    dtype : None keeps float64; pass np.float32 to halve the memory of the
        feature matrix on large problems.

    Returns
    -------
    NystromFit

    Notes
    -----
    The only requirements on the kernel are that K_MM be symmetric PSD and
    that K_mM be built with the same centres in the same order. Everything
    else -- stationarity, the form of the feature map, whether the inputs are
    vectors at all -- is irrelevant here.
    """
    K_mM = np.asarray(K_mM, dtype=float)
    K_MM = np.asarray(K_MM, dtype=float)
    y = np.asarray(y).ravel()
    y = np.where(y > 0, 1, -1)

    m, M = K_mM.shape
    if K_MM.shape != (M, M):
        raise ValueError(f"K_MM must be ({M}, {M}), got {K_MM.shape}")
    if len(y) != m:
        raise ValueError(f"y has length {len(y)}, expected {m}")

    K_MM = 0.5 * (K_MM + K_MM.T)

    # --- whitening factor -------------------------------------------------- #
    scale = float(np.trace(K_MM)) / M
    eps = jitter * max(scale, 1e-12)
    L = None
    for _ in range(12):
        try:
            L = cholesky(K_MM + eps * np.eye(M), lower=True)
            break
        except np.linalg.LinAlgError:
            eps *= 10.0
    if L is None:
        raise RuntimeError("could not factor K_MM; increase jitter or use "
                           "fewer centres")

    # --- whitened features, in row blocks ---------------------------------- #
    out_dtype = np.float64 if dtype is None else dtype
    Phi = np.empty((m, M), dtype=out_dtype)
    for s in range(0, m, chunk_size):
        e = min(s + chunk_size, m)
        Phi[s:e] = solve_triangular(L, K_mM[s:e].T, lower=True).T.astype(
            out_dtype, copy=False)

    # --- standard L2 logistic regression ----------------------------------- #
    if solver == "auto":
        solver = "newton-cholesky" if m > 5 * M else "lbfgs"
    clf = LogisticRegression(
        C=1.0 / (2.0 * lam * m),
        fit_intercept=fit_intercept,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
    )
    clf.fit(Phi, y)
    beta = clf.coef_.ravel().astype(np.float64)
    alpha = solve_triangular(L.T, beta, lower=False)

    return NystromFit(alpha=alpha, beta=beta, L=L, lam=float(lam),
                      jitter_used=float(eps), n_features=M)


# =========================================================================== #
# Layer 2 -- kernel combinators. Bring your own base kernel.
# =========================================================================== #
#
# A base kernel is any callable k(A, B) -> (len(A), len(B)) array.
# Below are a few common ones, but a callable is a callable: pass your own,
# including one that indexes a precomputed lookup table for discrete items.

def rbf(lengthscale: float = 1.0) -> Callable:
    def k(A, B):
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / lengthscale**2)
    return k


def matern52(lengthscale: float = 1.0) -> Callable:
    def k(A, B):
        r = cdist(A, B, "euclidean") * (np.sqrt(5.0) / lengthscale)
        return (1.0 + r + r**2 / 3.0) * np.exp(-r)
    return k


def laplacian(lengthscale: float = 1.0) -> Callable:
    def k(A, B):
        return np.exp(-cdist(A, B, "cityblock") / lengthscale)
    return k


def linear(bias: float = 0.0) -> Callable:
    def k(A, B):
        return A @ B.T + bias
    return k


def polynomial(degree: int = 3, coef0: float = 1.0, gamma: float = 1.0) -> Callable:
    def k(A, B):
        return (gamma * (A @ B.T) + coef0) ** degree
    return k


def median_lengthscale(X, n_sub: int = 2000, seed: int = 0) -> float:
    """Median heuristic. A suggestion, not a default imposed on you."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sub, len(X)), replace=False)
    d = cdist(X[idx], X[idx], "sqeuclidean")
    med = np.median(d[np.triu_indices_from(d, k=1)]) if len(idx) > 1 else 0.0
    return float(np.sqrt(med / 2.0)) if med > 1e-12 else 1.0


def preferential(k: Callable) -> Callable:
    """
    Lift a base kernel on items to the skew-symmetric kernel on duels:

        k_E((a,b),(c,d)) = k(a,c) k(b,d) - k(a,d) k(b,c)

    Returns a callable kE(Al, Ar, Bl, Br) -> (len(Al), len(Bl)).
    """
    def kE(Al, Ar, Bl, Br):
        return k(Al, Bl) * k(Ar, Br) - k(Al, Br) * k(Ar, Bl)
    return kE


def with_context(kE: Callable, k_u: Callable) -> Callable:
    """
    Multiply a duel kernel by a context kernel (the C-GPM construction):

        k_E^U((u,a,b),(v,c,d)) = k_u(u,v) * k_E((a,b),(c,d))

    Returns kEU(Al, Ar, Ua, Bl, Br, Ub).
    """
    def kEU(Al, Ar, Ua, Bl, Br, Ub):
        return kE(Al, Ar, Bl, Br) * k_u(Ua, Ub)
    return kEU


# =========================================================================== #
# Layer 3 -- convenience wrapper for the common case
# =========================================================================== #

@dataclass
class PreferenceModel:
    fit: NystromFit
    kE: Callable
    Xl_c: np.ndarray
    Xr_c: np.ndarray
    U_c: Optional[np.ndarray] = None
    centres: Optional[np.ndarray] = None
    chunk_size: int = 20000

    def _kernel_against_centres(self, Xl, Xr, U=None):
        Xl = np.asarray(Xl, dtype=float)
        Xr = np.asarray(Xr, dtype=float)
        n = len(Xl)
        out = np.empty((n, len(self.Xl_c)), dtype=float)
        for s in range(0, n, self.chunk_size):
            e = min(s + self.chunk_size, n)
            if self.U_c is None:
                out[s:e] = self.kE(Xl[s:e], Xr[s:e], self.Xl_c, self.Xr_c)
            else:
                if U is None:
                    raise ValueError("model was fitted with context; pass U=")
                out[s:e] = self.kE(Xl[s:e], Xr[s:e], np.asarray(U, float)[s:e],
                                   self.Xl_c, self.Xr_c, self.U_c)
        return out

    def decision_function(self, Xl, Xr, U=None):
        """g(left, right). Positive favours the left item."""
        return self._kernel_against_centres(Xl, Xr, U) @ self.fit.alpha

    def predict_proba(self, Xl, Xr, U=None):
        g = self.decision_function(Xl, Xr, U)
        return np.where(g >= 0, 1.0 / (1.0 + np.exp(-np.abs(g))),
                        np.exp(-np.abs(g)) / (1.0 + np.exp(-np.abs(g))))

    def predict(self, Xl, Xr, U=None):
        return np.where(self.decision_function(Xl, Xr, U) >= 0, 1, -1)

    def score(self, Xl, Xr, y, U=None):
        return float(roc_auc_score(np.asarray(y).ravel() > 0,
                                   self.decision_function(Xl, Xr, U)))

    @property
    def alpha(self):
        return self.fit.alpha


def fit_preference_model(
    X_left,
    X_right,
    y,
    kernel: Optional[Callable] = None,
    context=None,
    context_kernel: Optional[Callable] = None,
    n_centres: int | str = "auto",
    lam: float = 1e-6,
    random_state: int = 0,
    **solver_kwargs,
) -> PreferenceModel:
    """
    Convenience wrapper. Every choice here is a default you can override.

    kernel : base kernel callable k(A, B) on the item space. If None, an RBF
        with the median-heuristic lengthscale is used -- a starting point, not
        a recommendation. Pass your own to control it.
    context, context_kernel : optional context covariates and their kernel
        (defaults to RBF with median heuristic if context is given).
    n_centres : "auto" uses ~5*sqrt(m) capped at 1500.

    Remaining keyword arguments go to fit_nystrom_logistic.
    """
    Xl = np.asarray(X_left, dtype=float)
    Xr = np.asarray(X_right, dtype=float)
    y = np.asarray(y).ravel()
    m = len(Xl)
    rng = np.random.default_rng(random_state)

    if kernel is None:
        kernel = rbf(median_lengthscale(np.vstack([Xl, Xr]), seed=random_state))
    kE = preferential(kernel)

    U = None
    if context is not None:
        U = np.asarray(context, dtype=float)
        if context_kernel is None:
            context_kernel = rbf(median_lengthscale(U, seed=random_state))
        kE = with_context(kE, context_kernel)

    if n_centres == "auto":
        n_centres = int(min(m, max(100, 5 * np.sqrt(m)), 1500))
    M = int(min(n_centres, m))
    centres = rng.choice(m, size=M, replace=False)
    Xl_c, Xr_c = Xl[centres], Xr[centres]
    U_c = U[centres] if U is not None else None

    if U is None:
        K_MM = kE(Xl_c, Xr_c, Xl_c, Xr_c)
        K_mM = kE(Xl, Xr, Xl_c, Xr_c)
    else:
        K_MM = kE(Xl_c, Xr_c, U_c, Xl_c, Xr_c, U_c)
        K_mM = kE(Xl, Xr, U, Xl_c, Xr_c, U_c)

    fit = fit_nystrom_logistic(K_mM, K_MM, y, lam=lam, **solver_kwargs)
    return PreferenceModel(fit=fit, kE=kE, Xl_c=Xl_c, Xr_c=Xr_c, U_c=U_c,
                           centres=centres)


# =========================================================================== #
# Demo
# =========================================================================== #

def _synthetic(n_items=600, n_matches=12000, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_items, 4))
    c = rng.integers(0, 3, size=n_items)
    X = np.hstack([P, np.eye(3)[c]])
    i = rng.integers(0, n_items, size=n_matches)
    j = rng.integers(0, n_items, size=n_matches)
    keep = i != j
    i, j = i[keep], j[keep]
    f = np.where(c[i] == c[j], 0, np.where(c[i] + c[j] == 1, 1,
                                           np.where(c[i] + c[j] == 2, 2, 3)))
    return X[i], X[j], np.where(P[i, f] > P[j, f], 1, -1)


if __name__ == "__main__":
    Xl, Xr, y = _synthetic()
    k = int(0.8 * len(y))
    tr = slice(0, k)
    te = slice(k, None)

    print("=" * 64)
    print("layer 1: bring your own kernel matrices")
    print("=" * 64)
    ls = median_lengthscale(np.vstack([Xl, Xr]))
    kE = preferential(rbf(ls))
    rng = np.random.default_rng(0)
    cen = rng.choice(k, 400, replace=False)
    Xl_c, Xr_c = Xl[:k][cen], Xr[:k][cen]
    K_MM = kE(Xl_c, Xr_c, Xl_c, Xr_c)
    K_mM = kE(Xl[tr], Xr[tr], Xl_c, Xr_c)
    fit = fit_nystrom_logistic(K_mM, K_MM, y[tr], lam=1e-6)
    K_te = kE(Xl[te], Xr[te], Xl_c, Xr_c)
    print(f"  solver saw only matrices. test AUC = "
          f"{roc_auc_score(y[te] > 0, fit.scores_from_kernel(K_te)):.4f}")
    print(f"  jitter needed: {fit.jitter_used:.2e}")

    print()
    print("=" * 64)
    print("layer 2: swap the base kernel, same solver")
    print("=" * 64)
    for name, base in [
        ("rbf (median ls)", rbf(ls)),
        ("rbf (ls / 2)", rbf(ls / 2)),
        ("matern 5/2", matern52(ls)),
        ("laplacian", laplacian(ls)),
        ("linear", linear()),
        ("polynomial d=3", polynomial(degree=3, gamma=0.2)),
    ]:
        mo = fit_preference_model(Xl[tr], Xr[tr], y[tr], kernel=base, lam=1e-6)
        ab = mo.decision_function(Xl[:150], Xr[:150])
        ba = mo.decision_function(Xr[:150], Xl[:150])
        print(f"  {name:<18} test AUC = {mo.score(Xl[te], Xr[te], y[te]):.4f}"
              f"   skew err = {np.abs(ab + ba).max():.1e}")

    print()
    print("=" * 64)
    print("layer 3: defaults, if you want them")
    print("=" * 64)
    mo = fit_preference_model(Xl[tr], Xr[tr], y[tr])
    print(f"  test AUC = {mo.score(Xl[te], Xr[te], y[te]):.4f}")

    print()
    print("=" * 64)
    print("a non-vector kernel: precomputed lookup over discrete items")
    print("=" * 64)
    n_types = 12
    rng = np.random.default_rng(3)
    S = rng.normal(size=(n_types, n_types))
    S = S @ S.T                                   # arbitrary PSD similarity
    ids_l = rng.integers(0, n_types, size=4000)
    ids_r = rng.integers(0, n_types, size=4000)
    adv = rng.normal(size=(n_types, n_types))
    adv = adv - adv.T                             # skew "type advantage"
    yy = np.where(adv[ids_l, ids_r] > 0, 1, -1)

    def table_kernel(A, B):
        """A, B are (n,1) integer ids; kernel is a table lookup."""
        return S[np.ix_(A[:, 0].astype(int), B[:, 0].astype(int))]

    Al, Ar = ids_l[:, None].astype(float), ids_r[:, None].astype(float)
    kk = int(0.8 * len(yy))
    mo = fit_preference_model(Al[:kk], Ar[:kk], yy[:kk],
                              kernel=table_kernel, n_centres=300, lam=1e-5)
    print(f"  test AUC = {mo.score(Al[kk:], Ar[kk:], yy[kk:]):.4f}"
          f"   (no coordinates involved, only a similarity table)")
