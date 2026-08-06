"""
Learn the preference function g(x_l, x_r) for the Pokemon duel dataset using
`pref_fit.fit_preference_model` (Nystrom-whitened kernel logistic
regression on the preferential kernel k_E). 

Produces and saves (`pokemon_g_fit.npz`):
    alpha        (M,)     learned dual coefficients
    Xl_c, Xr_c   (M, d)   the Nystrom centre duels alpha is expanded over
    lengthscale  scalar   RBF lengthscale used for the preferential kernel
    lam          scalar   ridge strength used in the fit
    test_auc     scalar   held-out AUC, sanity check that g learned something
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

import pref_fit as pref_fit

from pokemon_data import build_duels, load_combats, load_item_stats  # noqa: E402


def fit_g(
    n_duels: int = 20000,
    n_centres: int = 400,
    lam: float = 1e-5,
    test_frac: float = 0.2,
    random_state: int = 0,
):
    item_data = load_item_stats()
    combats = load_combats()
    X_l, X_r, y = build_duels(item_data, combats, n_duels=n_duels, random_state=random_state)

    m = len(y)
    k = int((1 - test_frac) * m)
    rng = np.random.default_rng(random_state)
    # train/test-split
    perm = rng.permutation(m)
    tr, te = perm[:k], perm[k:]

    # compute lengthscale for rbf-kernel
    # for training:    pref_fit.rbf(ls)
    # for explanation: rbf_kernel_torch(...)
    # the saved lengthscale guaranties that during training and explaning the same g is used.
    ls = pref_fit.median_lengthscale(np.vstack([X_l[tr], X_r[tr]]), seed=random_state)
    kernel = pref_fit.rbf(ls)

    model = pref_fit.fit_preference_model(
        X_l[tr], X_r[tr], y[tr],
        kernel=kernel, n_centres=n_centres, lam=lam, random_state=random_state,
    )

    # AUC: Can the learned preference function predict unseen Pokémon fights?
    # 0.5 (similar to random guessing) to 1.0 (perfect at distinguishing winners)
    test_auc = model.score(X_l[te], X_r[te], y[te])

    # sanity check: skew-symmetry, g(a,b) = -g(b,a)
    ab = model.decision_function(X_l[:200], X_r[:200])
    ba = model.decision_function(X_r[:200], X_l[:200])
    skew_err = float(np.abs(ab + ba).max())
    # expected = g(a,b) + g(b,a) ~ 0

    print(f"\nd = {X_l.shape[1]}  m_train = {len(tr)}  M (centres) = {model.fit.n_features}")

    print(f"\nmax |g(a,b)+g(b,a)| = {skew_err:.2e}  (should be ~0, kernel is exactly skew-symmetric)")
    np.savez(
        "pokemon_g_fit.npz",
        alpha=model.fit.alpha,
        Xl_c=model.Xl_c,
        Xr_c=model.Xr_c,
        lengthscale=ls,
        lam=lam,
        test_auc=test_auc,
        feature_names=np.array(item_data.feature_names),
    )
    print("\nSaved to: pokemon_g_fit.npz")
    print("Alpha      :", model.alpha.shape)
    print("Xl_c       :", model.Xl_c.shape)
    print("Xr_c       :", model.Xr_c.shape)
    print("lengthscale:", ls)
    print("lam        :", lam)
    print("AUC        :", test_auc)

    return model, item_data, (X_l, X_r, y, tr, te)


if __name__ == "__main__":
    fit_g()
