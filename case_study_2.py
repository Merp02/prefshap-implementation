"""
Second local-explanation case study: 
instead of a blowout (Charizard vs Squirtle), 
explain a genuinely contested duel -- two Pokemon with quite
different stat profiles (stat_dist=5.75 in the 6 numeric stats) for which
the learned g is nearly zero (g=+0.0011), i.e. the model calls it a
near-coinflip. Found by scoring ~40k random item pairs with the fitted
preferential kernel and filtering for |g| small AND stat_dist large (to
exclude trivial near-duplicate items, which are also g~0 but for the boring
reason that they're nearly the same Pokemon).

Same pipeline as benchmark_pokemon.py, just a different (x_l, x_r).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pokemon_data import load_item_stats, background_sample
from prefshap_core import active_features_item
from prefshap_shapiq_clean import (
    PrefShapItemGame, make_exact_computer, run_shapiq_exact,
    run_sv_approximator, run_interaction_approximator,
)
from benchmark_pokemon import rbf_kernel_torch, interactions_to_vec, error_table_sv

torch.set_default_dtype(torch.float64)

NAME_L, NAME_R = "Skorupi", "Wailord"



def build_game_for_duel(x_l_row: np.ndarray, x_r_row: np.ndarray, fit_path="pokemon_g_fit.npz",
                        n_background=200, lambda_reg=1e-2, random_state=0):
    # load `pokemon_g_fit.npz`
    data = np.load(fit_path, allow_pickle=True)
    # reconstruct alpha and the centers
    alpha = torch.as_tensor(data["alpha"]).reshape(1, -1)
    X_l = torch.as_tensor(data["Xl_c"])
    X_r = torch.as_tensor(data["Xr_c"])
    # load the saved lengthscale
    ls = float(data["lengthscale"])

    # 200 background pokemons
    item_data = load_item_stats()
    X_bg = torch.as_tensor(background_sample(item_data, n_ref=n_background, random_state=random_state))

    # test pokemons
    x_l = torch.as_tensor(x_l_row).reshape(1, -1)
    x_r = torch.as_tensor(x_r_row).reshape(1, -1)

    mask = active_features_item(X_l, X_r, X_bg)

    def kernel(A, B, S):
        return rbf_kernel_torch(A, B, S, lengthscale=ls)

    game = PrefShapItemGame(
        alpha=alpha, X_l=X_l, X_r=X_r, X=X_bg, x_l=x_l, x_r=x_r,
        kernel=kernel, lambda_reg=lambda_reg, mask=mask,
    )
    return game, item_data


def main():
    print("Loading fitted g and building the explanation game...")
    item_data = load_item_stats()
    row_l, row_r = item_data.names.index(NAME_L), item_data.names.index(NAME_R)
    x_l_row, x_r_row = item_data.stats[row_l], item_data.stats[row_r]

    game, item_data = build_game_for_duel(x_l_row, x_r_row)
    print(f"Case study 2: {NAME_L} vs {NAME_R}  (d_eff={game.n_players})")

    computer = make_exact_computer(game)  # all coalition evaluations happen once, cached
    
    t0 = time.time()
    sv_exact = run_shapiq_exact(computer, index="SV")
    t_exact_sv = time.time() - t0
    beta_exact = torch.as_tensor(sv_exact.to_first_order_array())
    print(f"\nExact SV computed in {t_exact_sv:.1f}s")
    print("v(empty) =", game(np.zeros((1, game.n_players), dtype=bool))[0])
    print("v(full)  =", game(np.ones((1, game.n_players), dtype=bool))[0])

    # Pair each feature name with its computed Shapley value
    print("\nfeature : exact SV")
    for name, v in zip(item_data.feature_names, beta_exact.tolist()):
        print(f"  {name:10s} {v:+.4f}")


    # Find largest contributor by magnitude
    max_idx = beta_exact.abs().argmax().item()
    top_name = item_data.feature_names[max_idx]
    top_val = beta_exact[max_idx].item()

    # Determine direction (positive -> Skorupi, negative -> Wailord)
    target = "Skorupi" if top_val > 0 else "Wailord"

    print(f"\nlargest contributor: {top_name} {top_val:+.4f}")
    print(f"-> {top_name} strongly pushes the model toward {target}.")  


    t0 = time.time()
    order2_exact = run_shapiq_exact(computer, index="k-SII", order=2)
    t_exact_int = time.time() - t0
    print(f"\nExact order-2 k-SII computed in {t_exact_int:.1f}s")


    # exact_first (d_eff,)	        Single feature contributions
    exact_first = interactions_to_vec(order2_exact.dict_values, game.mask, order=1)
    # exact_pair (d_eff, d_eff) 	Symmetric matrix for interactions
    exact_pair = interactions_to_vec(order2_exact.dict_values, game.mask, order=2)

    # --------------------------------------------------------------- #
    # order-1 approximators
    # --------------------------------------------------------------- #
    print("\n" + "-" * 70)
    print("order-1 (Shapley value) approximators vs exact SV")
    print("-" * 70)
    budgets = [200, 500, 1000, 2000]
    sv_names = ["KernelSHAP", "RegressionMSR", "PermutationSamplingSV", "SHAPIQ"]
    sv_results = []
    for name in sv_names:
        for budget in budgets:
            t0 = time.time()
            approx = run_sv_approximator(name, game, budget=budget, random_state=0)
            dt = time.time() - t0
            beta_approx = torch.as_tensor(approx.to_first_order_array())
            err = error_table_sv(beta_exact, beta_approx)
            sv_results.append((name, budget, err["max_err"], err["mean_err"], dt))
            print(f"  {name:24s} budget={budget:5d}  max_err={err['max_err']:.4f}  "
                  f"mean_err={err['mean_err']:.4f}  ({dt:.2f}s)")

    # --------------------------------------------------------------- #
    # order-2 approximators
    # --------------------------------------------------------------- #
    print("\n" + "-" * 70)
    print("order-2 (k-SII interaction) approximators vs exact k-SII")
    print("-" * 70)
    int_names = ["KernelSHAPIQ", "ProxySHAP", "PermutationSamplingSII", "SHAPIQ"]
    int_results = []
    for name in int_names:
        for budget in budgets:
            t0 = time.time()
            approx = run_interaction_approximator(name, game, budget=budget, random_state=0)
            dt = time.time() - t0
            approx_first = interactions_to_vec(approx.dict_values, game.mask, order=1)
            approx_pair = interactions_to_vec(approx.dict_values, game.mask, order=2)
            max_err_1 = np.abs(approx_first - exact_first).max()
            max_err_2 = np.abs(approx_pair - exact_pair).max()
            int_results.append((name, budget, max_err_1, max_err_2, dt))
            print(f"  {name:24s} budget={budget:5d}  max_err(order1)={max_err_1:.4f}  "
                  f"max_err(order2)={max_err_2:.4f}  ({dt:.2f}s)")

    np.savez(
        "benchmark_results_case2.npz",
        name_l=NAME_L, name_r=NAME_R,
        feature_names=np.array(item_data.feature_names),
        beta_exact=beta_exact.numpy(),
        exact_first=exact_first, exact_pair=exact_pair,
        sv_results=np.array(sv_results, dtype=object),
        int_results=np.array(int_results, dtype=object),
    )
    print("\nSaved benchmark_results_case2.npz")


if __name__ == "__main__":
    main()