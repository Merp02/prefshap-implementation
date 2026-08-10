"""
Local explanation + benchmarking case study on the Pokemon dataset.

Pipeline:
    fit_g_pokemon.py  ->  pokemon_g_fit.npz  (learned alpha, centres, lengthscale)
        -> here: wrap as PrefShapItemGame for one specific duel
        -> shapiq.ExactComputer  =>  ground truth SV and order-2 k-SII (d_eff=13, 8192 coalitions)
        -> 4 SV approximators, 4 order-2 approximators, compared against ground truth
           at several budgets.
"""

from __future__ import annotations

# --- OpenMP Conflict Fixes (Must be executed before torch or xgboost imports) ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

import numpy as np
import torch
import xgboost as xgb

from pokemon_data import load_item_stats, background_sample
from prefshap_core import active_features_item
from prefshap_shapiq_clean import (
    PrefShapItemGame, make_exact_computer, run_shapiq_exact,
    run_sv_approximator, run_interaction_approximator, run_order3_approximator
)

torch.set_default_dtype(torch.float64)

# Torch Version of rbf kernel used while training
def rbf_kernel_torch(A, B, S=None, lengthscale=1.0):
    if B is None:
        B = A
    diff = A.unsqueeze(1) - B.unsqueeze(0)
    sqdist = (diff ** 2).sum(-1)
    return torch.exp(-sqdist / (2 * lengthscale ** 2))


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

def pick_duel(item_data, name_l="Charizard", name_r="Squirtle"):
    '''
    Search for the both pokemons (here: "Charizard" and "Squirtle")
    Return feature vectors
    '''
    row_l = item_data.names.index(name_l)
    row_r = item_data.names.index(name_r)
    return item_data.stats[row_l], item_data.stats[row_r]

def error_table_sv(exact_beta: torch.Tensor, approx_beta: torch.Tensor):
    diff = (approx_beta - exact_beta).abs()
    return {"max_err": diff.max().item(), "mean_err": diff.mean().item()}

def interactions_to_vec(values, mask, order):
    """
    Converts dictionary formatted interaction values into clean 1D (vector) and 2D (matrix) NumPy arrays
    """
    active_idx = torch.where(mask)[0].tolist()
    idx_pos = {g: p for p, g in enumerate(active_idx)}
    d_eff = len(active_idx)
    if order == 1:
        out = np.zeros(d_eff)
        for key, val in values.items():
            if len(key) == 1:
                out[key[0]] = val
        return out
    if order == 2:
        out = np.zeros((d_eff, d_eff))
        for key, val in values.items():
            if len(key) == 2:
                i, j = key
                out[i, j] = out[j, i] = val
        return out
    if order == 3 :
        out = np.zeros((d_eff, d_eff, d_eff))
        for key,val in values.items():
            if len(key) == 3:
                i,j,k = key
                out[i,j,k] = out[i,k,j] = out[j,i,k] = out[j,k,i] = out[k,i,j] = out[k,j,i] = val
        return out
    raise ValueError(order)



def main():
    print("Loading fitted g and building the explanation game...")
    # dummy game: just to get item_data cheaply
    _, item_data = build_game_for_duel(np.zeros(13), np.zeros(13))  
    x_l_row, x_r_row = pick_duel(item_data, "Charizard", "Squirtle")
    #actual PREF_SHAP game
    game, item_data = build_game_for_duel(x_l_row, x_r_row)
    print(f"d_eff = {game.n_players}  (2**d_eff = {2**game.n_players} coalitions for the exact reference)")

    computer = make_exact_computer(game)  # 8192 coalition evaluations happen once, cached

    t0 = time.time()
    sv_exact = run_shapiq_exact(computer, index="SV")
    t_exact_sv = time.time() - t0
    beta_exact = torch.as_tensor(sv_exact.to_first_order_array())
    print(f"\nExact SV computed in {t_exact_sv:.1f}s")
    print("v(empty) =", game(np.zeros((1, game.n_players), dtype=bool))[0])
    print("v(full)  =", game(np.ones((1, game.n_players), dtype=bool))[0])
    print("sum(beta_exact) =", beta_exact.sum().item())

    # Pair each feature name with its computed Shapley value
    print("\nfeature : exact SV")
    for name, v in zip(item_data.feature_names, beta_exact.tolist()):
        print(f"  {name:10s} {v:+.4f}")

    # Find largest contributor by magnitude
    max_idx = beta_exact.abs().argmax().item()
    top_name = item_data.feature_names[max_idx]
    top_val = beta_exact[max_idx].item()

    # Determine direction (positive -> Charizard, negative -> Squirtle)
    target = "Charizard" if top_val > 0 else "Squirtle"

    print(f"\nlargest contributor: {top_name} {top_val:+.4f}")
    print(f"-> {top_name} strongly pushes the model toward {target}.")  
    
    t0 = time.time()
    order2_exact = run_shapiq_exact(computer, index="k-SII", order=2)
    t_exact_int = time.time() - t0
    print(f"\nExact order-2 k-SII computed in {t_exact_int:.1f}s")

    t0 = time.time()
    order3_exact = run_shapiq_exact(computer, index="k-SII", order=3)
    t_exact_o3 = time.time() - t0
    print(f"\nExact order-3 k-SII computed in {t_exact_int:.1f}s") 


    # Note: order3_exact contains all orders up to 3, so you can extract all from order3_exact

    # exact_first (d_eff,)	                    Single feature contributions
    exact_first = interactions_to_vec(order3_exact.dict_values, game.mask, order=1)
    # exact_pair (d_eff, d_eff) 	            Symmetric matrix for interactions
    exact_pair = interactions_to_vec(order3_exact.dict_values, game.mask, order=2)
    # exact:triplet = (d_eff, d_eff, d_eff)     3d tensor
    exact_triplet = interactions_to_vec(order3_exact.dict_values, game.mask, order=3)

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
            try:
                approx = run_sv_approximator(name, game, budget=budget, random_state=0)
            except Exception as e:
                print(f"  {name:24s} budget={budget:5d}  FAILED: {e}")
                continue
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
            try:
                approx = run_interaction_approximator(name, game, budget=budget, random_state=0)
            except Exception as e:
                print(f"  {name:24s} budget={budget:5d}  FAILED: {e}")
                continue
            dt = time.time() - t0
            approx_first = interactions_to_vec(approx.dict_values, game.mask, order=1)
            approx_pair = interactions_to_vec(approx.dict_values, game.mask, order=2)
            max_err_1 = np.abs(approx_first - exact_first).max()
            max_err_2 = np.abs(approx_pair - exact_pair).max()
            int_results.append((name, budget, max_err_1, max_err_2, dt))
            print(f"  {name:24s} budget={budget:5d}  max_err(order1)={max_err_1:.4f}  "
                  f"max_err(order2)={max_err_2:.4f}  ({dt:.2f}s)")

    # --------------------------------------------------------------- #
    # order-3 approximators
    # --------------------------------------------------------------- #
    print("\n" + "-" * 70)
    print("order-3 (k-SII interaction) approximators vs exact k-SII")
    print("-" * 70)
    int_names = ["KernelSHAPIQ", "ProxySHAP", "PermutationSamplingSII", "SHAPIQ"]
    int_results = []
    for name in int_names:
        for budget in budgets:
            t0 = time.time()
            try:
                approx = run_order3_approximator(name, game, budget=budget, random_state=0)
            except Exception as e:
                print(f"  {name:24s} budget={budget:5d}  FAILED: {e}")
                continue
            dt = time.time() - t0
            approx_first = interactions_to_vec(approx.dict_values, game.mask, order=1)
            approx_pair = interactions_to_vec(approx.dict_values, game.mask, order=2)
            approx_triplet = interactions_to_vec(approx.dict_values, game.mask, order=3)
            max_err_1 = np.abs(approx_first - exact_first).max()
            max_err_2 = np.abs(approx_pair - exact_pair).max()
            max_err_3 = np.abs(approx_triplet - exact_triplet).max()
            int_results.append((name, budget, max_err_1, max_err_2,max_err_3, dt))
            print(f"  {name:24s} budget={budget:5d}  max_err(order1)={max_err_1:.4f}  "
                  f"max_err(order2)={max_err_2:.4f} max_err(order3)={max_err_3:.4f} ({dt:.2f}s)")


    np.savez(
        "benchmark_results.npz",
        feature_names=np.array(item_data.feature_names),
        beta_exact=beta_exact.numpy(),
        exact_first=exact_first, exact_pair=exact_pair,exact_triplet=exact_triplet,
        sv_results=np.array(sv_results, dtype=object),
        int_results=np.array(int_results, dtype=object),
    )
    print("\nSaved benchmark_results.npz")


if __name__ == "__main__":
    main()
