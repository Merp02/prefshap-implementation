"""
Are the results stable or did we get lucky with one random seed?

Repeats the order-1 approximator budget sweep for the Charizard vs
Squirtle duel across random seeds

-> the error numbers are a mean +/- std rather than a single lucky/unlucky draw.

Reuses the exact SV already computed and cached in benchmark_results.npz
(no need to pay the 8192-coalition ExactComputer cost again)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from benchmark_pokemon import build_game_for_duel, pick_duel
from pokemon_data import load_item_stats
from prefshap_shapiq_clean import run_sv_approximator, run_interaction_approximator, run_order3_approximator
from benchmark_pokemon import interactions_to_vec

torch.set_default_dtype(torch.float64)

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [200, 1000, 2000]
SV_NAMES = ["KernelSHAP", "RegressionMSR", "PermutationSamplingSV", "SHAPIQ"]
INT_NAMES = ["KernelSHAPIQ", "ProxySHAP", "PermutationSamplingSII", "SHAPIQ"]
O3_NAMES = ["KernelSHAPIQ", "ProxySHAP", "PermutationSamplingSII", "SHAPIQ"]


def main():
    cached = np.load("benchmark_results.npz", allow_pickle=True)
    beta_exact = torch.as_tensor(cached["beta_exact"])
    exact_first = cached["exact_first"]
    exact_pair = cached["exact_pair"]
    exact_triplet = cached["exact_triplet"]

    item_data = load_item_stats()
    x_l, x_r = pick_duel(item_data, "Charizard", "Squirtle")
    game, _ = build_game_for_duel(x_l, x_r)

    sv_rows = []  # (name, budget, seed, max_err, mean_err)
    print("order-1 SV: multi-seed sweep")
    for name in SV_NAMES:
        for budget in BUDGETS:
            errs_max, errs_mean = [], []
            for seed in SEEDS:
                t0 = time.time()
                approx = run_sv_approximator(name, game, budget=budget, random_state=seed)
                dt = time.time() - t0
                beta_approx = torch.as_tensor(approx.to_first_order_array())
                diff = (beta_approx - beta_exact).abs()
                errs_max.append(diff.max().item())
                errs_mean.append(diff.mean().item())
                sv_rows.append((name, budget, seed, errs_max[-1], errs_mean[-1]))
            print(f"  {name:24s} budget={budget:5d}  "
                  f"max_err={np.mean(errs_max):.4f}+-{np.std(errs_max):.4f}  "
                  f"mean_err={np.mean(errs_mean):.4f}+-{np.std(errs_mean):.4f}  ({dt:.2f}s/run)")

    int_rows = []  # (name, budget, seed, max_err1, max_err2)
    print("\norder-2 k-SII: multi-seed sweep")
    for name in INT_NAMES:
        for budget in BUDGETS:
            errs1, errs2 = [], []
            for seed in SEEDS:
                t0 = time.time()
                approx = run_interaction_approximator(name, game, budget=budget, random_state=seed)
                dt = time.time() - t0
                approx_first = interactions_to_vec(approx.dict_values, game.mask, order=1)
                approx_pair = interactions_to_vec(approx.dict_values, game.mask, order=2)
                errs1.append(np.abs(approx_first - exact_first).max())
                errs2.append(np.abs(approx_pair - exact_pair).max())
                int_rows.append((name, budget, seed, errs1[-1], errs2[-1]))
            print(f"  {name:24s} budget={budget:5d}  "
                  f"max_err(order1)={np.mean(errs1):.4f}+-{np.std(errs1):.4f}  "
                  f"max_err(order2)={np.mean(errs2):.4f}+-{np.std(errs2):.4f}  ({dt:.2f}s/run)")

    o3_rows = []  # (name, budget, seed, max_err1, max_err2, max_err3)
    print("\norder-3 k-SII: multi-seed sweep")
    for name in O3_NAMES:
        for budget in BUDGETS:
            errs1, errs2, errs3 = [], [], []
            for seed in SEEDS:
                t0 = time.time()
                approx = run_order3_approximator(name, game, budget=budget, random_state=seed)
                dt = time.time() - t0
                approx_first = interactions_to_vec(approx.dict_values, game.mask, order=1)
                approx_pair = interactions_to_vec(approx.dict_values, game.mask, order=2)
                approx_triplet = interactions_to_vec(approx.dict_values, game.mask, order=3)
                errs1.append(np.abs(approx_first - exact_first).max())
                errs2.append(np.abs(approx_pair - exact_pair).max())
                errs3.append(np.abs(approx_triplet - exact_triplet).max())
                o3_rows.append((name, budget, seed, errs1[-1], errs2[-1], errs3[-1]))
            print(f"  {name:24s} budget={budget:5d}  "
                  f"max_err(order1)={np.mean(errs1):.4f}+-{np.std(errs1):.4f}  "
                  f"max_err(order2)={np.mean(errs2):.4f}+-{np.std(errs2):.4f}  ({dt:.2f}s/run)"
                  f"max_err(order3)={np.mean(errs3):.4f}+-{np.std(errs3):.4f}  ({dt:.2f}s/run)")
                
    np.savez(
        "test_seeds_results.npz",
        sv_rows=np.array(sv_rows, dtype=object),
        int_rows=np.array(int_rows, dtype=object),
        o3_rows=np.array(o3_rows,dtype=object),
        seeds=np.array(SEEDS), budgets=np.array(BUDGETS),
        sv_names=np.array(SV_NAMES), int_names=np.array(INT_NAMES), o3_names=np.array(O3_NAMES)
    )
    print("\nSaved test_seeds_results.npz")


if __name__ == "__main__":
    main()
