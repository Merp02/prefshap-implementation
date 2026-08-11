"""Global PREF-SHAP aggregation over multiple Pokemon duels, budgets, and seeds.

It keeps two questions separate:

1. GLOBAL EXPLANATION
   Which single features, feature pairs, and feature triples are important across
   many Pokemon combats?

2. APPROXIMATOR ROBUSTNESS
   Does that global picture remain stable across sampling budgets and random seeds,
   and (optionally) how far is it from exact ground truth?

The default configuration is deliberately small so it can be run on my laptop:
    n_duels = 3
    budgets = [200, 1000, 2000]
    seeds   = [0, 1, 2]


Important orientation
---------------------
Each observed combat is re-oriented as WINNER (left) vs LOSER (right). The
PREF-SHAP game is skew-symmetric, so random left/right orientation would cause
signed global values to cancel. With winner-left orientation:

    positive contribution  -> pushes the model toward the observed winner
    negative contribution  -> pushes the model toward the observed loser

Global importance
-----------------
For a fixed budget b and duel m, multiple random-seed estimates of a term theta
are first averaged:

    theta_bar[m,b] = mean_seed theta_hat[m,b,seed]

The final global importance is then

    mean_duel |theta_bar[m,b]|.

This prevents pure Monte-Carlo seed noise from artificially inflating global
importance. Signed means and positive rates are also saved.

Exact mode
----------
If ``--compute-exact`` is supplied, one ExactComputer is created per duel and
reused for SV, order-2 k-SII, and order-3 k-SII. This costs 2**13 = 8192 coalition
values per duel in the reduced Pokemon representation, but gives exact global
values and multi-duel approximation error curves.

Outputs
-------
The output folder contains raw local explanations, robust global summaries,
seed-uncertainty summaries, optional exact values/errors, and plots.

"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_pokemon import build_game_for_duel
from pokemon_data import load_combats, load_item_stats
from prefshap_shapiq_clean import (
    make_exact_computer,
    run_interaction_approximator,
    run_shapiq_exact,
    run_sv_approximator,
    run_order3_approximator
)


torch.set_default_dtype(torch.float64)

SV_ALL = ["KernelSHAP", "RegressionMSR", "PermutationSamplingSV", "SHAPIQ"]
INT_ALL = ["KernelSHAPIQ", "ProxySHAP", "PermutationSamplingSII", "SHAPIQ"]


# ---------------------------------------------------------------------------
# Data / duel selection
# ---------------------------------------------------------------------------

def _resolve_combat_columns(combats: pd.DataFrame) -> Tuple[str, str, str]:
    lower = {c.lower(): c for c in combats.columns}
    for a, b, w in [
        ("first_pokemon", "second_pokemon", "winner"),
        ("first pokemon", "second pokemon", "winner"),
    ]:
        if a in lower and b in lower and w in lower:
            return lower[a], lower[b], lower[w]
    raise KeyError(
        "Could not find First_pokemon / Second_pokemon / Winner columns. "
        f"Found: {list(combats.columns)}"
    )


def sample_winner_left_duels(n_duels: int, seed: int) -> Tuple[pd.DataFrame, object]:
    """Sample observed combats and always place the observed winner on the left."""
    item_data = load_item_stats()
    combats = load_combats().copy()
    c_first, c_second, c_winner = _resolve_combat_columns(combats)

    if n_duels > len(combats):
        raise ValueError(f"n_duels={n_duels} > number of combats={len(combats)}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(combats), size=n_duels, replace=False)

    rows = []
    for duel_id, combat_row in enumerate(chosen):
        row = combats.iloc[int(combat_row)]
        first_id = int(row[c_first])
        second_id = int(row[c_second])
        winner_id = int(row[c_winner])
        loser_id = second_id if winner_id == first_id else first_id

        winner_row = int(item_data.id_to_row[winner_id])
        loser_row = int(item_data.id_to_row[loser_id])
        rows.append(
            {
                "duel_id": duel_id,
                "combat_row": int(combat_row),
                "winner_id": winner_id,
                "loser_id": loser_id,
                "winner_row": winner_row,
                "loser_row": loser_row,
                "winner_name": item_data.names[winner_row],
                "loser_name": item_data.names[loser_row],
            }
        )

    return pd.DataFrame(rows), item_data


# ---------------------------------------------------------------------------
# Interaction extraction with active->original feature mapping
# ---------------------------------------------------------------------------

def _active_to_original(game) -> List[int]:
    if hasattr(game, "_active_idx"):
        idx = game._active_idx.detach().cpu().numpy().astype(int).tolist()
    else:
        idx = torch.where(game.mask)[0].detach().cpu().numpy().astype(int).tolist()
    return idx


def extract_sv_full(iv, game, feature_names: Sequence[str]) -> List[dict]:
    active_idx = _active_to_original(game)
    arr = np.asarray(iv.to_first_order_array(), dtype=float)
    if len(arr) != len(active_idx):
        raise RuntimeError(f"SV length {len(arr)} != number active features {len(active_idx)}")

    rows = []
    for active_pos, value in enumerate(arr):
        original_i = active_idx[active_pos]
        rows.append(
            {
                "feature_idx": original_i,
                "feature": feature_names[original_i],
                "value": float(value),
            }
        )
    return rows


def extract_exact_order2(iv, game, feature_names: Sequence[str]) -> List[dict]:
    active_idx = _active_to_original(game)
    rows = []
    
    for key, value in iv.dict_values.items():
        if len(key) != 2:
            continue
            
        original = tuple(active_idx[int(k)] for k in key)
        names = tuple(feature_names[i] for i in original)
        
        # Sort or unpack directly since length is guaranteed to be 2
        idx_i, idx_j = original
        name_i, name_j = names

        rows.append({
            "value": float(value),
            "interaction": f"{name_i} × {name_j}",
            "i": int(idx_i),
            "j": int(idx_j),
            "feature_i": name_i,
            "feature_j": name_j
        })
        
    return rows

def extract_exact_order(iv, game, feature_names: Sequence[str], order: int) -> List[dict]:
    active_idx = _active_to_original(game)
    rows = []
    for key, value in iv.dict_values.items():
        if len(key) != order:
            continue
        original = tuple(active_idx[int(k)] for k in key)
        names = tuple(feature_names[i] for i in original)

        row = {"value": float(value), "interaction": " × ".join(names)}
        for pos, (idx, name) in enumerate(zip(original, names)):
            row[chr(ord("i") + pos)] = int(idx)  # i, j, k
            row[f"feature_{chr(ord('i') + pos)}"] = name
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# CSV checkpoint helpers
# ---------------------------------------------------------------------------

def _append_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _reset_outputs(out: Path) -> None:
    for name in [
        "duels.csv",
        "sv_approx.csv",
        "pair_approx.csv",
        "triple_approx.csv",
        "sv_exact.csv",
        "pair_exact.csv",
        "triple_exact.csv",
        "completed_duels.txt",
    ]:
        p = out / name
        if p.exists():
            p.unlink()


def _read_completed(out: Path) -> set[int]:
    p = out / "completed_duels.txt"
    if not p.exists():
        return set()
    return {int(x.strip()) for x in p.read_text().splitlines() if x.strip()}


def _mark_completed(out: Path, duel_id: int) -> None:
    with (out / "completed_duels.txt").open("a") as f:
        f.write(f"{duel_id}\n")


def _remove_partial_duel_rows(out: Path, duel_id: int) -> None:
    """Make resume safe if a previous run stopped halfway through this duel."""
    for name in [
        "sv_approx.csv", "pair_approx.csv", "triple_approx.csv",
        "sv_exact.csv", "pair_exact.csv", "triple_exact.csv",
    ]:
        path = out / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "duel_id" in df.columns and (df["duel_id"] == duel_id).any():
            df = df[df["duel_id"] != duel_id]
            if len(df) == 0:
                path.unlink()
            else:
                df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute(
    n_duels: int,
    budgets: Sequence[int],
    seeds: Sequence[int],
    sv_methods: Sequence[str],
    int_methods: Sequence[str],
    o3_methods: Sequence[str],
    sample_seed: int,
    compute_exact: bool,
    n_background: int,
    lambda_reg: float,
    out_dir: str,
    resume: bool,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "n_duels": int(n_duels),
        "budgets": [int(x) for x in budgets],
        "seeds": [int(x) for x in seeds],
        "sv_methods": list(sv_methods),
        "int_methods": list(int_methods),
        "o3_methods": list(o3_methods),
        "sample_seed": int(sample_seed),
        "compute_exact": bool(compute_exact),
        "n_background": int(n_background),
        "lambda_reg": float(lambda_reg),
    }
    config_path = out / "run_config.json"

    if resume:
        if not config_path.exists():
            raise RuntimeError("--resume requested but run_config.json is missing")
        old = json.loads(config_path.read_text())
        if old != config:
            raise RuntimeError(
                "--resume configuration differs from the previous run. "
                "Use a new output folder or rerun without --resume."
            )
    else:
        _reset_outputs(out)
        config_path.write_text(json.dumps(config, indent=2))

    duel_table, item_data = sample_winner_left_duels(n_duels=n_duels, seed=sample_seed)
    feature_names = list(item_data.feature_names)
    completed = _read_completed(out)

    # Save the selected duel list once. On resume it will be recreated identically.
    if not (out / "duels.csv").exists():
        duel_table.to_csv(out / "duels.csv", index=False)

    for row in duel_table.itertuples(index=False):
        duel_id = int(row.duel_id)
        if duel_id in completed:
            print(f"[skip] duel {duel_id}: already completed")
            continue

        # If the previous process stopped halfway through this duel, remove its
        # partial rows before recomputing it. Completed duels are never touched.
        _remove_partial_duel_rows(out, duel_id)

        x_w = np.asarray(item_data.stats[int(row.winner_row)], dtype=float)
        x_l = np.asarray(item_data.stats[int(row.loser_row)], dtype=float)

        # Fixed background seed for every duel keeps the reference distribution
        # comparable across all local explanations.
        game, _ = build_game_for_duel(
            x_w,
            x_l,
            n_background=n_background,
            lambda_reg=lambda_reg,
            random_state=0,
        )

        full = np.ones((1, game.n_players), dtype=bool)
        v_full = float(game(full)[0])
        print(
            f"\nDuel {duel_id + 1}/{n_duels}: "
            f"{row.winner_name} > {row.loser_name} | g={v_full:+.4f}"
        )

        common_meta = {
            "duel_id": duel_id,
            "winner_name": row.winner_name,
            "loser_name": row.loser_name,
            "v_full": v_full,
        }

        # ---- exact reference -------------------------------------------------
        if compute_exact:
            t0 = time.time()
            computer = make_exact_computer(game)
            exact_sv = run_shapiq_exact(computer, index="SV")
            exact_o2 = run_shapiq_exact(computer, index="k-SII", order=2)
            exact_o3 = run_shapiq_exact(computer, index="k-SII", order=3)
            dt = time.time() - t0

            rows = [{**common_meta, **r} for r in extract_sv_full(exact_sv, game, feature_names)]
            _append_csv(rows, out / "sv_exact.csv")
            rows = [
                {**common_meta, **r}
                for r in extract_exact_order(exact_o2, game, feature_names, order=2)
            ]
            _append_csv(rows, out / "pair_exact.csv")
            rows = [
                {**common_meta, **r}
                for r in extract_exact_order(exact_o3, game, feature_names, order=3)
            ]
            _append_csv(rows, out / "triple_exact.csv")
            print(f"  exact SV + order2 + order3: {dt:.1f}s")

        # ---- approximations --------------------------------------------------
        for budget in budgets:
            for seed in seeds:
                for method in sv_methods:
                    t0 = time.time()
                    iv = run_sv_approximator(method, game, budget=budget, random_state=seed)
                    rows = []
                    for r in extract_sv_full(iv, game, feature_names):
                        rows.append(
                            {
                                **common_meta,
                                "method": method,
                                "budget": int(budget),
                                "seed": int(seed),
                                **r,
                            }
                        )
                    _append_csv(rows, out / "sv_approx.csv")
                    print(f"  SV  {method:18s} b={budget:4d} seed={seed}: {time.time()-t0:.1f}s")

                for method in int_methods:
                    t0 = time.time()
                    iv = run_interaction_approximator(
                        method, game, budget=budget, random_state=seed
                    )
                    rows = []
                    for r in extract_exact_order2(iv, game, feature_names):
                        rows.append(
                            {
                                **common_meta,
                                "method": method,
                                "budget": int(budget),
                                "seed": int(seed),
                                **r,
                            }
                        )
                    _append_csv(rows, out / "pair_approx.csv")
                    print(f"  O2  {method:18s} b={budget:4d} seed={seed}: {time.time()-t0:.1f}s")

                for method in o3_methods:
                    t0 = time.time()
                    iv = run_order3_approximator(
                        method, game, budget=budget, random_state=seed
                    )
                    rows = []
                    for r in extract_exact_order(iv, game, feature_names, order=3):
                        rows.append(
                            {
                                **common_meta,
                                "method": method,
                                "budget": int(budget),
                                "seed": int(seed),
                                **r,
                            }
                        )
                    _append_csv(rows, out / "triple_approx.csv")
                    print(f"  O3  {method:18s} b={budget:4d} seed={seed}: {time.time()-t0:.1f}s")

        _mark_completed(out, duel_id)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _robust_summary(raw: pd.DataFrame, term_cols: Sequence[str]) -> pd.DataFrame:
    """Average seeds per duel first, then aggregate the stable local values over duels."""
    group_local = ["method", "budget", "duel_id", *term_cols]
    per_duel = (
        raw.groupby(group_local, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "seed_mean_value"})
    )

    group_global = ["method", "budget", *term_cols]
    tmp = per_duel.copy()
    tmp["abs_value"] = tmp["seed_mean_value"].abs()
    tmp["positive"] = (tmp["seed_mean_value"] > 0).astype(float)

    out = (
        tmp.groupby(group_global, as_index=False)
        .agg(
            n_duels=("duel_id", "nunique"),
            mean_abs=("abs_value", "mean"),
            std_abs_duels=("abs_value", "std"),
            mean_signed=("seed_mean_value", "mean"),
            std_signed_duels=("seed_mean_value", "std"),
            positive_rate=("positive", "mean"),
        )
        .sort_values(["budget", "mean_abs"], ascending=[True, False])
    )
    return out


def _seed_uncertainty(raw: pd.DataFrame, term_cols: Sequence[str]) -> pd.DataFrame:
    """Global importance per seed, followed by mean/std across seeds."""
    x = raw.copy()
    x["abs_value"] = x["value"].abs()
    per_seed = (
        x.groupby(["method", "budget", "seed", *term_cols], as_index=False)
        .agg(
            global_mean_abs=("abs_value", "mean"),
            global_mean_signed=("value", "mean"),
        )
    )

    return (
        per_seed.groupby(["method", "budget", *term_cols], as_index=False)
        .agg(
            mean_abs_over_seeds=("global_mean_abs", "mean"),
            std_abs_over_seeds=("global_mean_abs", "std"),
            mean_signed_over_seeds=("global_mean_signed", "mean"),
            std_signed_over_seeds=("global_mean_signed", "std"),
            n_seeds=("seed", "nunique"),
        )
    )


def _exact_summary(raw: pd.DataFrame, term_cols: Sequence[str]) -> pd.DataFrame:
    x = raw.copy()
    x["abs_value"] = x["value"].abs()
    x["positive"] = (x["value"] > 0).astype(float)
    return (
        x.groupby(list(term_cols), as_index=False)
        .agg(
            n_duels=("duel_id", "nunique"),
            mean_abs=("abs_value", "mean"),
            std_abs_duels=("abs_value", "std"),
            mean_signed=("value", "mean"),
            std_signed_duels=("value", "std"),
            positive_rate=("positive", "mean"),
        )
        .sort_values("mean_abs", ascending=False)
    )


def _error_summary(
    approx: pd.DataFrame,
    exact: pd.DataFrame,
    match_cols: Sequence[str],
    order_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    e = exact[["duel_id", *match_cols, "value"]].rename(columns={"value": "exact_value"})
    a = approx.merge(e, on=["duel_id", *match_cols], how="inner")
    a["abs_error"] = (a["value"] - a["exact_value"]).abs()
    a["sq_error"] = (a["value"] - a["exact_value"]) ** 2

    per_duel_seed = (
        a.groupby(["method", "budget", "seed", "duel_id"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            max_abs_error=("abs_error", "max"),
            n_terms=("abs_error", "size"),
        )
    )
    per_duel_seed["order"] = order_name

    per_seed = (
        per_duel_seed.groupby(["order", "method", "budget", "seed"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            mse=("mse", "mean"),
            max_abs_error=("max_abs_error", "mean"),
            n_duels=("duel_id", "nunique"),
        )
    )
    per_seed["order"] = order_name

    agg = (
        per_seed.groupby(["order", "method", "budget"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            max_abs_error_mean=("max_abs_error", "mean"),
            max_abs_error_std=("max_abs_error", "std"),
            n_seeds=("seed", "nunique"),
            n_duels=("n_duels", "max"),
        )
    )
    return per_duel_seed, per_seed, agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_budget_profiles(
    robust: pd.DataFrame,
    uncertainty: pd.DataFrame,
    term_col: str,
    top_n: int,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    methods = robust["method"].unique().tolist()
    max_budget = robust["budget"].max()

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        top = (
            robust[(robust["method"] == method) & (robust["budget"] == max_budget)]
            .nlargest(top_n, "mean_abs")[term_col]
            .tolist()
        )
        for term in top:
            sub = uncertainty[
                (uncertainty["method"] == method) & (uncertainty[term_col] == term)
            ].sort_values("budget")
            label = f"{term}" if len(methods) == 1 else f"{method}: {term}"
            ax.errorbar(
                sub["budget"],
                sub["mean_abs_over_seeds"],
                yerr=sub["std_abs_over_seeds"].fillna(0.0),
                marker="o",
                capsize=3,
                label=label,
            )

    ax.set_xlabel("Coalition evaluation budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top_at_budget(
    summary: pd.DataFrame,
    term_col: str,
    top_n: int,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    budget = summary["budget"].max()
    # If several methods are requested, keep method in the label.
    sub = summary[summary["budget"] == budget].copy()
    if sub["method"].nunique() > 1:
        sub["label"] = sub["method"] + ": " + sub[term_col].astype(str)
    else:
        sub["label"] = sub[term_col].astype(str)
    top = sub.nlargest(top_n, "mean_abs").sort_values("mean_abs")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(top) + 2)))
    ax.barh(top["label"], top["mean_abs"])
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title} (budget={budget})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_error_curves(error_agg: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    for (order, method), sub in error_agg.groupby(["order", "method"]):
        sub = sub.sort_values("budget")
        ax.errorbar(
            sub["budget"],
            sub[mean_col],
            yerr=sub[std_col].fillna(0.0),
            marker="o",
            capsize=3,
            label=f"{order}: {method}",
        )
    ax.set_xlabel("Coalition evaluation budget")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Multi-duel approximation error (mean ± std over seeds)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_approximator_evaluation(
    error_agg: pd.DataFrame,
    order_name: str,
    metric: str,
    out_path: Path,
    title: str,
    log_scale: bool = True,
) -> None:
    """Plot the four approximators for one explanation order."""
    sub_all = error_agg[error_agg["order"] == order_name].copy()
    if sub_all.empty:
        print(f"[skip] no evaluation rows for {order_name}: {out_path.name}")
        return

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    metric_labels = {
        "mae": "Mean Absolute Error (MAE)",
        "mse": "Mean Squared Error (MSE)",
        "max_abs_error": "Mean duel-wise Max Absolute Error",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, sub in sub_all.groupby("method", sort=False):
        sub = sub.sort_values("budget")
        ax.errorbar(
            sub["budget"],
            sub[mean_col],
            yerr=sub[std_col].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=4,
            label=method,
        )

    n_seeds = int(sub_all["n_seeds"].max()) if "n_seeds" in sub_all else 0
    n_duels = int(sub_all["n_duels"].max()) if "n_duels" in sub_all else 0
    ax.set_xlabel("Budget (coalition evaluations)")
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(
        f"{title}\n"
        f"mean ± std over {n_seeds} seeds; averaged across {n_duels} duels"
    )
    if log_scale and np.all(sub_all[mean_col].to_numpy(dtype=float) > 0):
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def summarize_and_plot(out: Path) -> None:
    files = {
        "sv": out / "sv_approx.csv",
        "pair": out / "pair_approx.csv",
        "triple": out / "triple_approx.csv",
    }
    if not all(p.exists() for p in files.values()):
        raise RuntimeError("Approximation CSV files are incomplete; computation may not have finished.")

    sv = pd.read_csv(files["sv"])
    pair = pd.read_csv(files["pair"])
    triple = pd.read_csv(files["triple"])

    # Robust global summaries: average seeds within each duel first.
    sv_g = _robust_summary(sv, ["feature_idx", "feature"])
    p2_g = _robust_summary(pair, ["i", "j", "interaction"])
    p3_g = _robust_summary(triple, ["i", "j", "k", "interaction"])
    sv_u = _seed_uncertainty(sv, ["feature_idx", "feature"])
    p2_u = _seed_uncertainty(pair, ["i", "j", "interaction"])
    p3_u = _seed_uncertainty(triple, ["i", "j", "k", "interaction"])

    sv_g.to_csv(out / "global_sv_summary.csv", index=False)
    p2_g.to_csv(out / "global_order2_summary.csv", index=False)
    p3_g.to_csv(out / "global_order3_summary.csv", index=False)
    sv_u.to_csv(out / "global_sv_seed_uncertainty.csv", index=False)
    p2_u.to_csv(out / "global_order2_seed_uncertainty.csv", index=False)
    p3_u.to_csv(out / "global_order3_seed_uncertainty.csv", index=False)

    _plot_budget_profiles(
        sv_g, sv_u, "feature", 6,
        "Global single-feature PREF-SHAP importance across budgets",
        "Mean |Shapley value| across duels",
        out / "global_sv_budget_profiles.png",
    )
    _plot_budget_profiles(
        p2_g, p2_u, "interaction", 6,
        "Global pairwise PREF-SHAP interaction importance across budgets",
        "Mean |order-2 k-SII| across duels",
        out / "global_order2_budget_profiles.png",
    )
    _plot_budget_profiles(
        p3_g, p3_u, "interaction", 8,
        "Global three-way PREF-SHAP interaction importance across budgets",
        "Mean |order-3 k-SII| across duels",
        out / "global_order3_budget_profiles.png",
    )

    _plot_top_at_budget(
        sv_g, "feature", 13,
        "Global single-feature PREF-SHAP importance",
        "Mean |seed-averaged Shapley value| across duels",
        out / "global_sv_top.png",
    )
    _plot_top_at_budget(
        p2_g, "interaction", 15,
        "Top global pairwise interactions",
        "Mean |seed-averaged order-2 k-SII| across duels",
        out / "global_order2_top.png",
    )
    _plot_top_at_budget(
        p3_g, "interaction", 20,
        "Top global three-way interactions",
        "Mean |seed-averaged order-3 k-SII| across duels",
        out / "global_order3_top.png",
    )

    # Exact summaries + error curves, if exact was requested.
    exact_files = [out / "sv_exact.csv", out / "pair_exact.csv", out / "triple_exact.csv"]
    if all(p.exists() for p in exact_files):
        sv_e = pd.read_csv(exact_files[0])
        p2_e = pd.read_csv(exact_files[1])
        p3_e = pd.read_csv(exact_files[2])

        _exact_summary(sv_e, ["feature_idx", "feature"]).to_csv(
            out / "global_sv_exact_summary.csv", index=False
        )
        _exact_summary(p2_e, ["i", "j", "interaction"]).to_csv(
            out / "global_order2_exact_summary.csv", index=False
        )
        _exact_summary(p3_e, ["i", "j", "k", "interaction"]).to_csv(
            out / "global_order3_exact_summary.csv", index=False
        )

        duel_parts, seed_parts, agg_parts = [], [], []
        d, s, a = _error_summary(sv, sv_e, ["feature_idx"], "SV")
        duel_parts.append(d); seed_parts.append(s); agg_parts.append(a)
        d, s, a = _error_summary(pair, p2_e, ["i", "j"], "order-2")
        duel_parts.append(d); seed_parts.append(s); agg_parts.append(a)
        d, s, a = _error_summary(triple, p3_e, ["i", "j", "k"], "order-3")
        duel_parts.append(d); seed_parts.append(s); agg_parts.append(a)

        err_duel_seed = pd.concat(duel_parts, ignore_index=True)
        err_seed = pd.concat(seed_parts, ignore_index=True)
        err_agg = pd.concat(agg_parts, ignore_index=True)

        err_duel_seed.to_csv(out / "approximation_error_per_duel_seed.csv", index=False)
        err_seed.to_csv(out / "approximation_error_per_seed.csv", index=False)
        err_agg.to_csv(out / "approximation_error_summary.csv", index=False)
        print(f"[saved] {out / 'approximation_error_per_duel_seed.csv'}")
        print(f"[saved] {out / 'approximation_error_per_seed.csv'}")
        print(f"[saved] {out / 'approximation_error_summary.csv'}")

        # Dedicated CSVs for each explanation order.
        for order_name, slug in [("SV", "sv"), ("order-2", "order2"), ("order-3", "order3")]:
            order_df = err_agg[err_agg["order"] == order_name].sort_values(
                ["method", "budget"]
            )
            csv_path = out / f"approximator_eval_{slug}.csv"
            order_df.to_csv(csv_path, index=False)
            print(f"[saved] {csv_path}")

        all_eval_path = out / "approximator_eval_all.csv"
        err_agg.sort_values(["order", "method", "budget"]).to_csv(
            all_eval_path, index=False
        )
        print(f"[saved] {all_eval_path}")

        # Combined cross-order figures (useful overview).
        _plot_error_curves(err_agg, "mae", out / "multi_duel_error_mae.png")
        _plot_error_curves(err_agg, "mse", out / "multi_duel_error_mse.png")
        _plot_error_curves(err_agg, "max_abs_error", out / "multi_duel_error_max.png")

        # The requested 4-approximator figures, separately for SV/order-2/order-3.
        order_specs = [
            ("SV", "sv", "Order-1 SV approximators"),
            ("order-2", "order2", "Order-2 k-SII interaction approximators"),
            ("order-3", "order3", "Order-3 k-SII interaction approximators"),
        ]
        for order_name, slug, title in order_specs:
            _plot_approximator_evaluation(
                err_agg, order_name, "max_abs_error",
                out / f"approximator_eval_{slug}.png", title, log_scale=True
            )
            _plot_approximator_evaluation(
                err_agg, order_name, "mae",
                out / f"approximator_eval_{slug}_mae.png", title + " — MAE", log_scale=True
            )
            _plot_approximator_evaluation(
                err_agg, order_name, "mse",
                out / f"approximator_eval_{slug}_mse.png", title + " — MSE", log_scale=True
            )

        manifest = out / "APPROXIMATOR_OUTPUTS.txt"
        manifest.write_text(
            "Approximator evaluation outputs\n"
            "================================\n"
            "approximator_eval_sv.csv\n"
            "approximator_eval_order2.csv\n"
            "approximator_eval_order3.csv\n"
            "approximator_eval_all.csv\n"
            "approximator_eval_sv.png\n"
            "approximator_eval_sv_mae.png\n"
            "approximator_eval_sv_mse.png\n"
            "approximator_eval_order2.png\n"
            "approximator_eval_order2_mae.png\n"
            "approximator_eval_order2_mse.png\n"
            "approximator_eval_order3.png\n"
            "approximator_eval_order3_mae.png\n"
            "approximator_eval_order3_mse.png\n"
        )
        print(f"[saved] {manifest}")

    # Console report at largest budget.
    bmax = int(sv_g["budget"].max())
    print("\n" + "=" * 78)
    print(f"GLOBAL SUMMARY AT BUDGET {bmax}")
    print("=" * 78)
    print("\nTop single features:")
    print(
        sv_g[sv_g["budget"] == bmax]
        .nlargest(8, "mean_abs")[["method", "feature", "mean_abs", "mean_signed", "positive_rate"]]
        .to_string(index=False)
    )
    print("\nTop pairwise interactions:")
    print(
        p2_g[p2_g["budget"] == bmax]
        .nlargest(10, "mean_abs")[["method", "interaction", "mean_abs", "mean_signed", "positive_rate"]]
        .to_string(index=False)
    )
    print("\nTop three-way interactions:")
    print(
        p3_g[p3_g["budget"] == bmax]
        .nlargest(12, "mean_abs")[["method", "interaction", "mean_abs", "mean_signed", "positive_rate"]]
        .to_string(index=False)
    )
    print(f"\nSaved summaries and plots to: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-duels", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[200, 1000, 2000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--sv-methods", nargs="+", default=["RegressionMSR"])
    parser.add_argument("--int-methods", nargs="+", default=["ProxySHAP"])
    parser.add_argument("--o3-methods", nargs="+", default=["ProxySHAP"])
    parser.add_argument(
        "--all-methods", action="store_true",
        help="Run all 4 approximators for SV, order-2, and order-3."
    )
    parser.add_argument(
        "--benchmark-all", action="store_true",
        help=("Convenience flag: equivalent to --all-methods --compute-exact. "
              "Required for the full 4-approximator evaluation plots.")
    )
    parser.add_argument("--compute-exact", action="store_true")
    parser.add_argument("--n-background", type=int, default=200)
    parser.add_argument("--lambda-reg", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="pokemon_multi_duel_budget_seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate summaries/plots from CSV files already in --out-dir."
    )
    args = parser.parse_args()

    run_all_methods = args.all_methods or args.benchmark_all
    compute_exact = args.compute_exact or args.benchmark_all

    sv_methods = SV_ALL if run_all_methods else args.sv_methods
    int_methods = INT_ALL if run_all_methods else args.int_methods
    o3_methods = INT_ALL if run_all_methods else args.o3_methods

    bad_sv = set(sv_methods) - set(SV_ALL)
    bad_i = (set(int_methods) | set(o3_methods)) - set(INT_ALL)
    if bad_sv:
        parser.error(f"unknown SV methods: {sorted(bad_sv)}")
    if bad_i:
        parser.error(f"unknown interaction methods: {sorted(bad_i)}")

    if args.plot_only:
        out = Path(args.out_dir)
        summarize_and_plot(out)
        return

    out = compute(
        n_duels=args.n_duels,
        budgets=args.budgets,
        seeds=args.seeds,
        sv_methods=sv_methods,
        int_methods=int_methods,
        o3_methods=o3_methods,
        sample_seed=args.sample_seed,
        compute_exact=compute_exact,
        n_background=args.n_background,
        lambda_reg=args.lambda_reg,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    summarize_and_plot(out)


if __name__ == "__main__":
    main()
