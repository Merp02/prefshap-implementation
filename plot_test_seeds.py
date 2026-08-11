"""
Convergence plots (mean +/- std over 5 seeds) for the order-1 SV and
order-2 k-SII approximators, on the Charizard vs Squirtle case study.
Saved as robustness_sv.png / robustness_interactions.png.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("test_seeds_results.npz", allow_pickle=True)


sv_rows = (data["sv_rows"])    # [name, budget, seed, max_err, mean_err]
int_rows = (data["int_rows"])  # [name, budget, seed, max_err1, max_err2]
o3_rows = (data["o3_rows"])  # [name, budget, seed, max_err1, max_err2, max_err3]
budgets = data["budgets"]

sv_names = data["sv_names"]
int_names = data["int_names"]
o3_names = data["o3_names"]


def agg(rows, name, metric_col):
    rows_n = rows[rows[:, 0] == name]
    out_mean, out_std = [], []
    for b in budgets:
        vals = rows_n[rows_n[:, 1] == b][:, metric_col].astype(float)
        out_mean.append(vals.mean())
        out_std.append(vals.std())
    return np.array(out_mean), np.array(out_std)

# --------------------------------------------------------------------------- #
# Plot 1: Order-1 SV
# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
for name in sv_names:
    mean, std = agg(sv_rows, name, 3)  # max_err column
    ax.errorbar(budgets, mean, yerr=std, marker="o", capsize=3, label=str(name))

ax.set_yscale("log")
ax.set_xlabel("budget (of 8192 possible coalitions)")
ax.set_ylabel("max |approx SV - exact SV|")
ax.set_title("Charizard vs Squirtle: order-1 SV approximators\n(mean ± std over 5 seeds)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/test_seeds_sv.png", dpi=150)
print("saved test_seeds_sv.png")

# --------------------------------------------------------------------------- #
# Plot 2: Order-2 k-SII
# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
for name in int_names:
    mean, std = agg(int_rows , name, 4)  # max_err(order2) column
    ax.errorbar(budgets, mean, yerr=std, marker="o", capsize=3, label=str(name))
ax.set_xlabel("budget (of 8192 possible coalitions)")
ax.set_ylabel("max |approx k-SII(order2) - exact k-SII(order2)|")
ax.set_title("Charizard vs Squirtle: order-2 interaction approximators\n(mean ± std over 5 seeds)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/test_seeds_interactions.png", dpi=150)
print("saved test_seeds_interactions.png")

# ---------------------------------------------------------------------------
# Plot 3: Order-3 k-SII
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 5.0))
for name in o3_names:
    mean, std = agg(o3_rows, name, 5) # max_err_order3
    ax.errorbar(budgets, mean, yerr=std, marker="o", capsize=3, linewidth=2, label=str(name))
ax.set_yscale("log")

ax.set_xlabel("budget (of 8192 possible coalitions)")
ax.set_ylabel( "max |approx k-SII(order 3) - exact k-SII(order 3)|")
ax.set_title("Charizard vs Squirtle: order-3 interaction approximators\n""(mean ± std over seeds)")
ax.legend()
ax.grid(alpha=0.3,which="both")
fig.tight_layout()
fig.savefig("plots/test_seeds_order3_log.png", dpi=150)
plt.close(fig)
print("saved test_seeds_order3_log.png")