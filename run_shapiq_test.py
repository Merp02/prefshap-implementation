import torch
import numpy as np

from prefshap_core import *
from prefshap_shapiq_clean import (
    PrefShapItemGame,
    run_shapiq_exact,
    run_shapiq_kernelshap,
    run_shapiq_order2,
    beta_from_shapley_values,
)

torch.manual_seed(0)
np.random.seed(0)


"""
n_ref = 15   # reference datasest
d = 5        # number of item features (small better for exact SV)
X   = torch.randn(n_ref, d, dtype=torch.float32)
X_l = torch.randn(n_ref, d, dtype=torch.float32)
X_r = torch.randn(n_ref, d, dtype=torch.float32)
x_l = torch.randn(1, d, dtype=torch.float32)
x_r = torch.randn(1, d, dtype=torch.float32)

# Placeholder alpha (row vector, shape (1, n_ref)) -- NOT a trained model yet.
alpha = torch.randn(1, n_ref, dtype=torch.float32)
"""

# ----------------------------------------------------------------------
# Testing on toy-dataset
# ----------------------------------------------------------------------

X = torch.from_numpy(np.load("toy_data_5000_10_2/S.npy")).float()
X_l = torch.from_numpy(np.load("toy_data_5000_10_2/l_processed.npy")).float()
X_r = torch.from_numpy(np.load("toy_data_5000_10_2/r_processed.npy")).float()
x_l = X_l[0:1]
x_r = X_r[0:1]
alpha = torch.randn(1,X_l.shape[0])

print("X shape:", X.shape)
print("X_l shape:", X_l.shape)
print("X_r shape:", X_r.shape)
print("x_l shape:", x_l.shape)
print("x_r shape:", x_r.shape)
print("alpha shape:", alpha.shape)

# ----------------------------------------------------------------------
# rbf_kernel
# ----------------------------------------------------------------------
def rbf_kernel(A,B=None,S=None,sigma=1.0):
    '''
    Parameters:
    σ small : only very close points are considered similar
    σ large : even farther points are still considered somewhat similar

    A.shape = (n, d)
    B.shape = (m, d)
    returns (n, m)
    '''
    if B is None:
        B = A

    A_2 = (A ** 2).sum(dim=1,keepdim = True)         # (n, 1)
    B_2 = (B ** 2).sum(dim=1,keepdim = True).T       # (1, m)

    # ∥A−B∥^2 =∥A∥^2 +∥B∥^2 −2(A⋅B T)
    distance = A_2 + B_2 - 2 * A @ B.T             # (n, m)

    return torch.exp(- 0.5 * distance / (sigma ** 2))

d = X.shape[1]
lambda_reg = 1e-2
mask = active_features_item(X_l, X_r, X)
d_eff = int(mask.sum().item())
print(f"d = {d}, d_eff (active features) = {d_eff}")

# ----------------------------------------------------------------------
# 1) existing pipeline (ground truth to compare against)
# ----------------------------------------------------------------------

# computing coalition-matrix (all coalitions considered)
Z, weights = build_item_coalitions_and_weights(
    mask=mask, n_features=X.shape[1], n_samples=2 ** d_eff, device=X.device, big_weight=1e5
)

# for each Z computing coalition values
v_x = compute_pref_values_item_all_S(
    alpha=alpha, X_l=X_l, X_r=X_r, X=X, x_l=x_l, x_r=x_r,
    Z=Z, kernel=rbf_kernel, lambda_reg=lambda_reg,
)

# computing own beta
beta_manual = solve_weighted_regression_clean(v_x, Z, weights, big_weight=1e5)
print("\nmanual beta (my own KernelSHAP regression):")
print(beta_manual.squeeze().numpy())

# ----------------------------------------------------------------------
# 2) shapiq : same value function, shapiq does sampling/weighting/solving
# ----------------------------------------------------------------------

game = PrefShapItemGame(
    alpha=alpha, X_l=X_l, X_r=X_r, X=X, x_l=x_l, x_r=x_r,
    kernel=rbf_kernel, lambda_reg=lambda_reg, mask=mask,
)

# ExactComputer evaluates all coalitions and
# computes the exact shapely values
sv_exact = run_shapiq_exact(game)
beta_exact = beta_from_shapley_values(sv_exact, mask)
print("\nbeta (shapiq ExactComputer, ground truth SV):")
print(beta_exact.numpy())


# KernelShap-Approximator samples and evalutes the coalitions
# defined by the budget
sv_kshap = run_shapiq_kernelshap(game, budget=2 ** d_eff)
beta_kshap = beta_from_shapley_values(sv_kshap, mask)
print("\nbeta (shapiq KernelSHAP, exhaustive budget):")
print(beta_kshap.numpy())

# how good is the approximation of the actual shapley values
# by the manual beta
print("\nmax |manual - shapiq_exact|  =", (beta_manual.squeeze() - beta_exact).abs().max().item())

# cause the budget is exhaustive and should cover all coalitions
# the result should be similar to the ExactComputer
print("max |shapiq_kshap - shapiq_exact| =", (beta_kshap - beta_exact).abs().max().item())

print("\n------------------")
print("Empty Coallition:")
print("------------------")
empty = np.zeros(
    (1, game.n_players),
    dtype=bool,
    )

v_empty = game(empty)[0]
print("Empty: " , empty)
print("V_empty = ", v_empty) # soll 0 sein

from prefshap_core import g_hat

print("\n------------------")
print("FUll Coallition:")
print("------------------")

full = np.ones(
    (1, game.n_players),
    dtype=bool,
)

v_full = game(full)[0]

expected_full = g_hat(
    alpha=alpha,
    X_l=X_l,
    X_r=X_r,
    x_l=x_l,
    x_r=x_r,
    kernel=rbf_kernel,
).item()

print("Full: ", full)
print("V_full                   = ",v_full)
print("\nExpected_full          = ",expected_full)

print("\n------------------")
print("Testing Efficiency:")
print("------------------")
print("Sum(beta)        = ", beta_exact.sum().item())
print("V_full - V_empty = ", v_full - v_empty)
assert(abs(beta_exact.sum().item() - (v_full - v_empty)) < 1e-6 )

print("\n------------------")
print("v_{x_l,x_r}(S) = -v_{x_r,x_l}(S)")
print("Testing skew-symmetry:")
print("------------------")
game_swapped = PrefShapItemGame(
    alpha=alpha,
    X_l=X_l,
    X_r=X_r,
    X=X,
    x_l=x_r, 
    x_r=x_l,
    kernel=rbf_kernel,
    lambda_reg=lambda_reg,
    mask=mask,
)

coalitions = base_10_base_2(
    np.arange(2 ** d_eff),
    d_eff,
).astype(bool)

values_original = game(coalitions)
values_swapped = game_swapped(coalitions)

print("v(x_l, x_r) = ", values_original)
print("\nv(x_r, x_l) = ", values_swapped)
print("\nv(x_l, x_r) + v(x_r, x_l) =  ", values_original + values_swapped)
print("\nmax skew-symmetry error =  ", np.max(np.abs(values_original + values_swapped)))
assert np.allclose(
    values_original,
    -values_swapped,
    atol=1e-5,
    rtol=1e-5,
)
print("\nSkew-symmetry test passed.")

print("\n------------------")
print("Testing for same items:")
print("------------------")

game_same = PrefShapItemGame(
    alpha=alpha,
    X_l=X_l,
    X_r=X_r,
    X=X,
    x_l=x_l,
    x_r=x_l.clone(),
    kernel=rbf_kernel,
    lambda_reg=lambda_reg,
    mask=mask,
)

coalitions = base_10_base_2(
    np.arange(2 ** d_eff),
    d_eff,
).astype(bool)

values_same = game_same(coalitions)

print("v(x_l, x_l) =", values_same)
print("max identical-item error =",
    np.max(np.abs(values_same)),
)
assert np.allclose(
    values_same,
    0.0,
    atol=1e-7,
)

coalitions = base_10_base_2(
    np.arange(2 ** d_eff),
    d_eff,
).astype(bool)

values_same = game_same(coalitions)

print("\nv(x_l, x_l) =", values_same)
print("max identical-item error =", np.max(np.abs(values_same)))
assert np.allclose(
    values_same,
    0.0,
    atol=1e-7,
)
print("Identical-item test passed.")

# ----------------------------------------------------------------------
# Budget Tests
# ----------------------------------------------------------------------

print("\n------------")
print("Budget test:")
print("------------")

all_coalitions = 2 ** d_eff
candidate_budgets = [
    25,
    50,
    100,
    200,
    400,
    700,
    all_coalitions,
]

# Doppelte Werte entfernen und Budget auf maximal 2**d_eff begrenzen.
budgets = sorted({
    min(max(int(budget), d_eff + 2), all_coalitions)
    for budget in candidate_budgets
})
budget_results = []

for budget in budgets:
    sv_budget = run_shapiq_kernelshap(
        game,
        budget=budget,
        random_state=0,
    )

    beta_budget = beta_from_shapley_values(
        sv_budget,
        mask,
    )

    abs_error = (beta_budget - beta_exact).abs()

    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    efficiency_error = abs(
        beta_budget.sum().item()
        - (v_full - v_empty)
    )

    budget_results.append(
        {
            "budget": budget,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "efficiency_error": efficiency_error,
        }
    )

    print(
        f"budget={budget:>3}/{all_coalitions:<3} | "
        f"max error={max_abs_error:.8f} | "
        f"mean error={mean_abs_error:.8f} | "
        f"efficiency error={efficiency_error:.8f}"
    )

#assert budget_results[-1]["max_abs_error"] < 1e-6

print(
    "Exhaustive-budget KernelSHAP matches "
    "the exact Shapley values."
)

# ----------------------------------------------------------------------
# Order-2 Interaction Tests
# ----------------------------------------------------------------------


print("\n-------------------------")
print("Order-2 interaction test:")
print("-------------------------")

order2_values = run_shapiq_order2(
    game,
    budget=2 ** d_eff,
    random_state=0,
)

print("Interaction index:", order2_values.index)
print("Maximum order:", order2_values.max_order)

first_order = {}
pairwise = {}

for interaction, value in order2_values.dict_values.items():
    if len(interaction) == 1:
        first_order[interaction] = value
    elif len(interaction) == 2:
        pairwise[interaction] = value

print("\nFirst-order k-SII values:")
for interaction, value in sorted(first_order.items()):
    print(f"{interaction}: {value:.8f}")


print("\nPairwise k-SII interactions:")
for interaction, value in sorted(pairwise.items()):
    print(f"{interaction}: {value:.8f}")

print("\nlength of first order    = ", len(first_order))
print("length of pairwise order = ", len(pairwise) )
assert len(first_order) == d_eff
assert len(pairwise) == d_eff * (d_eff - 1) // 2

def compare_contributions(first_order, pairwise):
    # Find max positive and negative for First-Order
    fo_pos_key = max(first_order, key=first_order.get)
    fo_pos_val = first_order[fo_pos_key]
    fo_neg_key = min(first_order, key=first_order.get)
    fo_neg_val = first_order[fo_neg_key]

    # Find max positive and negative for Pairwise
    pw_pos_key = max(pairwise, key=pairwise.get)
    pw_pos_val = pairwise[pw_pos_key]
    pw_neg_key = min(pairwise, key=pairwise.get)
    pw_neg_val = pairwise[pw_neg_key]

    print("\nMaximum Positive Value in first order: ", fo_pos_key, fo_pos_val)
    print("Maximum Negative Value in first order: ", fo_neg_key, fo_neg_val)
    print("Maximum Positive Value in pariwise   : ", pw_pos_key, pw_pos_val)
    print("Maximum Negative Value in pairwise   : ", pw_neg_key, pw_neg_val)

    print("\n--------------------")
    print("Positive Comparision: Left (x_l​) ≻ Right (x_r)")
    print("--------------------")
    if pw_pos_val > fo_pos_val:
        f1, f2 = pw_pos_key
        print(f"Features {f1} and {f2} strongly support the preference particularly when considered together.")
    else:
        f1 = fo_pos_key[0]
        print(f"Feature {f1} strongly supports the preference on its own.")

    print("\n--------------------")
    print("Negative Comparision: Right (x_r) ≻ Left (x_l)")
    print("--------------------")
    if abs(pw_neg_val) > abs(fo_neg_val):
        f1, f2 = pw_neg_key
        print(f"Features {f1} and {f2} strongly oppose the preference particularly when considered together.")
    else:
        f1 = fo_neg_key[0]
        print(f"Feature {f1} strongly opposes the preference on its own.")

compare_contributions(first_order=first_order, pairwise=pairwise)

baseline = order2_values.baseline_value

sum_first_order = sum(first_order.values())
sum_pairwise = sum(pairwise.values())

reconstructed_value = (
    baseline
    + sum_first_order
    + sum_pairwise
)

order2_efficiency_error = abs(
    reconstructed_value - v_full
)

print("\n-------------------------------")
print("Order-2 generalized efficiency:")
print("-------------------------------")
print("\nBaseline              =", baseline)
print("Sum first-order       =", sum_first_order)
print("Sum pairwise          =", sum_pairwise)
print("Reconstructed v(full) =", reconstructed_value)
print("Actual v(full)        =", v_full)
print("Efficiency error      =", order2_efficiency_error)

assert order2_efficiency_error < 1e-5

print("\nSum(beta)        = ", beta_exact.sum().item())
if (beta_exact.sum().item()) > 0:
    print ("--> Overall, the model prefers the left item x_l over the right item x_r.\n")
else:
    print ("--> Overall, the model prefers the right item x_r over the left item x_l.\n") 