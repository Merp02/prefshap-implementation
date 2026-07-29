# PREF-SHAP with shapiq Bridge

This repository contains the item-level PREF-SHAP implementation and its bridge to the `shapiq` library.

The current scope ends at the **shapiq bridge and validation stage**. 

Not indcluded yet:

- preference-model training,
- `g_learning`,
- Pokémon data,
- Nyström training,
- learned `alpha` integration.

A random placeholder `alpha` is currently used to validate the PREF-SHAP game and the shapiq integration.

---

## 1. Goal

The implementation separates PREF-SHAP-specific mathematics from generic Shapley computation.

```text
Coalition S
    ↓
compute_pref_value_item_single_S
    ↓
PREF-SHAP game value v(S)
    ↓
PrefShapItemGame(shapiq.Game)
    ↓
ExactComputer / KernelSHAP / KernelSHAPIQ
    ↓
Shapley values or order-2 interactions
```

The existing value function remains responsible for the PREF-SHAP mathematics. shapiq replaces generic coalition sampling, weighting, regression, exact computation, and interaction handling.

---

## 2. Setup

```bash
cd /Users/pradarshandahal/Desktop/Arbeit/prefshap_imp
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python run_test.py
python run_shapiq_test.py
```

Main shapiq documentation:

```text
https://shapiq.readthedocs.io/en/latest/index.html
```

---

## 3. Main Files

```text
prefshap_imp/
├── prefshap_math/
    ├── prefshap_core.py
    ├── run_test.py
├── prefshap_shapiq_bridge/
    ├── prefshap_shapiq_clean.py
    ├── run_shapiq_test.py
├── prefshap_g_learning/
    ├── pref_fit.py
    ├── run_g_learn_test.py
├── requirements.txt
├── RUN_INST.txt
├── datasets/
    └── toy_data_5000_10_2/
        ├── S.npy
        ├── l_processed.npy
    └── alan_data_5000_1000/
        ├── S.npy
        ├── l_processed.npy
    └── pokemon/
        ├── combats.xls
        ├── pokemon.xls
├── artifacts

```

---

## 4. Current Data

Observed shapes:

```text
X shape:      torch.Size([200, 10])
X_l shape:    torch.Size([5000, 10])
X_r shape:    torch.Size([5000, 10])
x_l shape:    torch.Size([1, 10])
x_r shape:    torch.Size([1, 10])
alpha shape:  torch.Size([1, 5000])
```

Meaning:

- `X`: background/reference items used by the conditional mean embedding.
- `X_l`, `X_r`: left and right items from 5000 preference pairs.
- `x_l`, `x_r`: one duel to explain.
- `alpha`: placeholder coefficient vector.

Current placeholder:

```python
alpha = torch.randn(1, X_l.shape[0])
```

The current multiplication expects:

```text
alpha:             (1, n_pairs)
preference kernel: (n_pairs, 1)
result:            (1, 1)
```

---

## 5. Core PREF-SHAP Game

The central function is:

```python
compute_pref_value_item_single_S(...)
```

It maps one coalition `S` to one scalar game value `v(S)`.

Inputs:

```python
alpha
X_l
X_r
X
x_l
x_r
S
kernel
lambda_reg
```

### Empty coalition

```python
if S.sum() == 0:
    return 0
```

Expected:

```math
v(∅) = 0
```

### Full coalition

```python
if (~S).sum() == 0:
    return g_hat(...)
```

Expected:

```math
v(N) = g_hat(x_l, x_r)
```

### Partial coalition

For partial `S`, the implementation:

1. selects the observed features `S`,
2. constructs `K_(X_S, X_S)`,
3. solves the conditional mean embedding system,
4. integrates missing features `S_C`,
5. evaluates the positive preference direction,
6. evaluates the reverse preference direction,
7. subtracts both directions,
8. combines the result with `alpha`.

Conceptually:

```math
v(S) =
alpha^T [
Gamma(X_l,x_l) Gamma(X_r,x_r)
-
Gamma(X_l,x_r) Gamma(X_r,x_l)
]
```

---

## 6. Preference Kernel

The generalized preferential kernel is:

```math
k_E((X_l,X_r),(x_l,x_r))
=
k(X_l,x_l)k(X_r,x_r)
-
k(X_l,x_r)k(X_r,x_l)
```

The preference-model output is:

```math
g_hat(x_l,x_r)
=
alpha^T k_E((X_l,X_r),(x_l,x_r))
```

Example implementation:

```python
def g_hat(alpha, X_l, X_r, x_l, x_r, kernel):
    pref_kernel = preference_kernel_value(
        X_l=X_l,
        X_r=X_r,
        x_l=x_l,
        x_r=x_r,
        kernel=kernel,
    )
    return (alpha @ pref_kernel).reshape(1)
```

---

## 7. Active Features

Current mask:

```python
mask = active_features_item(X_l, X_r, X)
```

Effective dimension:

```python
d = X.shape[1]
d_eff = int(mask.sum().item())
```

Current dataset:

```text
d = 10
d_eff = 10
```

Do not leave `d = 5` hard-coded.

---

## 8. shapiq Bridge

The bridge is:

```python
class PrefShapItemGame(shapiq.Game):
```

shapiq uses coalitions in active-feature space:

```text
0, ..., d_eff - 1
```

The PREF-SHAP value function expects coalitions in the original feature space:

```text
0, ..., d - 1
```

The bridge expands each coalition:

```python
def value_function(self, coalitions: np.ndarray) -> np.ndarray:
    values = np.zeros(coalitions.shape[0], dtype=float)

    with torch.no_grad():
        for i in range(coalitions.shape[0]):
            S_full = torch.zeros(
                self.n_features,
                dtype=torch.bool,
                device=self.mask.device,
            )

            S_active = torch.as_tensor(
                coalitions[i],
                dtype=torch.bool,
                device=self.mask.device,
            )

            S_full[self._active_idx] = S_active

            value = compute_pref_value_item_single_S(
                alpha=self.alpha,
                X_l=self.X_l,
                X_r=self.X_r,
                X=self.X,
                x_l=self.x_l,
                x_r=self.x_r,
                S=S_full,
                kernel=self.kernel,
                lambda_reg=self.lambda_reg,
                y_pred_mean=self.y_pred_mean,
            )

            values[i] = value.item()

    return values
```

The bridge does not change the PREF-SHAP mathematics. It only adapts the coalition representation and output type expected by shapiq.

---

## 9. Exact Shapley Values

```python
def run_shapiq_exact(game: PrefShapItemGame):
    computer = shapiq.ExactComputer(
        n_players=game.n_players,
        game=game,
    )
    return computer(index="SV")
```

`ExactComputer` evaluates all:

```math
2^d_eff
```

coalitions.

Examples:

```text
d_eff = 5  -> 32 coalitions
d_eff = 10 -> 1024 coalitions
```

This result is the validation reference.

---

## 10. KernelSHAP

```python
def run_shapiq_kernelshap(
    game: PrefShapItemGame,
    budget: int | None = None,
    random_state: int | None = 0,
):
    d_eff = game.n_players
    max_budget = 2 ** d_eff

    if budget is None:
        budget = max_budget

    approximator = shapiq.KernelSHAP(
        n=d_eff,
        random_state=random_state,
    )

    return approximator.approximate(
        budget=min(int(budget), max_budget),
        game=game,
    )
```

shapiq now handles:

- coalition selection,
- KernelSHAP weights,
- regression constraints,
- weighted regression,
- result storage.

The old manual pipeline remains only as a validation baseline.

---

## 11. Mapping Back to Original Features

```python
def beta_from_shapley_values(sv, mask):
    active_values = sv.to_first_order_array()

    beta = torch.zeros(
        mask.shape[0],
        dtype=torch.float64,
    )

    beta[mask.bool()] = torch.as_tensor(
        active_values,
        dtype=torch.float64,
    )

    return beta
```

Inactive features receive attribution zero.

---

## 12. Order-2 Interactions

```python
def run_shapiq_order2(
    game: PrefShapItemGame,
    budget: int | None = None,
    random_state: int | None = 0,
):
    d_eff = game.n_players
    max_budget = 2 ** d_eff

    if budget is None:
        budget = max_budget

    approximator = shapiq.KernelSHAPIQ(
        n=d_eff,
        max_order=2,
        index="k-SII",
        random_state=random_state,
    )

    return approximator.approximate(
        budget=min(int(budget), max_budget),
        game=game,
    )
```

Returned interaction keys:

```text
(i,)      first-order component
(i, j)    pairwise interaction
```

For 10 active features:

```text
10 first-order values
45 pairwise interactions
```

---

## 13. Current Conclusion

The item-level PREF-SHAP value function has been successfully integrated with shapiq.

The implementation now supports:

- exact ordinary Shapley values,
- budgeted KernelSHAP approximation,
- order-2 k-SII interactions,
- active-feature mapping,
- empty/full boundary checks,
- skew-symmetry,
- identical-item validation,
- ordinary and generalized efficiency checks.

The bridge is functionally validated and ready for final cleanup and direct comparison with the original batched PREF-SHAP implementation.
