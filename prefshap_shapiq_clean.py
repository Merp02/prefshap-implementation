"""
Bridge between the item-level PREF-SHAP value function v(S) and shapiq.

Idea: 
`compute_pref_value_item_single_S` already computes v(S) for one coalition S,
exactly as Proposition 3.2 . That function is the actual "game". 

(sample_Z, kernelshap_weights_from_Z, expand_Z_to_original_features,
solve_weighted_regression_clean, ...) is generic KernelSHAP machinery that
shapiq already implements and has tested. 

    v(S)  ->  shapiq.Game  ->  shapiq.KernelSHAP / shapiq.ExactComputer  ->  beta

This keeps the value function (the actual PREF-SHAP-specific math) but swaps
the custom "beta = (Z^T W Z)^-1 Z^T W v" solver for shapiq's.
"""

import numpy as np
import torch
import shapiq

from prefshap_core import compute_pref_value_item_single_S


class PrefShapItemGame(shapiq.Game):
    """
    A shapiq Game wrapping the item-level PREF-SHAP value function v(S) for a
    single duel (x_l, x_r).

    Important: shapiq only ever sees the ACTIVE features (d_eff = mask.sum()),
    not the full original feature dimension d. Internally, every coalition
    that shapiq generates over the d_eff active features is expanded back to
    the original d-dimensional S (inactive features are always absent from S,
    exactly like `expand_Z_to_original_features` does), and then
    `compute_pref_value_item_single_S` is called on that expanded S.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        X_l: torch.Tensor,
        X_r: torch.Tensor,
        X: torch.Tensor,
        x_l: torch.Tensor,
        x_r: torch.Tensor,
        kernel,
        lambda_reg: float,
        mask: torch.Tensor,
        y_pred_mean: torch.Tensor | None = None,
        verbose: bool = False,
    ):
        self.alpha = alpha
        self.X_l, self.X_r, self.X = X_l, X_r, X
        self.x_l, self.x_r = x_l, x_r
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.y_pred_mean = y_pred_mean

        self.mask = mask.bool()
        self.n_features = mask.shape[0]                  # d (original dim)
        self._active_idx = torch.where(self.mask)[0]     # original indices of active features
        d_eff = int(self.mask.sum().item())

        # shapiq centeres v(empty) = 0 but game is already centered in compute_pref_value_item_single_S
        # so normalize = False
        super().__init__(n_players=d_eff, normalize=False, verbose=verbose)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        coalitions: bool array, shape (n_coalitions, d_eff) -- shapiq's coalitions
                    over the ACTIVE features only.
        returns:    float array, shape (n_coalitions,)
        """
        n_coalitions = coalitions.shape[0]
        values = np.zeros(n_coalitions, dtype=float)

        # v(S) for each coalition
        for i in range(n_coalitions):
            S_full = torch.zeros(self.n_features, dtype=torch.bool, device=self.mask.device)
            S_active = torch.as_tensor(coalitions[i], dtype=torch.bool, device=self.mask.device)
            S_full[self._active_idx] = S_active

            v = compute_pref_value_item_single_S(
                alpha=self.alpha,
                X_l=self.X_l, X_r=self.X_r, X=self.X,
                x_l=self.x_l, x_r=self.x_r,
                S=S_full,
                kernel=self.kernel,
                lambda_reg=self.lambda_reg,
                y_pred_mean=self.y_pred_mean,
            )
            values[i] = v.item()

        return values


def beta_from_shapley_values(sv: "shapiq.InteractionValues", mask: torch.Tensor) -> torch.Tensor:
    """
    Map shapiq's first-order Shapley values (defined over the d_eff ACTIVE
    features) back to a beta vector over the original d features

    Equivalent to: `expand_Z_to_original_features` for the coalition matrix
    """
    d_eff_values = sv.to_first_order_array()       # shape (d_eff,)
    beta = torch.zeros(mask.shape[0], dtype=torch.float64)
    beta[mask.bool()] = torch.as_tensor(d_eff_values, dtype=torch.float64) # filling only for active features, inactive = 0
    return beta


def run_shapiq_exact(game: PrefShapItemGame) -> "shapiq.InteractionValues":
    """
    Exact Shapley values: evaluates the game on all 2**d_eff coalitions
    Only feasible for small d_eff
    
    Gives an exact ground truth for validating my implementation/the shapiq part
    """
    computer = shapiq.ExactComputer(n_players=game.n_players, game=game)
    return computer(index="SV")


def run_shapiq_kernelshap(game: PrefShapItemGame, budget: int | None = None,
                           random_state: int | None = 0) -> "shapiq.InteractionValues":
    """
    Approximate Shapley values via shapiq's own KernelSHAP regression solver
    (replaces sample_Z + kernelshap_weights_from_Z + solve_weighted_regression_clean).

    budget: number of coalitions to sample & evaluate. 
    If None, uses 2**d_eff,
    i.e. exhaustive sampling (exact-equivalent, 
    useful to cross check against run_shapiq_exact)
    """
    d_eff = game.n_players
    if budget is None:
        budget = 2 ** d_eff

    approximator = shapiq.KernelSHAP(n=d_eff, random_state=random_state)
    return approximator.approximate(budget=budget, game=game)

def run_shapiq_order2(game: PrefShapItemGame,budget: int | None = None,
                      random_state: int | None = 0,) -> "shapiq.InteractionValues":
    """
    Approximate first-order and pairwise interactions with KernelSHAP-IQ.
    Uses k-SII with maximum interaction order 2.
    Returned keys:
        (i,)      first-order contribution
        (i, j)    pairwise interaction
    """
    d_eff = game.n_players
    if budget is None:
        budget = 2 ** d_eff

    approximator = shapiq.KernelSHAPIQ(n=d_eff,max_order=2,
                                       index="k-SII",
                                       random_state=random_state)

    return approximator.approximate(budget=min(int(budget), 2 ** d_eff),
                                    game=game)
