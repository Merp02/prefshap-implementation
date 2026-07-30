"""
Bridge between the item-level PREF-SHAP value function v(S) and shapiq.

Idea: 
`compute_pref_value_item_single_S` already computes v(S) for one coalition S,
exactly as Proposition 3.2 . That function is the actual "game". 

(sample_Z, kernelshap_weights_from_Z, expand_Z_to_original_features,
solve_weighted_regression_clean, ...) is generic KernelSHAP machinery that
shapiq already implements and has tested. 

    order-1 (Shapley values)        order-2 (k-SII interactions)
    ------------------------        ----------------------------
    KernelSHAP                      KernelSHAPIQ
    RegressionMSR (index="SV")      ProxySHAP        (index="k-SII")
    PermutationSamplingSV           PermutationSamplingSII
    SHAPIQ (max_order=1, "SV")      SHAPIQ (max_order=2, "k-SII")
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
        self, alpha, X_l, X_r, X, x_l, x_r, kernel, lambda_reg, mask,
        y_pred_mean=None, verbose: bool = False,
    ):
        self.alpha = alpha
        self.X_l, self.X_r, self.X = X_l, X_r, X
        # learned from `fit_g_pokemon.py``
        # alpha: (1, M) with M: length of Nyström centers
        # X_l:   (M, 13)
        # X_r:   (M, 13)

        self.x_l, self.x_r = x_l, x_r
        # X_l:   (1, 13)
        # X_r:   (1, 13)
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

def make_exact_computer(game: PrefShapItemGame) -> "shapiq.ExactComputer":
    """
    One ExactComputer per game. Build it once and reuse it across index=
    calls (SV, k-SII, ...) -- the underlying 2**d_eff coalition value
    function evaluations are cached internally and not recomputed.
    """
    return shapiq.ExactComputer(n_players=game.n_players, game=game)

def run_shapiq_exact(game_or_computer, index: str = "SV", order: int | None = None):
    """
    Exact reference values. Evaluates all 2**d_eff coalitions (once).
    
    If the supplied object is already an ExactComputer: use it directly
    otherwise: assume it is a game and create an ExactComputer
    """
    if isinstance(game_or_computer,shapiq.ExactComputer,):
        computer = game_or_computer
    else:
        computer = make_exact_computer(
        game_or_computer
    )
        
    return computer(index=index, order = order)

# --------------------------------------------------------------------------- #
# order-1 (Shapley value) approximators
# --------------------------------------------------------------------------- #

_SV_APPROXIMATORS = {
    "KernelSHAP": lambda n, seed: shapiq.KernelSHAP(n=n, random_state=seed),
    "RegressionMSR": lambda n, seed: shapiq.RegressionMSR(n=n, index="SV", random_state=seed),
    "PermutationSamplingSV": lambda n, seed: shapiq.PermutationSamplingSV(n=n, random_state=seed),
    "SHAPIQ": lambda n, seed: shapiq.SHAPIQ(n=n, max_order=1, index="SV", random_state=seed),
}

def run_sv_approximator(name: str, game: PrefShapItemGame, budget: int, random_state: int = 0):
    approximator = _SV_APPROXIMATORS[name](game.n_players, random_state)
    return approximator.approximate(budget=budget, game=game)

# check


# --------------------------------------------------------------------------- #
# order-2 (k-SII interaction) approximators
# --------------------------------------------------------------------------- #

_INTERACTION_APPROXIMATORS = {
    "KernelSHAPIQ": lambda n, seed: shapiq.KernelSHAPIQ(n=n, max_order=2, index="k-SII", random_state=seed),
    "ProxySHAP": lambda n, seed: shapiq.ProxySHAP(n=n, max_order=2, index="k-SII", random_state=seed),
    "PermutationSamplingSII": lambda n, seed: shapiq.PermutationSamplingSII(n=n, max_order=2, index="k-SII", random_state=seed),
    "SHAPIQ": lambda n, seed: shapiq.SHAPIQ(n=n, max_order=2, index="k-SII", random_state=seed),
}

def run_interaction_approximator(name:str, game: PrefShapItemGame,budget: int,
                      random_state: int = 0,):

    approximator = _INTERACTION_APPROXIMATORS[name](game.n_players, random_state)
    return approximator.approximate(budget=budget, game=game)



"""
-----------------------
Order-1-Approximatoren:
-----------------------

KernelSHAP: Approximiert Shapley Values über eine gewichtete Regression auf Koalitionswerten.

RegressionMSR: Verwendet eine Regression auf marginalen Substitutionen zur Schätzung der SV.

PermutationSamplingSV: Sampelt Permutationen der Spieler und mittelt marginale Beiträge.

SHAPIQ mit max_order=1: Die allgemeine SHAP-IQ-Methode wird auf reine Shapley Values beschränkt.

-----------------------
Order-2-Approximatoren:
-----------------------

KernelSHAPIQ: Regression-basierte Approximation von Interaktionen.

ProxySHAP: Approximation über eine Proxy-Darstellung.

PermutationSamplingSII: Schätzt Interaktionen über Permutationssampling.

SHAPIQ: Allgemeiner SHAP-IQ-Samplingalgorithmus für Interaktionen.

"""