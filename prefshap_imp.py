# item-only PREF-SHAP implementation
import torch
import numpy as np

def active_features_item(X_l, X_r, X=None):
    """
    Entscheidet, welche Features überhaupt variieren

    Original in pref_shap.fit:
        self.mask = ((x.var(0) + x_prime.var(0)) > 0).cpu()
        self.eff_dim = self.mask.sum().item()

    """
    variances = X_l.var(0) + X_r.var(0)
    if X is not None:
        variances = variances + X.var(0)
    return (variances > 0)


def kernelshap_weights_from_Z(Z_eff, big_weight=1e5):
    """
    Berechnet die KernelSHAP-Gewichte für die Coalitions

    Original  pref_shap.setup_the_rest:
        abs_S = Z.sum(1)
        a = exp(lgamma(d+1) - lgamma(d-|S|+1) - lgamma(|S|+1))
        weights = (d - 1) / (a * |S| * (d - |S|))

    """
    # d_eff: number of active features
    d_eff = Z_eff.shape[1]
    # how many features are active in each coalition
    abs_S = Z_eff.sum(1)

    # empty vector for the weights
    weights = torch.empty(Z_eff.shape[0])
    # empty or full coalitions
    middle = (abs_S > 0) & (abs_S < d_eff)

    #lgamma(d + 1) = log(d!)
    const = torch.lgamma(torch.tensor(float(d_eff)) + 1)
    # a = C(d_eff, |S|)
    a = torch.exp(
        const
        - torch.lgamma((d_eff - abs_S[middle]) + 1)
        - torch.lgamma(abs_S[middle] + 1)
    )

    weights[middle] = (d_eff - 1) / (a * abs_S[middle] * (d_eff - abs_S[middle]))

    # Normalization
    if middle.any():
        weights[middle] = weights[middle] / weights[middle].sum()

    # empty or full coalitions recieve big weight
    weights[~middle] = big_weight
    return weights.unsqueeze(-1)


def expand_Z_to_original_features(Z_eff, mask, n_features):
    """
    Z_eff: nur im aktiven Feature-Raum d_eff
    Am Ende braucht man aber PREF-SHAP-Werte für alle ursprünglichen d Features
    Deswegen map back von d-eff zu d

    Origial  pref_shap.setup_the_rest(...):
        self.Z = torch.zeros(Z.shape[0], self.m).to(self.device)
        self.Z[:, self.mask] = Z
    """
    Z = torch.zeros(Z_eff.shape[0], n_features, device=Z_eff.device, dtype=Z_eff.dtype)
    Z[:, mask] = Z_eff
    return Z

def base_10_base_2(indices: np.array,d:int=10):
    S= np.zeros((indices.shape[0],d))
    rest = indices
    valid_rows = rest>0
    while True:
        set_to_1 =  np.floor(np.log2(rest)).astype(int)
        set_to_1_prime=set_to_1[valid_rows][:,np.newaxis]
        p = S[valid_rows,:]
        np.put_along_axis(p, set_to_1_prime, 1, axis=1)
        S[valid_rows, :] = p
        # S[valid_rows,:][:,set_to_1_prime]=1
        rest = rest-2**(np.clip(set_to_1,0,np.inf))
        valid_rows = rest>0
        if valid_rows.sum()==0:
            return S

def base_2_base_10(N:int=2500,d:int=10):
    P = np.random.binomial(size=(N,d), n=1, p=0.5)
    b = 2 ** np.arange(0, d)
    unique_ref = (b*P).sum(1)
    _,idx=np.unique(unique_ref,return_index=True)
    return P[idx,:]

def sample_Z(D,max_S):
    max_range = min(2**D,2**63-1)
    if max_S>=max_range:
        configs= np.arange(max_range)
        return base_10_base_2(configs, D)
    else:
        return base_2_base_10(max_S, D)

def build_item_coalitions_and_weights(mask, n_features, n_samples, device, big_weight=1e5):
    """
    Original Idea:
    coalition and weight construction in setup_the_rest()
    """

    #d_eff berechnen
    d_eff = int(mask.sum().item())
    #Baut die Coalition-Matrix Z_eff für die aktiven Features
    Z_eff = torch.from_numpy(sample_Z(d_eff, n_samples)).float().to(device)

    # empty und full coalitions bilden
    empty = torch.zeros(1, d_eff, device=device)
    full = torch.ones(1, d_eff, device=device)

    if Z_eff.shape[0] > 0:
        middle = Z_eff[(Z_eff.sum(1) > 0) & (Z_eff.sum(1) < d_eff)]
        Z_eff = torch.cat([empty, middle, full], dim=0)
    else:
        Z_eff = torch.cat([empty, full], dim=0)

    # KernalSHAP weights berechnen
    weights = kernelshap_weights_from_Z(Z_eff, big_weight=big_weight)
    Z = expand_Z_to_original_features(Z_eff, mask, n_features)
    return Z, weights


def compute_pref_value_item_single_S(
        alpha,
        X_l,
        X_r,
        X,
        x_l,
        x_r,
        S,
        kernel,
        lambda_reg,
        y_pred_mean = None #Basiswert; ein Durchschnittwert; deint zur Messung der Abweichung der Coallition-values von dem Durchschnittswert
):
    """
    Berechnet item-only PREF-SHAP value function für EINE coalition S: v(S)
    Implementation: Proposition 3.2 von original kernel_tensor_batch()
    """

    S = S.bool()
    S_C = ~S

    device = X.device
    dtype = X.dtype
    n_ref = X.shape[0]

    # Regularization in the original code: n_ref · λ · I
    # self.reg = self.N_x * self.lamb
    # self.eye = torch.eye(self.N_x) * self.reg
    reg_eye = torch.eye(n_ref, device=device, dtype=dtype) * (n_ref * lambda_reg)

    # For empty S:
    if S.sum() == 0:
        return torch.zeros(1, device=device, dtype=dtype)

    # For empty S_C:
    if S_C.sum() == 0:
        return torch.zeros(1, device=device, dtype=dtype)

    # 1. Kernel matrix K_{X_S, X_S}
    # Original:
    #   inv_tens.append(self.k(self.X[:, S], None, S))
    K_XS_XS = kernel(X[:, S], None, S)


    # 2. Kernel vectors K_{X_S, x_l_S} and K_{X_S, x_r_S}
    # Original:
    #   x_S, x_prime_S = x[:, S], x_prime[:, S]
    #   xs_cat = torch.cat([x_S, x_prime_S], dim=0)
    #    vec_cat = self.k(self.X[:, S], xs_cat, S)
    K_XS_xlS = kernel(X[:, S], x_l[:, S], S)
    K_XS_xrS = kernel(X[:, S], x_r[:, S], S)

    # 3. conditional mean embedding
    # Original:
    #   cg_output = self.tensor_CG.solve(inv_tens, vec)

    # Clean version: solve directly instead of using tensor_CG.solve
    cme_l = torch.linalg.solve(K_XS_XS + reg_eye, K_XS_xlS)
    cme_r = torch.linalg.solve(K_XS_XS + reg_eye, K_XS_xrS)

    # 4. Kernel terms for S
    # Original:
    #    stacked_lr = torch.cat([self.X_l[:, S], self.X_r[:, S]], dim=0)
    #    l_hadamard_mat = self.k(stacked_lr, xs_cat, S)
    # then sliced into four parts
    K_XlS_xlS = kernel(X_l[:, S], x_l[:, S], S)
    K_XlS_xrS = kernel(X_l[:, S], x_r[:, S], S)
    K_XrS_xlS = kernel(X_r[:, S], x_l[:, S], S)
    K_XrS_xrS = kernel(X_r[:, S], x_r[:, S], S)

    # 5. Kernel terms for S_C
    # Original:
    #   klsc_xsc = self.k(self.X_l[:, S_C], self.X[:, S_C], S_C)
    #   krsc_xsc = self.k(self.X_r[:, S_C], self.X[:, S_C], S_C)
    K_XlSc_XSc = kernel(X_l[:, S_C], X[:, S_C], S_C)
    K_XrSc_XSc = kernel(X_r[:, S_C], X[:, S_C], S_C)


    # 6. Missing-feature integration terms
    # Original:
    #   cg_output_a = torch.bmm(cat_klsc_xsc, cg_output)
    #   cg_output_b = torch.bmm(cat_krsc_xsc, cg_output)
    M_l_l = K_XlSc_XSc @ cme_l
    M_l_r = K_XlSc_XSc @ cme_r

    M_r_l = K_XrSc_XSc @ cme_l
    M_r_r = K_XrSc_XSc @ cme_r

    # 7. Gamma terms from Proposition 3.2
    # Original mapping:
    #   Gamma(X_l, x_l) = cat_xls_xs       * cg_l_a
    #   Gamma(X_r, x_r) = cat_xrs_xs_prime * cg_r_a
    #   Gamma(X_l, x_r) = cat_xls_xs_prime * cg_l_b
    #   Gamma(X_r, x_l) = cat_xrs_xs       * cg_r_b
    Gamma_l_l = K_XlS_xlS * M_l_l
    Gamma_r_r = K_XrS_xrS * M_l_r

    Gamma_l_r = K_XlS_xrS * M_r_l
    Gamma_r_l = K_XrS_xlS * M_r_r

    # 8. Preference structure
    # Proposition 3.2:
    # original direction - reversed direction
    pref_kernel_value = Gamma_l_l * Gamma_r_r - Gamma_l_r * Gamma_r_l

    # 9. Multiply with alpha
    # Original from value_observation:
    #   output = (self.alpha @ output).squeeze() - self.y_pred_mean
    value_S = (alpha @ pref_kernel_value).squeeze()

    if y_pred_mean is not None:
        value_S = value_S - y_pred_mean

    return value_S

def compute_pref_values_item_all_S(
        alpha,
        X_l,
        X_r,
        X,
        x_l,
        x_r,
        Z,
        kernel,
        lambda_reg,
        y_pred_mean = None
):
    """
    Berechnet v(S) für alle coalitions S.
    Ersetzt original kernel_tensor_batch und value_observation path.
    """

    values = []

    for i in range(Z.shape[0]):
        S = Z[i, :]

        value_S = compute_pref_value_item_single_S(
            alpha=alpha,
            X_l=X_l,
            X_r=X_r,
            X=X,
            x_l=x_l,
            x_r=x_r,
            S=S,
            kernel=kernel,
            lambda_reg=lambda_reg,
            y_pred_mean=y_pred_mean,
        )

        values.append(value_S)

    return torch.stack(values, dim=0)


def solve_weighted_regression_clean(Y_cat, Z, weights, big_weight=1e5):
    """
    Berechnet die PREF-SHAP Werte beta aus den Coalition Values
        beta = (Z^T W Z)^(-1) Z^T W v

    Original in pref_shap.py:
        construct_values(...)
        OLS_solve(...)


    Y_cat   = v (die Coalition Values)
    Z       = Coalition Matrix
    weights = KernelSHAP Gewichte W
    beta    = finale PREF-SHAP Werte
    """

    # Falls Y_cat nur ein Vektor ist, daraus eine Spalte machen
    if Y_cat.ndim == 1:
        Y_cat = Y_cat.unsqueeze(-1)

    # clonen, damit die originale Weights bleiben unverändert
    weights = weights.clone()

    # empty und full coalition get big weight
    weights[0] = big_weight
    weights[-1] = big_weight

    # Nur Features verwenden, die in mindestens einer Coalition vorkommen
    # Original OLS_solve:
    #   mask = Z_in.sum(0) > 0
    mask = Z.sum(0) > 0
    Z_masked = Z[:, mask]

    # Original construct_values:
    #   Y_target = weights * Y_cat
    # Entspricht Wv.
    Y_target = weights * Y_cat

    # Original OLS_solve:
    #   A = Z.t() @ (Z * weights)
    #   b = Z.t() @ Y_target
    # Entspricht:
    #   A = Z^T W Z
    #   b = Z^T W v
    A = Z_masked.t() @ (Z_masked * weights)
    b = Z_masked.t() @ Y_target

    # Statt A zu invertieren -> A beta = b lösen
    # Original:
    #   L = torch.linalg.cholesky(A)
    #   sol = torch.cholesky_solve(b, L)
    L = torch.linalg.cholesky(A)
    sol = torch.cholesky_solve(b, L)

    # beta hat komplette d, und nicht nur d_eff
    beta = torch.zeros(Z.shape[1], Y_cat.shape[1], device=Z.device, dtype=Y_cat.dtype)
    beta[mask, :] = sol

    return beta


def pref_shap_item_clean(
        alpha,
        X_l,
        X_r,
        X,
        x_l,
        x_r,
        kernel,
        n_samples,
        lambda_reg,
        y_pred_mean=None,
        big_weight=1e5,
):
    """
    Vollständige  item-only PREF-SHAP Implementierung ohne Kontext.

    Entspricht Algorithmus 1:
        1. aktive Features bestimmen
        2. Coalitions Z und Gewichte W bauen
        3. v(S) für alle Coalitions berechnen
        4. beta über weighted regression berechnen
        5. beta zurückgeben
    """

    device = X.device

    alpha = alpha.t().to(device)
    X_l = X_l.to(device)
    X_r = X_r.to(device)
    X = X.to(device)
    x_l = x_l.to(device)
    x_r = x_r.to(device)

    # 1: d_eff über active Features
    mask = active_features_item(X_l, X_r, X).to(device)

    # 2: Coalitions Z und KernelSHAP Gewichte W.
    Z, weights = build_item_coalitions_and_weights(
        mask=mask,
        n_features=X.shape[1],
        n_samples=n_samples,
        device=device,
        big_weight=big_weight,
    )

    # 3-8: Coalition Values v(S) berechnen
    Y_cat = compute_pref_values_item_all_S(
        alpha=alpha,
        X_l=X_l,
        X_r=X_r,
        X=X,
        x_l=x_l,
        x_r=x_r,
        Z=Z,
        kernel=kernel,
        lambda_reg=lambda_reg,
        y_pred_mean=y_pred_mean,
    )

    # 9: beta = (Z^T W Z)^(-1) Z^T W v.
    beta = solve_weighted_regression_clean(
        Y_cat=Y_cat,
        Z=Z,
        weights=weights,
        big_weight=big_weight,
    )

    # 10: return beta.
    return beta, Y_cat, weights, Z

