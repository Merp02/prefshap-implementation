# item-only PREF-SHAP implementation
import torch
import numpy as np

def active_features_item(X_l, X_r, X=None):
    """
    Determine which item features vary across the compared left (X_l) /righ (X_r) items.
    X: Optional reference data, whose variacne can also be considired

    Originally in pref_shap.fit:
        self.mask = ((x.var(0) + x_prime.var(0)) > 0).cpu()
        self.eff_dim = self.mask.sum().item()


    Returns a boolean mask indicating active features (non-zero varriance).
    """
    variances = X_l.var(0) + X_r.var(0)
    if X is not None:
        variances = variances + X.var(0)
    return (variances > 0)


def kernelshap_weights_from_Z(Z_eff, big_weight=1e5):
    """
    Compute KernelSHAP coalition weights.

    Originally in pref_shap.setup_the_rest:
        abs_S = Z.sum(1)
        a = exp(lgamma(d+1) - lgamma(d-|S|+1) - lgamma(|S|+1))
        weights = (d - 1) / (a * |S| * (d - |S|))
    
    In cases of empty and full coalitions, they recieve a large fixed weight (big_weight).
    Z_eff: coalition matrix in the active feature space

    Returns a column vector of coalition weights.

    """
    # d_eff: number of active features
    d_eff = Z_eff.shape[1]
    # how many features are active in each coalition
    abs_S = Z_eff.sum(1)

    # empty vector for the weights
    weights = torch.empty(Z_eff.shape[0], device = Z_eff.device, dtype = Z_eff.dtype)
    # miidle coalitions excluding empty and full coalitions
    middle = (abs_S > 0) & (abs_S < d_eff)

    #lgamma(d + 1) = log(d!)
    const = torch.lgamma(torch.tensor(float(d_eff),device = Z_eff.device, dtype = Z_eff.dtype) + 1)
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
    Expand coalitions from active feature space back to original feature space.

    Originally in pref_shap.setup_the_rest(...):
        self.Z = torch.zeros(Z.shape[0], self.m).to(self.device)
        self.Z[:, self.mask] = Z

    Pref-SHAP is computed in Z_eff but the final output should still have one entry per original features.
    Inactive features are set to 0. 

    """
    Z = torch.zeros(Z_eff.shape[0], n_features, device=Z_eff.device, dtype=Z_eff.dtype)
    Z[:, mask] = Z_eff
    return Z

def base_10_base_2(indices: np.array,d:int=10):
    """
    Convert integer coalition indentifiers into binary coalition vectors.
        Eg: index 5 with d=4 becomes [1,0,1,0] 
    """

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
    """ Randomly sample binary coaltiion vectors and remove duplicates. """

    P = np.random.binomial(size=(N,d), n=1, p=0.5)
    b = 2 ** np.arange(0, d)
    unique_ref = (b*P).sum(1)
    _,idx=np.unique(unique_ref,return_index=True)
    return P[idx,:]

def sample_Z(D,max_S):
    """Sample or enumerate coalitios for D active features."""

    max_range = min(2**D,2**63-1)
    if max_S>=max_range:
        configs= np.arange(max_range)
        return base_10_base_2(configs, D)
    else:
        return base_2_base_10(max_S, D)

def build_item_coalitions_and_weights(mask, n_features, n_samples, device, big_weight=1e5):
    """
    Build coalition matrix Z and KernalSHAP weights.


    Original Idea:
    coalition and weight construction in setup_the_rest()
    """


    d_eff = int(mask.sum().item())

    Z_eff = torch.from_numpy(sample_Z(d_eff, n_samples)).float().to(device)

    # empty and full coalitions 
    empty = torch.zeros(1, d_eff, device=device)
    full = torch.ones(1, d_eff, device=device)

    if Z_eff.shape[0] > 0:
        middle = Z_eff[(Z_eff.sum(1) > 0) & (Z_eff.sum(1) < d_eff)]
        Z_eff = torch.cat([empty, middle, full], dim=0)
    else:
        Z_eff = torch.cat([empty, full], dim=0)

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
    COmputes item_level PrefSHAP value function v(S) for one coalition.

    Implementation: Proposition 3.2 von original kernel_tensor_batch()
    value (S) = alpha^T * [Gamma(left,left) * Gamma(right,right)
                            - Gamma(left,right) * Gamma(right,left)]

    """

    S = S.bool()
    S_C = ~S

    device = X.device
    dtype = X.dtype
    n_ref = X.shape[0]

    # Regularization in the original code: n_ref · λ · I
    #   self.reg = self.N_x * self.lamb
    #   self.eye = torch.eye(self.N_x) * self.reg
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

    # Gamma_l_l = K_XlS_xlS * M_l_l
    # Gamma_r_r = K_XrS_xrS * M_l_r
    # Gamma_l_r = K_XlS_xrS * M_r_l
    # Gamma_r_l = K_XrS_xlS * M_r_r

    positive = (K_XlS_xlS * K_XrS_xrS * M_l_l * M_l_r)
    negative = (K_XlS_xrS * K_XrS_xlS * M_r_l * M_r_r)

    # DEBUG: for x_r = x_l.clone(), we expect Gamma_ll = Gamma_lr and Gamma_rr = Gamma_rl.
    if torch.equal(S, torch.tensor([1, 0, 0, 0, 0], device=S.device, dtype=S.dtype)):
        ''' 
       print("\nCoalition:", S.int().tolist())
        print("Gamma_ll:", Gamma_l_l)
        print("Gamma_rr:", Gamma_r_r)
        print("Gamma_lr:", Gamma_l_r)
        print("Gamma_rl:", Gamma_r_l)

        print("max |Gamma_ll - Gamma_lr| =",
          torch.max(torch.abs(Gamma_l_l - Gamma_l_r)))

        print("max |Gamma_rr - Gamma_rl| =",
          torch.max(torch.abs(Gamma_r_r - Gamma_r_l)))
        '''
    
        print(torch.max(torch.abs(K_XS_xlS - K_XS_xrS)))
        print(torch.max(torch.abs(K_XlS_xlS - K_XlS_xrS)))
        print(torch.max(torch.abs(K_XrS_xlS - K_XrS_xrS)))

        print(torch.max(torch.abs(cme_l - cme_r)))

        print(torch.max(torch.abs(M_l_l - M_l_r)))
        print(torch.max(torch.abs(M_r_l - M_r_r)))
        
    # 8. Preference structure
    # Proposition 3.2: original direction - reversed direction
    
    # pref_kernel_value = Gamma_l_l * Gamma_r_r - Gamma_l_r * Gamma_r_l
    pref_kernel_value = positive - negative

    # DEBUG:
    print(torch.max(torch.abs(pref_kernel_value)))

    # 9. Multiply with alpha
    # Original from value_observation:
    #   output = (self.alpha @ output).squeeze() - self.y_pred_mean
    value_S = alpha @ pref_kernel_value

    # optional centering by the model prediction mean
    if y_pred_mean is not None:
        value_S = value_S - y_pred_mean

    return value_S.reshape(1)

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
    Compute v(S) for all coalitions S in Z.
    Replaces original kernel_tensor_batch and value_observation path.
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
    Compute the final PREF-SHAP values beta from the Coalition Values using
        beta = (Z^T W Z)^(-1) Z^T W v

    Original in pref_shap.py:
        construct_values(...)
        OLS_solve(...)


    Y_cat   = v (coallitions values)
    Z       = Coalition Matrix
    weights = KernelSHAP Weights w
    beta    = final feature attribution vector
    """

    # If Y_cat is just a vector, ensure it is a column vector
    if Y_cat.ndim == 1:
        Y_cat = Y_cat.unsqueeze(-1)

    weights = weights.clone()

    # empty und full coalition get big weight
    weights[0] = big_weight
    weights[-1] = big_weight

    # Only solve for features that appear in at least one coallition.
    # Original OLS_solve:
    #   mask = Z_in.sum(0) > 0
    mask = Z.sum(0) > 0
    Z_masked = Z[:, mask]

    # Original construct_values:
    #   Y_target = weights * Y_cat
    # corresponds to Wv.
    Y_target = weights * Y_cat

    # Original OLS_solve:
    #   A = Z.t() @ (Z * weights)
    #   b = Z.t() @ Y_target
    # Corresponds to:
    #   A = Z^T W Z
    #   b = Z^T W v
    A = Z_masked.t() @ (Z_masked * weights)
    b = Z_masked.t() @ Y_target

    # Instead of inverting A -> solve A beta = b 
    # Original:
    #   L = torch.linalg.cholesky(A)
    #   sol = torch.cholesky_solve(b, L)
    L = torch.linalg.cholesky(A)
    sol = torch.cholesky_solve(b, L)

    # map beta back to original feature dimension
    beta = torch.zeros(Z.shape[1], Y_cat.shape[1], device=Z.device, dtype=Y_cat.dtype)
    beta[mask, :] = sol

    return beta
