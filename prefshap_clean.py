from prefshap_core import (
    active_features_item,
    build_item_coalitions_and_weights,
    compute_pref_values_item_all_S,
    solve_weighted_regression_clean,
)

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
    complete item-only PREF-SHAP implementation without context variables

    Algorithmus 1:
        1. Determine active item features
        2. Build coalitions Z and KernalSHAP weights W
        3. Compute coalition values v(S) 
        4. Solve beta with weighted least squares
        5. Return beta
    """

    device = X.device

    alpha = alpha.t().to(device)
    X_l = X_l.to(device)
    X_r = X_r.to(device)
    X = X.to(device)
    x_l = x_l.to(device)
    x_r = x_r.to(device)

    # 1: determine effective feature dimension
    mask = active_features_item(X_l, X_r).to(device)

    # 2: Coalitions Z and KernelSHAP weights W
    Z, weights = build_item_coalitions_and_weights(
        mask=mask,
        n_features=X.shape[1],
        n_samples=n_samples,
        device=device,
        big_weight=big_weight,
    )

    # 3-8: compute Coalition Values v(S)
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

    # 10: return beta
    return beta, Y_cat, weights, Z


