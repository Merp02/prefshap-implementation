# check whether the implemnetation is internally consistent.
# does it produce valid numerical outputs?
import torch

def test_pref_shap_item(beta, Y_cat, weights, Z):
    print("beta shape:", beta.shape)
    print("Y_cat shape:", Y_cat.shape)
    print("weights shape:", weights.shape)
    print("Z shape:", Z.shape)

    # Z has n coalitions, then weights also needs n rows.
    # Every coalition has a corresponding KernalSHAP weight.
    assert Z.shape[0] == weights.shape[0]

    # Evrey coalition has exactly one coalition value v(x).
    assert Z.shape[0] == Y_cat.shape[0]

    # beta must have exactly one attribution per feature.
    assert beta.shape[0] == Z.shape[1]

    # testing edge cases
    assert torch.isfinite(beta).all()
    assert torch.isfinite(Y_cat).all()
    assert torch.isfinite(weights).all()

    print("Test passed.")
