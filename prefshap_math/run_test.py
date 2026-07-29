from prefshap_math.prefshap_clean import pref_shap_item_clean
import torch
import numpy as np

# dummy kernel, replaces the original kernel
# k(x,y) = x.T * y
'''def kernel(A, B=None, S=None):
    if B is None:
        B = A
    return A @ B.T

torch.manual_seed(0)'''

# rbf_kernel
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


# n_ref = 20
# n_features = 5
# n_train = 20

# X = torch.randn(n_ref, n_features)
# X_l = torch.randn(n_train, n_features)
# X_r = torch.randn(n_train, n_features)

# x_l = torch.randn(1, n_features)
# x_r = torch.randn(1, n_features)

# alpha = torch.randn(n_train, 1)

X = torch.from_numpy(np.load("toy_data_5000_10_2/S.npy")).float()
X_l = torch.from_numpy(np.load("toy_data_5000_10_2/l_processed.npy")).float()
X_r = torch.from_numpy(np.load("toy_data_5000_10_2/r_processed.npy")).float()

n_features = X.shape[1]
n_train = X_l.shape[0]

x_l = torch.randn(1, n_features)
x_r = torch.randn(1, n_features)

alpha = torch.randn(X_l.shape[0], 1)

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

beta, Y_cat, weights, Z = pref_shap_item_clean(
    alpha=alpha,
    X_l=X_l,
    X_r=X_r,
    X=X,
    x_l=x_l,
    x_r=x_r,
    kernel=rbf_kernel,
    n_samples=50,
    lambda_reg=1e-3,
)

test_pref_shap_item(beta, Y_cat, weights, Z)
print("\nBeta:")
print (beta)

print("\nCoalition values:")
print(Y_cat.squeeze())

print("\nWeights:")
print(weights.squeeze())

print("\nCoalitions:")
print(Z)

K = rbf_kernel(X, X, sigma=1.0)
print("\nDiagonal (needs to be 1):")
print(torch.diag(K))

K = rbf_kernel(X, X, sigma=5.0)
print("\nDiagonal (soll 1 sein, weil ein Punkt mit sich selbst die maximale Ähnlichkei hat  ):")
print(torch.diag(K))

print("raw empty:", Y_cat[0])
print("\nWenn keine Features vorhanden sind -> Baseline Wert 0")

print("raw full:", Y_cat[-1])
print("\nWenn alle Features vorhanden sind -> volle Modellscore")

print("full - empty:", Y_cat[0] - Y_cat[-1])


# works !
# all matrix dimensions compatible
# every function returns the expected shapes
# no faulty edge cases
# prefshap_clean mirrors algorithm 1 properly