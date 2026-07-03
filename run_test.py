from prefshap_clean import pref_shap_item_clean
from prefshap_test import test_pref_shap_item
import torch
import numpy as np

# dummy kernel, replaces the original kernel
# k(x,y) = x.T * y
def kernel(A, B=None, S=None):
    if B is None:
        B = A
    return A @ B.T

torch.manual_seed(0)

# n_ref = 20
# n_features = 5
# n_train = 20

# X = torch.randn(n_ref, n_features)
# X_l = torch.randn(n_train, n_features)
# X_r = torch.randn(n_train, n_features)

#x_l = torch.randn(1, n_features)
#x_r = torch.randn(1, n_features)

# alpha = torch.randn(n_train, 1)

X = torch.from_numpy(np.load("toy_data_5000_10_2/S.npy")).float()
X_l = torch.from_numpy(np.load("toy_data_5000_10_2/l_processed.npy")).float()
X_r = torch.from_numpy(np.load("toy_data_5000_10_2/r_processed.npy")).float()

n_features = X.shape[1]
n_train = X_l.shape[0]

x_l = torch.randn(1, n_features)
x_r = torch.randn(1, n_features)

alpha = torch.randn(X_l.shape[0], 1)

beta, Y_cat, weights, Z = pref_shap_item_clean(
    alpha=alpha,
    X_l=X_l,
    X_r=X_r,
    X=X,
    x_l=x_l,
    x_r=x_r,
    kernel=kernel,
    n_samples=50,
    lambda_reg=1e-3,
)

test_pref_shap_item(beta, Y_cat, weights, Z)
print("\nBeta:")
print(beta)

print("\nCoalition values:")
print(Y_cat.squeeze())

print("\nWeights:")
print(weights.squeeze())

print("\nCoalitions:")
print(Z)


# works !
# all matrix dimensions compatible
# every function returns the expected shapes
# no faulty edge cases
# prefshap_clean mirrors algorithm 1 properly
