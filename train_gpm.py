'''
GPM:
g_hat(x_l, x_r) = Σ_j alpha_j k_E((X_l_j, X_r_j), (x_l, x_r))

tranieren mit:

softplus(-y * g_vec) + 0.5 * lambda * alpha^T K_E alpha

Idee:
Das GPM/PREF-SHAP-Modell lernt g direkt auf Paaren, 
und der Generalized Preferential Kernel k_E erzwingt die skew-symmetry. 

Präferenzen werden über p(y | x_l, x_r)=σ(y g(x_l,x_r)) modelliert, 
und k_E wird als Kernel auf Item-Paaren genutzt.


Schritt:
1. Zuerst berechnen:
K_E = K_ll * K_rr - K_lr * K_rl

2. Berechnen: Modellscore für jedes Trainingsduell
--> g_vec = K_E @ alpha 
mit g_vec[j] = g_hat(X_l_j, X_r_j)

3. Loss minimieren:
Logitic loss:log(1 + exp(-y * g_hat))
data_loss = softplus(-y * g_vec).mean()

RKHS-Regularization:
reg_loss = 0.5 * lambda_reg * alpha.T @ K_E @ alpha

loss = data_loss + reg_loss

Wenn y = +1, soll g_hat(x_l, x_r) positiv werden.
Wenn y = -1, soll g_hat(x_l, x_r) negativ werden.
'''


















'''
Training objective (standard L2-regularised kernel logistic regression):
    L(alpha) = mean( softplus(-y * g_vec) ) + 0.5 * lambda * alpha^T K_E alpha
 

This is convex (log-loss is convex; alpha^T K_E alpha >= 0 because k_E has an
explicit feature map -> it's a valid PSD kernel), so plain gradient descent
on alpha (via Adam) converges without needing FALKON / Nystrom / IRLS.
 
No separate skew-symmetry constraint is needed: k_E((x_l_j,x_r_j),(a,b)) =
-k_E((x_l_j,x_r_j),(b,a)) for ANY alpha, by construction of k_E.

'''