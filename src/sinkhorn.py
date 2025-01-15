import torch

import time

from utils import F_eps, __discrete_checks

@torch.no_grad()
def sinkhorn(
    X1, # Contains the data matrix of shape (n1, d)
    X2, # Contains the data matrix of shape (n2, d)
    mu=None, # Contains the weight matrix of shape (n1,). If not provided, automatically set to uniform weights 1/n1
    nu=None, # Contains the weight matrix of shape (n2,). If not provided, automatically set to uniform weights 1/n2
    epsilon=1e-1, # Regularization parameter
    n_iter=1000, # Number of maximum iterations
    tol=1e-6, # Stopping criterion
    record_history=False, # Whether to record the history of the objective function to be maximized
    return_pi=False, # Whether to return the optimal transport plan
    device=None, # Device to use for computations
    pi_GT=None, # Ground truth regularized optimal transport plan
):
    # Checks for shapes and consistency :
    X1, X2, mu, nu, device = __discrete_checks(X1, X2, mu, nu, device)
    n1, _ = X1.shape
    n2, _ = X2.shape
    
    # Initialization :
    u = torch.zeros(n1)
    v = torch.ones(n2)
    C = torch.cdist(X1, X2, p=2) ** 2 # Squared Euclidean distance, shape (n1, n2)
    K = torch.exp(- C / epsilon)
    
    history = {
        'Transport Cost': [],
        'primal': [],
        'dual': [],
        'iter': [],
        'n_samples': [],
        'time': [],
        'TV_to_mu': [],
        'TV_to_nu': [],
    }
    if pi_GT is not None:
        history['TV(pi, pi_GT)'] = []

    # Sinkhorn iterations :
    for k in range(n_iter):
        start = time.time()
        u0 = u
        v0 = v
        u = mu / (torch.einsum('nm, m -> n', K, v) + 1e-10)
        v = nu / (torch.einsum('nm, n -> m', K, u) + 1e-10)
        end = time.time()

        # primal : inf < P | C > - epsilon H(P)
        # dual : sup < u', mu > + < v', nu > + < K, P >
        # /!\ u' and v' are the dual potentials, sinkhorn's u and v are NOT the dual potentials However, they are related by the following formula :
        # u = mu_i exp(u'_i / epsilon), v = nu_j exp(v'_j / epsilon)
        pi = torch.einsum('n, nm, m -> nm', u, K, v)
        pi = pi / pi.sum()
        transport_cost = torch.sum(pi * C)
        primal = torch.sum(pi * C) + epsilon * torch.sum(pi * (torch.log(pi + 1e-20)))
        u_ = epsilon * (torch.log(u) - torch.log(mu))
        v_ = epsilon * (torch.log(v) - torch.log(nu))
        dual = torch.sum(mu * u_) + torch.sum(nu * v_) - epsilon * torch.sum(torch.exp((u_[:, None] + v_[None, :] - C) / epsilon) * mu[:, None] * nu[None, :]) + epsilon
        history['iter'].append(k)
        history["time"].append(end - start + history["time"][-1] if k > 0 else end - start)
        history["n_samples"].append(n1 + n2 + history["n_samples"][-1] if k > 0 else n1 + n2)
        history['Transport Cost'].append(transport_cost.item())
        history['primal'].append(primal.item())
        history['dual'].append(dual.item())
        history['TV_to_mu'].append((pi.sum(1) - mu).abs().sum().item())
        history['TV_to_nu'].append((pi.sum(0) - nu).abs().sum().item())
        if pi_GT is not None:
            history['TV(pi, pi_GT)'].append(torch.norm(pi - pi_GT, p=1).item())

        if k > 0 and history['dual'][-1] - history['dual'][-2] < tol:
            break
    if return_pi:
        # Compute the optimal transport plan
        pi = torch.einsum('n, nm, m -> nm', u, K, v)
        if record_history:
            return u, v, pi, history
        return u, v, pi
    if record_history:
        return u, v, history
    return u, v
