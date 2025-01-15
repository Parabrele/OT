import torch
import numpy as np
import time

from utils import __discrete_checks, DEFAULT_MAP_MLP, F_eps, H_eps, DEFAULT_DUAL_MLP

def __zero_grad_u_v(u):
    # is u is a nn.Module, set all parameters gradients to zero. Otherwise, set gradient to zero.
    if isinstance(u, torch.nn.Module):
        for param in u.parameters():
            if param.grad is not None:
                param.grad.zero_()
    else:
        if u.grad is not None:
            u.grad.zero_()

def __update_u_v(u, lr, k, idxs=None):
    if idxs is None:
        for param in u.parameters():
            param += (lr / k) * param.grad
    else:
        u.scatter_add_(0, idxs, (lr / k) * u.grad[idxs])

def stochastic_dual(
    X1=None, # If source measure is
                # - discrete or discretized, mendatory, contains the data matrix of shape (n1, d)
                # - continuous, ignored
    X2=None, # If target measure is
                # - discrete or discretized, mendatory, contains the data matrix of shape (n2, d)
                # - continuous, ignored
    mu=None, # If source measure is
                # - discrete, optional, contains the weight matrix of shape (n1,). If not provided, automatically set to uniform weights 1/n1
                # - continuous, mendatory, contains the callable function providing samples of shape (d,) or batch of samples of shape (p, d)
    nu=None, # If target measure is
                # - discrete, optional, contains the weight matrix of shape (n2,). If not provided, automatically set to uniform weights 1/n2
                # - continuous, mendatory, contains the callable function providing samples of shape (d,) or batch of samples of shape (p, d)
    epsilon=1e-1, # Regularization parameter
    n_iter=1000, # Number of maximum iterations for mini-batch gradient descent
    lr=1e-3, # Learning rate for mini-batch gradient descent
    batch_size=100, # Batch size for mini-batch gradient descent
    tol=1e-6, # Stopping criterion for mini-batch gradient descent
    device=None, # Device to use for computations
    seed=None, # Random seed for reproducibility
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    record_history=False, # whether to record the history of the objective function to be maximized
    pi_GT=None, # transport plan used as a reference. Callable function X x Y -> R
    force_mlp=False, # Whether to force the use of a MLP for the dual potentials even in the discrete case
):  
    if seed is not None:
        torch.manual_seed(seed)

    # Checks and initialization :
    if X1 is not None:
        n1, d = X1.shape
        if isinstance(mu, torch.distributions.Distribution):
            raise ValueError("If X1 is provided, mu should be a weight matrix corresponding to the source measure or None")
        if mu is None:
            mu = torch.ones(n1) / n1
        _n1 = mu.shape[0]
        if _n1 != n1:
            raise ValueError(f"mu should have shape ({n1},), got ({_n1},)")
        c_mu = False
        X1_choice = torch.distributions.Categorical(probs=mu)
        if force_mlp:
            u = DEFAULT_DUAL_MLP(d)
        else:
            u = torch.zeros(n1, requires_grad=True)
    else:
        if not isinstance(mu, torch.distributions.Distribution):
            raise ValueError("If X1 is not provided, mu should be a torch.distributions.Distribution corresponding to the source measure")
        c_mu = True
        try:
            d = mu.event_shape[0]
        except IndexError:
            d = 1
        u = DEFAULT_DUAL_MLP(d)
    if X2 is not None:
        n2, _d = X2.shape
        if isinstance(nu, torch.distributions.Distribution):
            raise ValueError("If X2 is provided, nu should be a weight matrix corresponding to the target measure or None")
        if nu is None:
            nu = torch.ones(n2) / n2
        _n2 = nu.shape[0]
        if _n2 != n2:
            raise ValueError(f"nu should have shape ({n2},), got ({_n2},)")
        if 'd' in locals() and d != _d:
            raise ValueError(f"X1 and X2 should have the same number of features, got {d} and {_d}")
        c_nu = False
        X2_choice = torch.distributions.Categorical(probs=nu)
        if force_mlp:
            v = DEFAULT_DUAL_MLP(d)
        else:
            v = torch.zeros(n2, requires_grad=True)
    else:
        if not isinstance(nu, torch.distributions.Distribution):
            raise ValueError("If X2 is not provided, nu should be a torch.distributions.Distribution corresponding to the target measure")
        c_nu = True
        v = DEFAULT_DUAL_MLP(d)

    if record_history:
        history = {
            "Transport Cost": [],
            "primal": [],
            "dual": [],
            "time": [],
            "nsamples": [],
            "iter": [],
            "TV_to_mu": [],
            "TV_to_nu": [],
        }
        if pi_GT is not None:
            history["TV(pi, pi_GT)"] = []

    # Mini-batch gradient descent :
    for cur_iter in range(n_iter):
        start = time.time()

        k = np.sqrt(cur_iter + 1)

        # Zero the gradients
        __zero_grad_u_v(u)
        __zero_grad_u_v(v)

        if not c_mu: X1_idxs = X1_choice.sample((batch_size,))
        if not c_nu: X2_idxs = X2_choice.sample((batch_size,))

        X1_batch = mu.sample((batch_size,)) if c_mu else X1[X1_idxs]
        X1_prob = mu.log_prob(X1_batch).exp() if c_mu else mu[X1_idxs]

        X2_batch = nu.sample((batch_size,)) if c_nu else X2[X2_idxs]
        X2_prob = nu.log_prob(X2_batch).exp() if c_nu else nu[X2_idxs]

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)

        u_values = u(X1_batch)[:, 0] if isinstance(u, torch.nn.Module) else u[X1_idxs]
        v_values = v(X2_batch)[:, 0] if isinstance(v, torch.nn.Module) else v[X2_idxs]

        G = (u_values[:, None] + v_values[None, :] - C)
        G = - epsilon * torch.exp(G / epsilon) * (1/batch_size**2)

        # Compute the scalar objective for automatic differentiation
        dual_objective = (1 / batch_size) * torch.sum(u_values) + (1 / batch_size) * torch.sum(v_values) + torch.sum(G) + epsilon
        dual_objective.backward()
        
        # Update dual potentials
        with torch.no_grad():
            __update_u_v(u, lr, k, X1_idxs if isinstance(u, torch.Tensor) else None)
            __update_u_v(v, lr, k, X2_idxs if isinstance(v, torch.Tensor) else None)

            end = time.time()

            if record_history:
                pi = (
                    torch.exp((u_values[:, None] + v_values[None, :] - C) / epsilon)
                    * X1_prob[:, None]
                    * X2_prob[None, :]
                )
                pi = pi / pi.sum()
                transport_cost = torch.sum(pi / pi.sum() * C)
                primal = torch.sum(pi * C) + epsilon * torch.sum(pi * (torch.log(pi + 1e-10)))
                if pi_GT is not None and callable(pi_GT):
                    pi_GT_batch = pi_GT(X1_batch, X2_batch)
                if pi_GT is not None and not callable(pi_GT):
                    pi_GT_batch = pi_GT[X1_idxs, :][:, X2_idxs]
                
                history["Transport Cost"].append(transport_cost.item())
                history["primal"].append(primal.item())
                history["dual"].append(dual_objective.item())
                history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
                history["nsamples"].append(2 * batch_size + history["nsamples"][-1] if cur_iter > 0 else 2 * batch_size)
                history["iter"].append(cur_iter)
                history["TV_to_nu"].append((pi.sum(axis=0) - X2_prob / X2_prob.sum()).abs().sum().item())
                history["TV_to_mu"].append((pi.sum(axis=1) - X1_prob / X1_prob.sum()).abs().sum().item())
                if pi_GT is not None:
                    history["TV(pi, pi_GT)"].append(np.abs(pi - pi_GT_batch / pi_GT_batch.sum()).sum())

    if record_history:
        return u, v, history                    
    return u, v

def discrete_map_learning(
    u, # Dual potentials of shape (n1,)
    v, # Dual potentials of shape (n2,)
    epsilon, # Regularization parameter used to get u and v
    X1, #
    X2, # 
    mu=None, # 
    nu=None, # 
    f_theta=None, # Contains the callable function parameterized by theta. Should be a torch.nn.Module. If None is provided, a default MLP is used.
    batch_size=100, # Batch size for mini-batch gradient descent
    lr=1e-3, # Learning rate for mini-batch gradient descent
    n_iter=1000, # Number of maximum iterations for mini-batch gradient descent
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    tol=1e-6, # Stopping criterion for mini-batch gradient descent
    device=None, # Device to use for computations
    Monge_map=None,
    record_history=False, # whether to record the history of the objective function to be maximized
):
    """
    u and v are from the stochastic dual algorithm, not from sinkhorn. Only the closed form projection is used for sinkhorn.
    """
    X1, X2, mu, nu, device = __discrete_checks(X1, X2, mu, nu, device)
    n1, d = X1.shape
    n2, _ = X2.shape

    if f_theta is None:
        f_theta = DEFAULT_MAP_MLP(d)
    optimizer = torch.optim.AdamW(f_theta.parameters(), lr=lr)

    if record_history:
        history = {
            "time": [],
            "iter": [],
            "n_samples": [],
            "L": [],
            "MSE T(X1) - Monge_map": [],
        }
    
    X1_choice = torch.distributions.Categorical(probs=mu)
    X2_choice = torch.distributions.Categorical(probs=nu)

    for cur_iter in range(n_iter):
        start = time.time()

        # Zero the gradients
        optimizer.zero_grad()
        # for param in f_theta.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()

        X1_idxs = X1_choice.sample((batch_size,))
        X2_idxs = X2_choice.sample((batch_size,))
        X1_batch = X1[X1_idxs]
        X2_batch = X2[X2_idxs]

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)

        if isinstance(u, torch.Tensor):
            u_values = u
            v_values = v
        else:
            u_values = u(X1)[:, 0]
            v_values = v(X2)[:, 0]

        H = H_eps(u_values[X1_idxs, None] + v_values[None, X2_idxs] - C, epsilon, entropy=entropy) # shape (batch_size, batch_size)

        L = H * torch.cdist(f_theta(X1_batch), X2_batch, p=2) # shape (batch_size, batch_size)
        L = ( (1/batch_size) ** 2 * L).sum()
        L.backward()
        optimizer.step()

        end = time.time()

        with torch.no_grad():
            # for param in f_theta.parameters():
            #     param -= (lr/k) * param.grad
            
            if record_history:
                history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
                history["iter"].append(cur_iter)
                history["n_samples"].append(2 * batch_size + history["n_samples"][-1] if cur_iter > 0 else 2 * batch_size)
                history["L"].append(L.item())
                if Monge_map is not None:
                    history["MSE T(X1) - Monge_map"].append((f_theta(X1) - Monge_map).pow(2).mean().item())
        
        """
        print("MSE T(X1) - X1 :", (T(X1) - X1).pow(2).mean())
        print("MSE T(X1) - X2 :", (T(X1) - X2).pow(2).mean())
        print("MSE T(X1) - Monge_map :", (T(X1) - Monge_map).pow(2).mean())"""

    if record_history:
        return f_theta, history
    return f_theta
 
def continuous_map_learning(
    mu, # torch.distributions.Distribution : source measure
    nu, # torch.distributions.Distribution : target measure
    u, # Contains callable function providing the dual potentials
    v, # Contains callable function providing the dual potentials
    epsilon, # Regularization parameter
    f_theta=None, # Contains a callable function parameterized by theta. Should be a torch.nn.Module.
    batch_size=100, # Batch size for mini-batch gradient descent
    lr=1e-3, # Learning rate for mini-batch gradient descent
    n_iter=1000, # Number of maximum iterations for mini-batch gradient descent
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    tol=1e-6, # Stopping criterion for mini-batch gradient descent*
    device=None, # Device to use for computations
    Monge_map=None, # Callable function X -> Y
    record_history=False, # whether to record the history of the objective function to be maximized
):
    if f_theta is None:
        f_theta = DEFAULT_MAP_MLP(mu.event_shape[0])
    optimizer = torch.optim.AdamW(f_theta.parameters(), lr=lr)

    if record_history:
        history = {
            "time": [],
            "iter": [],
            "n_samples": [],
            "L": [],
            "MSE T(X1) - Monge_map": [],
        }

    for cur_iter in range(n_iter):
        start = time.time()

        # Zero the gradients
        optimizer.zero_grad()

        X1_batch = mu.sample((batch_size,))
        X2_batch = nu.sample((batch_size,))

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)

        H = H_eps(u(X1_batch)[:, None] + v(X2_batch)[None, :] - C, epsilon, entropy=entropy) # shape (batch_size, batch_size)

        L = H * torch.cdist(f_theta(X1_batch), X2_batch, p=2) # shape (batch_size, batch_size)
        L = ( (1/batch_size) ** 2 * L).sum()
        L.backward()
        optimizer.step()

        end = time.time()

        with torch.no_grad():
            if record_history:
                history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
                history["iter"].append(cur_iter)
                history["n_samples"].append(2 * batch_size + history["n_samples"][-1] if cur_iter > 0 else 2 * batch_size)
                history["L"].append(L.item())
                if Monge_map is not None:
                    history["MSE T(X1) - Monge_map"].append((f_theta(X1_batch) - Monge_map(X1_batch)).pow(2).mean().item())

    if record_history:
        return f_theta, history
    return f_theta

def map_learning():
    pass

##########
# Archived code
##########

def _discrete_stochastic_dual_manual_grad(
    X1, # Contains the data matrix of shape (n1, d)
    X2, # Contains the data matrix of shape (n2, d)
    mu=None, # Contains the weight matrix of shape (n1,). If not provided, automatically set to uniform weights 1/n1
    nu=None, # Contains the weight matrix of shape (n2,). If not provided, automatically set to uniform weights 1/n2
    epsilon=1e-1, # Regularization parameter
    n_iter=1000, # Number of maximum iterations
    lr=1e-3, # Learning rate
    batch_size=100, # Batch size for mini-batch gradient descent
    tol=1e-6, # Stopping criterion
    device=None, # Device to use for computations
    seed=None, # Random seed for reproducibility
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    record_history=False, # whether to record the history of the objective function to be maximized
    pi_sink=None, # Optimal transport plan computed with Sinkhorn algorithm
):
    # Entropy regularization only for now. TODO : add L2 regularization

    if seed is not None:
        torch.manual_seed(seed)

    # Checks for shapes and consistency :
    X1, X2, mu, nu, device = __discrete_checks(X1, X2, mu, nu, device)
    n1, _ = X1.shape
    n2, _ = X2.shape
    
    u = torch.zeros(n1)
    v = torch.zeros(n2)
    if record_history:
        history = {
            "dual": [],
            "time": [],
            "nsamples": [],
            "iter": [],
            "TV_to_mu": [],
            "TV_to_nu": [],
        }
        if pi_sink is not None:
            history["TV(pi, pi_sinkhorn)"] = []
        
        C_complete = torch.cdist(X1, X2, p=2) ** 2

    """
    When sampling ~ mu, there is no need to weight the integral, all samples have the same 1/n1 weight as they are already weighted by mu.
    When sampling uniformly, every sample should be weighted by mu to get the correct expectation.
    """

    X1_choice = torch.distributions.Categorical(probs=mu)
    X2_choice = torch.distributions.Categorical(probs=nu)
    # X1_choice = lambda: torch.randperm(n1)[:batch_size]
    # X2_choice = lambda: torch.randperm(n2)[:batch_size]

    for cur_iter in range(n_iter):
        start = time.time()

        k = np.sqrt(cur_iter + 1)
        X1_idxs = X1_choice.sample((batch_size,))
        X2_idxs = X2_choice.sample((batch_size,))

        X1_batch = X1[X1_idxs]
        X2_batch = X2[X2_idxs]

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)
        G = (u[X1_idxs, None] + v[None, X2_idxs] - C)
        # G = torch.exp(G / epsilon) * mu[X1_idxs, None] * nu[None, X2_idxs] # shape (batch_size, batch_size)
        G = - torch.exp(G / epsilon) * (1/batch_size) ** 2 # shape (batch_size, batch_size)

        # mu_scale = mu.sum() / mu[X1_idxs].sum()
        # nu_scale = nu.sum() / nu[X2_idxs].sum()
        mu_scale = n1 / batch_size
        nu_scale = n2 / batch_size

        # update_u = mu_scale * nu_scale * ((1/nu_scale) * mu[X1_idxs] + G.sum(1)) # this gives an estimate of the full gradient using only a mini-batch, so we have to scale it
        # update_v = mu_scale * nu_scale * ((1/mu_scale) * nu[X2_idxs] + G.sum(0)) # same here
        update_u = 1 / batch_size + G.sum(1)
        update_v = 1 / batch_size + G.sum(0)
        
        u.scatter_add_(0, X1_idxs, (lr / k) * update_u)
        v.scatter_add_(0, X2_idxs, (lr / k) * update_v)

        end = time.time()

        if record_history:
            # compute the objective function to be maximized : sum_{i} u(x_i) * mu_i + sum_{j} v(y_j) * nu_j + sum_{ij} F_{\epsilon}(u(x_i) + v(y_j) - c(x_i, y_j)) * mu_i * nu_j
            F = F_eps(u[X1_idxs, None] + v[None, X2_idxs] - C, epsilon, entropy) # shape (batch_size, batch_size)
            dual = (1/batch_size) * torch.sum(u[X1_idxs]) + (1/batch_size) * torch.sum(v[X2_idxs]) + (1/batch_size**2) * torch.sum(F)
            pi = (
                np.exp((u[:, None] + v[None, :] - C_complete[:, :].numpy()) / epsilon)
                * mu[:, None].numpy()
                * nu[None, :].numpy()
            )
            pi = pi / pi.sum()
            history["dual"].append(dual)
            history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
            history["nsamples"].append(2 * batch_size + history["nsamples"][-1] if cur_iter > 0 else 2 * batch_size)
            history["iter"].append(cur_iter)
            history["TV_to_nu"].append((pi.sum(axis=0) - nu).abs().sum())
            history["TV_to_mu"].append((pi.sum(axis=1) - mu).abs().sum())
            if pi_sink is not None:
                history["TV(pi, pi_sinkhorn)"].append(np.abs(pi - pi_sink).sum())

        if torch.norm(update_u) < tol and torch.norm(update_v) < tol:
            break

    if record_history:
        return u, v, history
    return u, v

def __discrete_stochastic_dual(
    X1, # Contains the data matrix of shape (n1, d)
    X2, # Contains the data matrix of shape (n2, d)
    mu=None, # Contains the weight matrix of shape (n1,). If not provided, automatically set to uniform weights 1/n1
    nu=None, # Contains the weight matrix of shape (n2,). If not provided, automatically set to uniform weights 1/n2
    epsilon=1e-1, # Regularization parameter
    n_iter=1000, # Number of maximum iterations
    lr=1e-3, # Learning rate
    batch_size=100, # Batch size for mini-batch gradient descent
    tol=1e-6, # Stopping criterion
    device=None, # Device to use for computations
    seed=None, # Random seed for reproducibility
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    record_history=False, # whether to record the history of the objective function to be maximized
    pi_sink=None, # Optimal transport plan computed with Sinkhorn algorithm
):
    # Entropy regularization only for now. TODO : add L2 regularization

    if seed is not None:
        torch.manual_seed(seed)

    # Checks for shapes and consistency :
    X1, X2, mu, nu, device = __discrete_checks(X1, X2, mu, nu, device)
    n1, _ = X1.shape
    n2, _ = X2.shape
    
    u = torch.zeros(n1, requires_grad=True)
    v = torch.zeros(n2, requires_grad=True)

    if record_history:
        history = {
            "dual": [],
            "time": [],
            "nsamples": [],
            "iter": [],
            "TV_to_mu": [],
            "TV_to_nu": [],
        }
        if pi_sink is not None:
            history["TV(pi, pi_sinkhorn)"] = []
        C_complete = torch.cdist(X1, X2, p=2) ** 2

    X1_choice = torch.distributions.Categorical(probs=mu)
    X2_choice = torch.distributions.Categorical(probs=nu)

    for cur_iter in range(n_iter):
        start = time.time()

        k = np.sqrt(cur_iter + 1)

        # Zero the gradients
        if u.grad is not None:
            u.grad.zero_()
            v.grad.zero_()

        X1_idxs = X1_choice.sample((batch_size,))
        X2_idxs = X2_choice.sample((batch_size,))

        X1_batch = X1[X1_idxs]
        X2_batch = X2[X2_idxs]

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)
        G = (u[X1_idxs, None] + v[None, X2_idxs] - C)
        G = - epsilon * torch.exp(G / epsilon) * (1/batch_size**2) # shape (batch_size, batch_size)

        # Compute the scalar objective for automatic differentiation
        dual_objective = (1 / batch_size) * torch.sum(u[X1_idxs]) + (1 / batch_size) * torch.sum(v[X2_idxs]) + torch.sum(G)

        # Compute the gradients
        dual_objective.backward()

        # Update the dual potentials
        with torch.no_grad():
            u.scatter_add_(0, X1_idxs, (lr / k) * u.grad[X1_idxs])
            v.scatter_add_(0, X2_idxs, (lr / k) * v.grad[X2_idxs])

            end = time.time()

            if record_history:
                # compute the objective function to be maximized : sum_{i} u(x_i) * mu_i + sum_{j} v(y_j) * nu_j + sum_{ij} F_{\epsilon}(u(x_i) + v(y_j) - c(x_i, y_j)) * mu_i * nu_j
                F = F_eps(u[X1_idxs, None] + v[None, X2_idxs] - C, epsilon, entropy) # shape (batch_size, batch_size)
                dual = (1/batch_size) * torch.sum(u[X1_idxs]) + (1/batch_size) * torch.sum(v[X2_idxs]) + (1/batch_size**2) * torch.sum(F)
                pi = (
                    np.exp((u[:, None].detach() + v[None, :].detach() - C_complete[:, :].detach().numpy()) / epsilon)
                    * mu[:, None].numpy()
                    * nu[None, :].numpy()
                )
                pi = pi / pi.sum()
                history["dual"].append(dual.item())
                history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
                history["nsamples"].append(2 * batch_size + history["nsamples"][-1] if cur_iter > 0 else 2 * batch_size)
                history["iter"].append(cur_iter)
                history["TV_to_nu"].append((pi.sum(axis=0) - nu).abs().sum().item())
                history["TV_to_mu"].append((pi.sum(axis=1) - mu).abs().sum().item())
                if pi_sink is not None:
                    history["TV(pi, pi_sinkhorn)"].append(np.abs(pi - pi_sink).sum())

        if torch.norm(u.grad[X1_idxs]) < tol and torch.norm(v.grad[X2_idxs]) < tol:
            break

    if record_history:
        return u, v, history
    return u, v

# /!\ This function was never tested !
def __continuous_stochastic_dual(
    mu, # The continuous source measure, for which one can call mu.sample((d,)) to get a sample of shape (d,) in X.
    nu, # The continuous target measure, for which one can call nu.sample((d,)) to get a sample of shape (d,) in Y.
    epsilon=1e-1, # Regularization parameter
    n_iter=1000, # Number of maximum iterations
    lr=1e-3, # Learning rate
    batch_size=100, # Batch size for mini-batch gradient descent
    tol=1e-6, # Stopping criterion
    device=None, # Device to use for computations
    seed=None, # Random seed for reproducibility
    entropy=True, # Whether to use entropy regularization. If False, uses L2 regularization
    record_history=False, # whether to record the history of the objective function to be maximized
    pi_GT=None, # Optimal transport plan used as a reference. Callable function X x Y -> R
):
    # Entropy regularization only for now. TODO : add L2 regularization

    if seed is not None:
        torch.manual_seed(seed)

    # Checks for shapes and consistency :
    d = mu.event_shape[0]
    if d != nu.event_shape[0]:
        raise ValueError("mu and nu should have the same number of features")
    
    u = DEFAULT_DUAL_MLP(d)
    v = DEFAULT_DUAL_MLP(d)

    if record_history:
        history = {
            "dual": [],
            "time": [],
            "nsamples": [],
            "iter": [],
            "TV_to_mu": [],
            "TV_to_nu": [],
        }
        if pi_GT is not None:
            history["TV(pi, pi_GT)"] = []

    for cur_iter in range(n_iter):
        start = time.time()

        k = np.sqrt(cur_iter + 1)

        # Zero the gradients
        for param in u.parameters():
            if param.grad is not None:
                param.grad.zero_()
        for param in v.parameters():
            if param.grad is not None:
                param.grad.zero_()

        X1_batch = mu.sample((batch_size,)) # shape (batch_size, d)
        X1_prob = mu.log_prob(X1_batch).exp() # shape (batch_size,) : likelihood of each sample, as given by the probability density function
        X2_batch = nu.sample((batch_size,)) # shape (batch_size, d)
        X2_prob = nu.log_prob(X2_batch).exp()

        u_values = u(X1_batch) # shape (batch_size,)
        v_values = v(X2_batch) # shape (batch_size,)

        C = torch.cdist(X1_batch, X2_batch, p=2) ** 2 # Squared Euclidean distance, shape (batch_size, batch_size)
        G = (u_values[:, None] + v_values[None, :] - C)
        G = - epsilon * torch.exp(G / epsilon) * (1/batch_size**2) # shape (batch_size, batch_size) # TODO : use F_eps

        # Compute the (scalar) dual objective for automatic differentiation
        dual_objective = (1 / batch_size) * torch.sum(u_values) + (1 / batch_size) * torch.sum(v_values) + torch.sum(G)

        # Compute the gradients
        dual_objective.backward()

        # Update the dual potentials
        with torch.no_grad():
            for param in u.parameters():
                param += (lr / k) * param.grad
            for param in v.parameters():
                param += (lr / k) * param.grad
            
            end = time.time()

            if record_history:
                # TODO : record primal objective.
                pi = (
                    np.exp((u_values[:, None] + v_values[None, :] - C[:, :].detach().numpy()) / epsilon)
                    * X1_prob[:, None].detach().numpy()
                    * X2_prob[None, :].detach().numpy()
                )
                pi_GT_batch = pi_GT(X1_batch, X2_batch)
                history["dual"].append(dual_objective.item())
                history["time"].append(end - start + history["time"][-1] if cur_iter > 0 else end - start)
                history["nsamples"].append(2 * batch_size + history["nsamples"][-1] if cur_iter > 0 else 2 * batch_size)
                history["iter"].append(cur_iter)
                history["TV_to_nu"].append((pi.sum(axis=0) - X2_prob.detach().numpy()).abs().sum())
                history["TV_to_mu"].append((pi.sum(axis=1) - X1_prob.detach().numpy()).abs().sum())
                if pi_GT is not None:
                    history["TV(pi, pi_GT)"].append(np.abs(pi - pi_GT).sum())

    if record_history:
        return u, v, history
    return u, v
