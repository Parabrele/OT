import math
import torch
from utils import H_eps, __discrete_checks

@torch.no_grad()
def closed_form_projection(
    X1, # Contains the data matrix of shape (n1, d)
    X2, # Contains the data matrix of shape (n2, d)
    mu, # Contains the weight matrix of shape (n1,)
    nu, # Contains the weight matrix of shape (n2,)
    u, # Contains the dual potentials of shape (n1,)
    v, # Contains the dual potentials of shape (n2,)
    epsilon, # Regularization parameter
    sinkhorn, # Boolean indicating whether u and v come from a Sinkhorn algorithm or a stochastic dual algorithm
    entropy=True, # TODO
    record_history=False, # Whether to record the history of the objective function to be maximized (here, only one "iteration")
):
    """
    In the discrete case, the closed form projection is given by :
    \widetilde{\pi} = \frac{\pi @ X2}{\mu}
    """
    X1, X2, mu, nu, _ = __discrete_checks(X1, X2, mu, nu, None)
    n1, _ = X1.shape
    n2, _ = X2.shape
    C = torch.cdist(X1, X2, p=2) ** 2
    if sinkhorn:
        K = torch.exp(- C / epsilon)
        pi = torch.einsum('n, nm, m -> nm', u, K, v)
    else:
        if isinstance(u, torch.Tensor):
            u_values = u
            v_values = v
        else:
            u_values = u(X1)[:, 0]
            v_values = v(X2)[:, 0]
            
        K = torch.exp((u_values[:, None] + v_values[None, :] - C) / epsilon)
        pi = K * mu[:, None] * nu[None, :]

    # Shapes : pi : (n1, n2), mu : (n1,), X2 : (n2, d), pi_tilde : (n1, d)
    # pi_tilde = \frac{pi @ X2}{mu}
    pi_tilde = torch.einsum('nm, mj -> nj', pi, X2) / mu[:, None]

    if record_history:
        L = torch.cdist(pi_tilde, X2, p=2) * pi
        return pi_tilde, L.sum()

    return pi_tilde

def closed_form_gaussian_pi(
    mu_1, # mean of the source Gaussian distribution
    Sigma_1, # covariance matrix of the source Gaussian distribution
    mu_2, # mean of the target Gaussian distribution
    Sigma_2, # covariance matrix of the target Gaussian distribution
    epsilon, # Regularization parameter
):
    """
    Return the closed form regularized OT plan between two Gaussian distributions.
    """
    # epsilon = 2 sigma^2 -> sigma = sqrt(epsilon / 2)
    # mu = N(a, A), nu = N(b, B)
    # D_sigma := (4 A^1/2 B A^1/2 + sigma^4 I)^1/2
    # C_sigma := 1/2 A^1/2 D_sigma A^-1/2 - sigma^2/2 I
    # pi = N((a, b), ((A, C_sigma), (C_sigma^T, B)))

    sigma = math.sqrt(epsilon / 2)

    U, S, V = torch.svd(Sigma_1)
    A_sqrt = U @ torch.diag(torch.sqrt(S)) @ V.t()
    A_sqrt_inv = U @ torch.diag(1 / torch.sqrt(S)) @ V.t()
    I = torch.eye(Sigma_1.shape[0])

    D_sigma = 4 * A_sqrt @ Sigma_2 @ A_sqrt + sigma ** 4 * I
    U, S, V = torch.svd(D_sigma)
    D_sigma = U @ torch.diag(torch.sqrt(S)) @ V.t()

    C_sigma = 1 / 2 * A_sqrt @ D_sigma @ A_sqrt_inv - sigma ** 2 / 2 * I

    mean = torch.cat((mu_1, mu_2))
    cov = torch.cat((torch.cat((Sigma_1, C_sigma), dim=1), torch.cat((C_sigma.t(), Sigma_2), dim=1)), dim=0)

    dist = torch.distributions.MultivariateNormal(mean, cov)
    # pi(x, y) = e^dist.log_prob((x, y)) for all (x, y) in R^d x R^d
    # x : (n1, d), y : (n2, d), pi : (n1, n2)
    # 
    def pi(x, y):
        if len(x.shape) == 1:
            x = x[None, :]
        if len(y.shape) == 1:
            y = y[None, :]
        
        n1, d1 = x.shape
        n2, d2 = y.shape
        x_expanded = x.unsqueeze(1).expand(n1, n2, d1)  # (n1, n2, d1)
        y_expanded = y.unsqueeze(0).expand(n1, n2, d2)  # (n1, n2, d2)
        joint = torch.cat((x_expanded, y_expanded), dim=2)  # (n1, n2, d1 + d2)
        return torch.exp(dist.log_prob(joint))
    
    return pi

def closed_form_gaussian_T(
    a, # mean of the source Gaussian distribution
    A, # covariance matrix of the source Gaussian distribution
    b, # mean of the target Gaussian distribution
    B, # covariance matrix of the target Gaussian distribution
):
    """
    Return the closed form Monge map between two Gaussian distributions.
    """
    # mu = N(a, A), nu = N(b, B)
    # T = T^AB (x - a) + b
    # T^AB = A^-1/2 (A^1/2 B A^1/2)^1/2 A^-1/2

    U, S, V = torch.svd(A)
    A_sqrt = U @ torch.diag(torch.sqrt(S)) @ V.t()
    A_sqrt_inv = U @ torch.diag(1 / torch.sqrt(S)) @ V.t()

    T = A_sqrt @ B @ A_sqrt
    U, S, V = torch.svd(T)
    T = U @ torch.diag(torch.sqrt(S)) @ V.t()
    T = A_sqrt_inv @ T @ A_sqrt_inv

    map_fn = lambda x : ((x - a) @ T.t()) + b

    return map_fn