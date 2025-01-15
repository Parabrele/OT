import torch

def F_eps(x, epsilon, entropy):
    if entropy:
        return - epsilon * torch.exp(x / epsilon)
    else:
        return - 1 / (4 * epsilon) * torch.relu(x) ** 2

def H(x): # Entropy function
    return - torch.sum(x * torch.log(x + 1e-10))

def H_eps(x, epsilon, entropy):
    if entropy:
        return torch.exp(x / epsilon)
    else:
        return 1 / (2 * epsilon) * torch.relu(x)

class DEFAULT_DUAL_MLP(torch.nn.Module):
    """
    Default neural network architecture for the dual potentials in the stochastic dual algorithm. u, v : R^d -> R
    """
    def __init__(self, d, hidden_scale=10, final_activation='relu'): # TODO : don't forget to test with sigmoid
        super().__init__()
        hidden_size = hidden_scale * d
        self.layer1 = torch.nn.Linear(d, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 1)
        # if final_activation == 'relu':
        #     self.final_act = torch.relu
        #     self.final_layer = torch.nn.Identity()
        # elif final_activation == 'sigmoid':
        #     self.final_act = torch.sigmoid
        #     self.final_layer = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x # torch.relu(self.final_layer(self.final_act(x))) # Ensure dual potentials are in a valid range

class DEFAULT_MAP_MLP(torch.nn.Module):
    """
    Default neural network architecture for the mapping function in the barycentric projection algorithm. f_theta : R^d -> R^d
    """
    def __init__(self, d, hidden_scale=10):
        super().__init__()
        hidden_size = hidden_scale * d
        self.layer1 = torch.nn.Linear(d, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, d)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def __discrete_checks(X1, X2, mu, nu, device):
    # Checks for shapes :
    n1, d = X1.shape
    n2, _d = X2.shape
    if d != _d:
        raise ValueError(f"X1 and X2 should have the same number of features, got {d} and {_d}")
    if mu is None:
        mu = torch.ones(n1) / n1
    if nu is None:
        nu = torch.ones(n2) / n2
    if callable(mu):
        raise ValueError("mu should be a tensor, not a callable")
    if callable(nu):
        raise ValueError("nu should be a tensor, not a callable")
    _n1 = mu.shape[0]
    _n2 = nu.shape[0]
    if _n1 != n1:
        raise ValueError(f"mu should have shape ({n1},), got ({_n1},)")
    if _n2 != n2:
        raise ValueError(f"nu should have shape ({n2},), got ({_n2},)")
    
    # Check for devices :
    if device is None:
        # all should be on the same device
        device = X1.device
        if X2.device != device:
            raise ValueError("X1 and X2 should be on the same device")
        if mu.device != device:
            raise ValueError("mu should be on the same device as X1")
        if nu.device != device:
            raise ValueError("nu should be on the same device as X1")
    else:
        # move all to the device
        X1 = X1.to(device)
        X2 = X2.to(device)
        mu = mu.to(device)
        nu = nu.to(device)

    return X1, X2, mu, nu, device
