import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CouplingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, init_zero=True):

        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.n1 = input_dim // 2
        self.n2 = input_dim - self.n1

        # Scale network
        self.scale_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2),
            nn.Tanh()
        )
        # Translation network
        self.translate_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2)
        )

        if init_zero:
            # Zero initialization on the final layers
            final_scale = self.scale_net[-2]
            nn.init.zeros_(final_scale.weight)
            nn.init.zeros_(final_scale.bias)
            final_trans = self.translate_net[-1]
            nn.init.zeros_(final_trans.weight)
            nn.init.zeros_(final_trans.bias)

    def forward(self, x, reverse=False):

        x1 = x[:, :self.n1]
        x2 = x[:, self.n1:]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        if not reverse:
            # Forward transformation
            x2 = x2 * torch.exp(s) + t
        else:
            # Reverse transformation
            x2 = (x2 - t) * torch.exp(-s)

        x = torch.cat([x1, x2], dim=1)
        log_det = s.sum(dim=1)

        return x, log_det


class Permute(nn.Module):
# Permutation/inverse permutation of the input features

    def __init__(self, num_features):
        super(Permute, self).__init__()

        permutation = torch.randperm(num_features)
        self.register_buffer('permutation', permutation)

        inv_perm = torch.empty_like(permutation)
        inv_perm[self.permutation] = torch.arange(num_features)
        self.register_buffer('inv_permutation', inv_perm)

    def forward(self, x, reverse=False):

        if not reverse:
            return x[:, self.permutation]

        else:
            return x[:, self.inv_permutation], 0.0


class NormalizingFlow(nn.Module):
# RealNVP normalizing flow

    def __init__(self, input_dim, hidden_dim, n_layers, init_zero=True):

        super(NormalizingFlow, self).__init__()
        layers = []

        for _ in range(n_layers):
            layers.append(CouplingLayer(input_dim, hidden_dim, init_zero=init_zero))
            layers.append(Permute(input_dim))

        self.layers = nn.ModuleList(layers)

        # Define the base distribution as a standard multivariate normal
        self.base_dist = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))
        
    def forward(self, x):
    # Forward pass:  data space -> latent space

        log_det_jacobian = 0

        for layer in self.layers:
            if isinstance(layer, CouplingLayer):
                x, log_det = layer(x, reverse=False)
                log_det_jacobian += log_det
            else:
                x = layer(x, reverse=False)

        return x, log_det_jacobian
    
    def inverse(self, z):
    # Inverse pass:  latent space -> data space

        for layer in reversed(self.layers):
            if isinstance(layer, CouplingLayer):
                z, _ = layer(z, reverse=True)
            else:
                z, _ = layer(z, reverse=True)

        return z

    def log_prob(self, x):
        
        z, log_det = self.forward(x)

        return self.base_dist.log_prob(z) + log_det
