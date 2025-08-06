"""
flow_models.py

RealNVP normalizing flow layers and a learnable Gaussian Mixture prior.

This module provides:
- CouplingLayer: affine coupling transformation for RealNVP.
- Permute: random feature permutation layer.
- NormalizingFlow: sequence of coupling and permutation layers.
- MixturePrior: learnable mixture-of-Gaussians prior with sampling and log-probability.
"""

import logging
import math
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
torch.manual_seed(42)


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP flows.

    Splits input into two halves (x1, x2) and applies:
      x2' = x2 * exp(s(x1)) + t(x1)
    in forward mode, and the inverse in reverse mode.

    Parameters
    ----------
    input_dim : int
        Total dimensionality of the input vector.
    hidden_dim : int
        Size of the hidden layer in the scale and translate networks.
    init_zero : bool, default=True
        If True, initialize the last linear layers' weights and biases to zero
        so that the coupling starts as an identity transform.

    Attributes
    ----------
    n1 : int
        Dimensionality of x1 (first half).
    n2 : int
        Dimensionality of x2 (second half).
    scale_net : nn.Sequential
        Network producing the log-scale s(x1).
    translate_net : nn.Sequential
        Network producing the translation t(x1).
    """
    def __init__(self, input_dim, hidden_dim, init_zero=True):
        super().__init__()
        self.input_dim = input_dim
        self.n1 = input_dim // 2
        self.n2 = input_dim - self.n1

        # Scale network: outputs log-scale s
        self.scale_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2),
            nn.Tanh()
        )
        # Translation network: outputs t
        self.translate_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2)
        )

        if init_zero:
            # Zero-init last layers to start near identity
            nn.init.zeros_(self.scale_net[-2].weight)
            nn.init.zeros_(self.scale_net[-2].bias)
            nn.init.zeros_(self.translate_net[-1].weight)
            nn.init.zeros_(self.translate_net[-1].bias)

    def forward(self, x, reverse=False):
        """
        Apply the coupling transformation or its inverse.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input tensor.
        reverse : bool, default=False
            If False, apply forward transform; if True, apply inverse.

        Returns
        -------
        x_out : torch.Tensor of shape (batch_size, input_dim)
            Transformed output.
        log_det : torch.Tensor of shape (batch_size,)
            Log-determinant of the Jacobian for this layer.
        """
        x1, x2 = x[:, :self.n1], x[:, self.n1:]
        s = self.scale_net(x1)
        t = self.translate_net(x1)

        if not reverse:
            x2 = x2 * torch.exp(s) + t
        else:
            x2 = (x2 - t) * torch.exp(-s)

        x_out = torch.cat([x1, x2], dim=1)
        log_det = s.sum(dim=1)
        return x_out, log_det


class Permute(nn.Module):
    """
    Permutation layer for shuffle features between coupling layers.

    Parameters
    ----------
    num_features : int
        Number of features to permute.

    Attributes
    ----------
    perm : torch.LongTensor
        A random permutation of indices [0, 1, ..., num_features-1].
    inv : torch.LongTensor
        The inverse permutation.
    """
    def __init__(self, num_features):
        super().__init__()
        perm = torch.randperm(num_features)
        self.register_buffer('perm', perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(num_features)
        self.register_buffer('inv', inv)

    def forward(self, x, reverse=False):
        """
        Apply or invert the permutation.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_features)
            Input tensor.
        reverse : bool, default=False
            If False, apply perm; if True, apply inverse perm.

        Returns
        -------
        x_out : torch.Tensor
            Permuted tensor.
        log_det : float
            Always zero for permutation layers.
        """
        if not reverse:
            return x[:, self.perm]
        return x[:, self.inv], torch.tensor(0., device=x.device)


class NormalizingFlow(nn.Module):
    """
    RealNVP normalizing flow composed of alternating CouplingLayer and Permute.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    hidden_dim : int
        Hidden dimension used in coupling layers.
    n_layers : int
        Number of coupling-permute blocks.
    init_zero : bool, default=True
        Passed to each CouplingLayer to initialize identity.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, init_zero=True):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(CouplingLayer(input_dim, hidden_dim, init_zero=init_zero))
            layers.append(Permute(input_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass: data → latent.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input data.

        Returns
        -------
        z : torch.Tensor of shape (batch_size, input_dim)
            Latent codes.
        log_det : torch.Tensor of shape (batch_size,)
            Sum of log-determinants from each coupling layer.
        """
        log_det = 0
        for layer in self.layers:
            if isinstance(layer, CouplingLayer):
                x, ld = layer(x, reverse=False)
                log_det += ld
            else:
                x = layer(x)
        return x, log_det

    def inverse(self, z):
        """
        Inverse pass: latent → data.

        Parameters
        ----------
        z : torch.Tensor of shape (batch_size, input_dim)
            Latent codes.

        Returns
        -------
        x : torch.Tensor of shape (batch_size, input_dim)
            Reconstructed data.
        """
        for layer in reversed(self.layers):
            if isinstance(layer, CouplingLayer):
                z, _ = layer(z, reverse=True)
            else:
                z, _ = layer(z, reverse=True)
        return z

    def log_prob(self, x):
        
        z, log_det = self.forward(x)

        return self.base_dist.log_prob(z) + log_det

class MixturePrior(nn.Module):
    """
    Learnable mixture-of-Gaussians prior distribution in latent space.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of each Gaussian component.
    K : int, default=2
        Number of mixture components.

    Attributes
    ----------
    mu : nn.Parameter of shape (K, latent_dim)
        Learnable means for each component.
    log_sig : nn.Parameter of shape (K, latent_dim)
        Log-standard deviations for each component.
    log_pi : nn.Parameter of shape (K,)
        Unnormalized log-weights for the categorical mixing distribution.
    """
    def __init__(self, latent_dim, K=2):
        super().__init__()
        self.K = K
        self.mu = nn.Parameter(torch.zeros(K, latent_dim))
        self.log_sig = nn.Parameter(torch.zeros(K, latent_dim))
        self.log_pi = nn.Parameter(torch.full((K,), -math.log(K)))

    def log_prob(self, z):
        """
        Compute log-density under the GMM prior.

        Parameters
        ----------
        z : torch.Tensor of shape (batch_size, latent_dim)
            Latent codes.

        Returns
        -------
        logp : torch.Tensor of shape (batch_size,)
            Log-probability of each sample under the mixture.
        """
        B, D = z.shape
        zs = z.unsqueeze(1)                  # (B, 1, D)
        mu = self.mu.unsqueeze(0)            # (1, K, D)
        sig = self.log_sig.exp().unsqueeze(0)# (1, K, D)

        # component log-probabilities
        log_comp = -0.5 * (((zs - mu) / sig)**2 + 2*self.log_sig + math.log(2*math.pi)).sum(-1)
        mix_log = torch.log_softmax(self.log_pi, dim=0) + log_comp
        return torch.logsumexp(mix_log, dim=1)

    def sample(self, N, alpha=None, device=None):
        """
        Draw samples from the mixture prior.

        Parameters
        ----------
        N : int
            Number of samples to generate.
        alpha : float or None, default=None
            If float and K==2, use weights [1-alpha, alpha]; otherwise,
            use learned softmax(self.log_pi).
        device : torch.device, optional
            Device for output tensors; defaults to prior's parameters device.

        Returns
        -------
        z : torch.Tensor of shape (N, latent_dim)
            Sampled latent codes.
        comps : torch.LongTensor of shape (N,)
            Component indices for each sample.
        """
        device = device or self.mu.device
        if alpha is None or self.K != 2:
            w = torch.softmax(self.log_pi, dim=0)
        else:
            w = torch.tensor([1-alpha, alpha], device=device)
        cat = torch.distributions.Categorical(w)
        comps = cat.sample((N,))
        eps = torch.randn(N, self.mu.size(1), device=device)
        sig = self.log_sig.exp()
        z = self.mu[comps] + sig[comps] * eps
        return z, comps

    @property
    def pi(self):
        """
        Mixing weights of the prior as a probability vector.

        Returns
        -------
        pi : torch.Tensor of shape (K,)
            Softmax-normalized mixture weights.
        """
        return torch.softmax(self.log_pi, dim=0)
