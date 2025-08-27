"""
flow_models.py

RealNVP normalizing flow layers (no bounded-data bijector) and an optional
Gaussian Mixture prior for guided use-cases.

Changes vs previous variant:
- Removed LogitTransform and any `bounded_data` paths to keep the model tidy.
- Homogeneous layer interface: `forward(x, reverse=False) -> (y, log_det)`
  and same signature in reverse.
- `NormalizingFlow` manages the base distribution on the correct device via
  `ensure_base_dist` and uses tensor-shaped log-det accumulators.
- Colab-friendly, no extra dependencies.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# RealNVP blocks
# ---------------------------------------------------------------------------
class CouplingLayer(nn.Module):
    """Affine coupling (RealNVP).

    Splits input into (x1, x2), applies x2' = x2 * exp(s(x1)) + t(x1).
    Returns `(x_out, log_det)` in both directions.
    """
    def __init__(self, input_dim: int, hidden_dim: int, init_zero: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.n1 = input_dim // 2
        self.n2 = input_dim - self.n1

        self.scale_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2),
            nn.Tanh(),  # bounds log-scale for early stability
        )
        self.translate_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2),
        )
        if init_zero:
            nn.init.zeros_(self.scale_net[-2].weight)
            nn.init.zeros_(self.scale_net[-2].bias)
            nn.init.zeros_(self.translate_net[-1].weight)
            nn.init.zeros_(self.translate_net[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        x1, x2 = x[:, :self.n1], x[:, self.n1:]
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        if not reverse:
            x2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
        else:
            x2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
        x_out = torch.cat([x1, x2], dim=1)
        return x_out, log_det


class Permute(nn.Module):
    """Feature permutation; returns `(x, 0)` in both directions."""
    def __init__(self, num_features: int):
        super().__init__()
        perm = torch.randperm(num_features)
        self.register_buffer("perm", perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(num_features)
        self.register_buffer("inv", inv)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        out = x[:, self.inv] if reverse else x[:, self.perm]
        return out, torch.zeros(x.size(0), device=x.device)


# ---------------------------------------------------------------------------
# Flow (no bijector)
# ---------------------------------------------------------------------------
@dataclass
class FlowConfig:
    input_dim: int
    hidden_dim: int
    n_layers: int
    init_zero: bool = True


class NormalizingFlow(nn.Module):
    """Plain RealNVP (no bounded-data wrapper).

    Args:
        input_dim: data dimensionality
        hidden_dim: hidden units for s/t MLPs
        n_layers: number of coupling+permute blocks
        init_zero: near-identity start for stability
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 init_zero: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)

        layers = []
        for _ in range(n_layers):
            layers.append(CouplingLayer(input_dim, hidden_dim, init_zero=init_zero))
            layers.append(Permute(input_dim))
        self.layers = nn.ModuleList(layers)

        self._base_dist = None  # lazily created on the right device

    # ---- base distribution management ----
    def ensure_base_dist(self, device: torch.device):
        if (self._base_dist is None) or (self._base_dist.loc.device != device) or (self._base_dist.loc.numel() != self.input_dim):
            self._base_dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.input_dim, device=device),
                covariance_matrix=torch.eye(self.input_dim, device=device),
            )

    # ---- forward/inverse ----
    def forward(self, x: torch.Tensor):
        """Data → latent: returns (z, total_log_det)."""
        B = x.size(0)
        total_ld = torch.zeros(B, device=x.device)
        for layer in self.layers:
            x, ld = layer(x, reverse=False)
            total_ld = total_ld + ld
        return x, total_ld

    def inverse(self, z: torch.Tensor):
        """Latent → data."""
        for layer in reversed(self.layers):
            z, _ = layer(z, reverse=True)
        return z

    # ---- densities & sampling ----
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, total_ld = self.forward(x)
        self.ensure_base_dist(z.device)
        return self._base_dist.log_prob(z) + total_ld

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None, temperature: float = 1.0) -> torch.Tensor:
        device = device or next(self.parameters()).device
        self.ensure_base_dist(device)
        z = self._base_dist.sample((n,))
        if temperature != 1.0:
            z = z * float(temperature)
        return self.inverse(z)


# ---------------------------------------------------------------------------
# Optional GMM prior (for guided uses)
# ---------------------------------------------------------------------------
class MixturePrior(nn.Module):
    """Diagonal-covariance Gaussian mixture in latent space."""
    def __init__(self, latent_dim: int, K: int = 2):
        super().__init__()
        self.K = K
        self.mu = nn.Parameter(torch.zeros(K, latent_dim))
        self.log_sig = nn.Parameter(torch.zeros(K, latent_dim))
        self.log_pi = nn.Parameter(torch.full((K,), -math.log(K)))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        zs = z.unsqueeze(1)                    # (B, 1, D)
        mu = self.mu.unsqueeze(0)              # (1, K, D)
        sig = self.log_sig.exp().unsqueeze(0)  # (1, K, D)
        log_comp = -0.5 * (((zs - mu) / sig) ** 2 + 2 * self.log_sig + math.log(2 * math.pi)).sum(-1)
        mix_log = torch.log_softmax(self.log_pi, dim=0) + log_comp
        return torch.logsumexp(mix_log, dim=1)

    def sample(self, N: int, alpha: float | None = None, device: torch.device | None = None):
        device = device or self.mu.device
        if alpha is None or self.K != 2:
            w = torch.softmax(self.log_pi, dim=0).to(device)
        else:
            w = torch.tensor([1 - alpha, alpha], device=device)
        cat = torch.distributions.Categorical(w)
        comps = cat.sample((N,)).to(device)
        eps = torch.randn(N, self.mu.size(1), device=device)
        sig = self.log_sig.exp()
        z = self.mu[comps] + sig[comps] * eps
        return z, comps

    @property
    def pi(self) -> torch.Tensor:
        return torch.softmax(self.log_pi, dim=0)
