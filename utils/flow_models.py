"""
flow_models.py

RealNVP normalizing flow layers with an optional bounded-data bijector
and a learnable Gaussian Mixture prior.

Changes vs. previous version:
- NEW `LogitTransform` bijector to map data in [0,1]^D <-> R^D (with epsilon).
- `NormalizingFlow` can enable the bijector via `bounded_data=True` and `epsilon`.
  * Ensures samples from `inverse()` live in [0,1] (up to ±eps) and fixes support mismatch.
- Homogeneous layer interfaces: all layers return `(x, log_det)` in both directions.
- `NormalizingFlow` now manages a cached `base_dist` on the correct device via
  `ensure_base_dist(device)` and uses tensor-shaped `log_det` accumulators.

This file is self‑contained to avoid extra imports, but `LogitTransform` can be
moved to `utils/bijectors.py` later with no other code changes.
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
# Bijectors
# ---------------------------------------------------------------------------
class LogitTransform(nn.Module):
    """Bijector for data bounded in [0,1]^D.

    forward(x):   x∈[0,1] -> y∈R, returns (y, sum_log_det)
    inverse(y):   y∈R     -> x∈[0,1], returns (x, zeros)

    We scale x to (ε, 1-ε) before the logit; the Jacobian per dim is
      dy/dx = (1-2ε) / (x_s * (1 - x_s))
    so log|det J| = Σ [ log(1-2ε) - log(x_s) - log(1 - x_s) ].
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        eps = self.eps
        B = x.size(0)
        if not reverse:
            x = torch.clamp(x, 0.0, 1.0)
            x_s = x * (1.0 - 2.0 * eps) + eps  # (ε, 1-ε)
            y = torch.log(x_s) - torch.log1p(-x_s)  # logit
            # log|det J|
            sum_log_det = (
                torch.log(torch.tensor(1.0 - 2.0 * eps, device=x.device))
                - torch.log(x_s)
                - torch.log1p(-x_s)
            ).sum(dim=1)
            return y, sum_log_det
        else:
            s = torch.sigmoid(x)                 # (0,1)
            x = (s - eps) / (1.0 - 2.0 * eps)    # back to [0,1]
            x = torch.clamp(x, min=self.eps, max=1.0 - self.eps) # <---- clamping
            return x, torch.zeros(B, device=x.device)


# ---------------------------------------------------------------------------
# RealNVP blocks
# ---------------------------------------------------------------------------
class CouplingLayer(nn.Module):
    """Affine coupling (RealNVP).

    Splits input into (x1, x2), applies x2' = x2 * exp(s(x1)) + t(x1),
    and returns `(x_out, log_det)` in both directions.
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
            nn.Tanh(),  # bound log-scale to avoid explosion early on
        )
        self.translate_net = nn.Sequential(
            nn.Linear(self.n1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n2),
        )
        if init_zero:
            # last Linear before Tanh in scale_net is index -2
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
    """Feature permutation; returns (x, 0) in both directions."""
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
# Flow + optional bounded-data wrapper
# ---------------------------------------------------------------------------
@dataclass
class FlowConfig:
    input_dim: int
    hidden_dim: int
    n_layers: int
    init_zero: bool = True
    bounded_data: bool = True
    epsilon: float = 1e-6


class NormalizingFlow(nn.Module):
    """RealNVP flow with optional [0,1]^D support via `LogitTransform`.

    Args:
        input_dim: data dimensionality
        hidden_dim: hidden units for s/t MLPs
        n_layers: number of coupling-permute blocks
        init_zero: initialize last layers to zero (near-identity start)
        bounded_data: if True, prepend/append Logit/Sigmoid bijector
        epsilon: ε used by the logit bijector
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 init_zero: bool = True, bounded_data: bool = True, epsilon: float = 1e-6):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bounded_data = bool(bounded_data)
        self.epsilon = float(epsilon)

        self.pre_bijector = LogitTransform(self.epsilon) if self.bounded_data else None

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
        # bounded-data pre-bijector
        if self.pre_bijector is not None:
            x, ld = self.pre_bijector(x, reverse=False)
            total_ld = total_ld + ld
        # flow
        for layer in self.layers:
            x, ld = layer(x, reverse=False)
            total_ld = total_ld + ld
        return x, total_ld

    def inverse(self, z: torch.Tensor):
        """Latent → data: applies inverse flow and optional sigmoid."""
        for layer in reversed(self.layers):
            z, _ = layer(z, reverse=True)
        if self.pre_bijector is not None:
            z, _ = self.pre_bijector(z, reverse=True)  # into [0,1]
        return z

    # ---- densities & sampling ----
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, total_ld = self.forward(x)
        self.ensure_base_dist(z.device)
        return self._base_dist.log_prob(z) + total_ld

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        self.ensure_base_dist(device)
        z = self._base_dist.sample((n,))
        return self.inverse(z)


# ---------------------------------------------------------------------------
# Learnable GMM prior (for guided or hybrid losses)
# ---------------------------------------------------------------------------
class MixturePrior(nn.Module):
    """Diagonal-covariance mixture of Gaussians in latent space."""
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
