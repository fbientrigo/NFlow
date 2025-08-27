# training.py

"""
Training utilities for RealNVP flows with modular, composable losses.

What is new here
----------------
- Loss registry includes: N (NLL), G (Guided), C (Corr), J (Jacobian regularizer).
- `build_combined_loss` returns a composite with attributes: `.mods`, `.w`, `.ids`.
- **Per‑component loss breakdown**: helper `compute_loss_and_parts(...)` to get
  total + parts (e.g., {'N': ..., 'G': ..., 'C': ..., 'J': ...}).
- Training loop `train_model_modular(...)` now logs **per‑loss curves** to TB and
  also returns (optionally) a `history` dict with epochwise totals & parts.
- Lightweight Jacobian‐of‐flow stats (mean/std) are logged each epoch.
- Utilities to collect / plot latent metrics remain compatible.

This file is Colab‑friendly: no extra dependencies beyond PyTorch/Matplotlib.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple, Callable, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # optional

logger = logging.getLogger(__name__)

# ----------------------------
# Loss Interfaces & Registry
# ----------------------------
class LossBase(nn.Module):
    """All losses implement forward(batch: dict, model: nn.Module, epoch: int, epochs_tot: int) -> Tensor
    batch MUST contain at least: {'x': Tensor[B, D]}. Optionally: 'mask' (Tensor[B]).
    """
    def forward(self, batch: Dict[str, torch.Tensor], model: nn.Module, epoch: int, epochs_tot: int) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError

class NLLLoss(LossBase):
    """Standard negative log-likelihood: -E[log p(x)]."""
    def forward(self, batch: Dict[str, torch.Tensor], model: nn.Module, epoch: int, epochs_tot: int) -> torch.Tensor:
        x = batch['x']
        return -model.log_prob(x).mean()

@dataclass
class GuidedCfg:
    sigma1: float
    sigma2: float
    lambda1: float
    lambda2: Optional[float] = None
    lambda2_schedule: Optional[Callable[[int, int], float]] = None

class GuidedLoss(LossBase):
    """Guided loss (N/G style):
      total = NLL_global + lambda1 * E_{~mask}(l1) + lambda2 * E_{mask}(l2)
      where l1 = -log N(0, sigma1^2 I)(z), l2 = -log N(0, sigma2^2 I)(z)
      with z, log_det = model.forward(x)
    """
    def __init__(self, cfg: GuidedCfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, torch.Tensor], model: nn.Module, epoch: int, epochs_tot: int) -> torch.Tensor:
        x = batch['x']
        mask = batch.get('mask', None)
        # Global NLL over the whole batch
        loss_global = -model.log_prob(x).mean()

        # z for guided terms
        z, _ = model.forward(x)
        # log N(z; 0, sigma^2 I) = sum_i [ -0.5*log(2πσ^2) - z_i^2/(2σ^2) ]
        def neg_log_gauss(z, sigma):
            return -torch.distributions.Normal(0.0, sigma).log_prob(z).sum(dim=1)

        l1 = neg_log_gauss(z, self.cfg.sigma1)
        l2 = neg_log_gauss(z, self.cfg.sigma2)

        if mask is None:
            logger.warning("GuidedLoss: mask not provided; guided terms reduce to global")
            return loss_global

        mask = mask.bool()
        not_mask = ~mask
        if mask.sum() == 0 or not_mask.sum() == 0:
            logger.warning("GuidedLoss: level-2 subset empty at epoch %d", epoch)

        loss_not2 = l1[not_mask].mean() if not_mask.any() else z.new_tensor(0.0)
        loss_2    = l2[mask].mean()     if mask.any()     else z.new_tensor(0.0)

        lambda2 = (self.cfg.lambda2_schedule(epoch, epochs_tot)
                   if self.cfg.lambda2_schedule is not None
                   else (self.cfg.lambda2 if self.cfg.lambda2 is not None else 0.0))

        return loss_global + self.cfg.lambda1 * loss_not2 + lambda2 * loss_2

class CorrLoss(LossBase):
    """Correlation-structure matching between model samples and data.
    Penalizes mean squared difference between correlation matrices.
    """
    def __init__(self, n_samples: Optional[int] = None):
        super().__init__()
        self.n_samples = n_samples

    def _corr(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.std(dim=0, keepdim=True) + 1e-8)
        return (x.T @ x) / (x.shape[0] - 1)

    def forward(self, batch: Dict[str, torch.Tensor], model: nn.Module, epoch: int, epochs_tot: int) -> torch.Tensor:
        x = batch['x']
        n = self.n_samples or x.shape[0]
        with torch.no_grad():
            xs = model.sample(n)
        c_pred = self._corr(xs)
        c_true = self._corr(x)
        return torch.mean((c_pred - c_true) ** 2)

class JacRegLoss(LossBase):
    """Penaliza magnitud/varianza del log-det del flow (excluye el bijector)."""
    def __init__(self, alpha: float = 1e-3, beta: float = 1e-3):
        super().__init__()
        self.alpha = float(alpha)
        self.beta  = float(beta)

    def forward(self, batch, model, epoch, epochs_tot):
        x = batch['x']
        z, tot_ld = model.forward(x)  # incluye bijector + flow
        # sustrae LD del bijector (mismo eps que el modelo)
        eps = getattr(model, "epsilon", 1e-6)
        x_cl = x.clamp(0, 1)
        xs = x_cl * (1 - 2*eps) + eps
        ld_bij = ( torch.log(torch.tensor(1 - 2*eps, device=x.device))
                   - torch.log(xs) - torch.log1p(-xs) ).sum(dim=1)
        ld_flow = tot_ld - ld_bij
        return self.alpha * ld_flow.pow(2).mean() + self.beta * ld_flow.var(unbiased=False)

# Registry of available losses (extend here for more A/B variants)
LOSS_REGISTRY: Dict[str, Callable[..., LossBase]] = {
    'N': lambda **kw: NLLLoss(),
    'G': lambda **kw: GuidedLoss(GuidedCfg(**kw)),
    'C': lambda **kw: CorrLoss(**kw),
    'J': lambda **kw: JacRegLoss(**kw)
}


def build_combined_loss(keys: str,
                        weights: Optional[Dict[str, float]] = None,
                        per_loss_cfg: Optional[Dict[str, dict]] = None) -> LossBase:
    """Build a combined loss from a string like 'NGC'.

    Returns a module with attributes `.mods` (sub‑losses), `.w` (weights), `.ids` (keys).
    """
    per_loss_cfg = per_loss_cfg or {}
    ks = list(keys)
    if weights is None:
        weights = {k: 1.0 / max(1, len(ks)) for k in ks}

    modules = nn.ModuleList()
    ws: List[float] = []
    ids: List[str] = []
    for k in ks:
        if k not in LOSS_REGISTRY:
            raise KeyError(f"Unknown loss key '{k}'. Available: {list(LOSS_REGISTRY)}")
        cfg = per_loss_cfg.get(k, {})
        lf = LOSS_REGISTRY[k](**cfg)
        modules.append(lf)
        ws.append(float(weights[k]))
        ids.append(k)

    class _Combined(LossBase):
        def __init__(self, mods: nn.ModuleList, w: List[float], ids: List[str]):
            super().__init__()
            self.mods = mods
            self.w = w
            self.ids = ids
        def forward(self, batch: Dict[str, torch.Tensor], model: nn.Module, epoch: int, epochs_tot: int) -> torch.Tensor:
            total = 0.0
            for w, lf in zip(self.w, self.mods):
                total = total + w * lf(batch, model, epoch, epochs_tot)
            return total

    return _Combined(modules, ws, ids)

# ----------------------------
# Helpers: loss breakdown & LD stats
# ----------------------------

def compute_loss_and_parts(loss_fn: LossBase,
                           batch: Dict[str, torch.Tensor],
                           model: nn.Module,
                           epoch: int,
                           epochs_tot: int):
    """Compute total loss and per‑component parts when available.
    Returns (total_loss, parts_dict) where parts are **unweighted**: key -> Tensor.
    Falls back to {} if `loss_fn` has no composite structure.
    """
    if hasattr(loss_fn, 'mods') and hasattr(loss_fn, 'w') and hasattr(loss_fn, 'ids'):
        parts = {}
        total = 0.0
        for k, w, lf in zip(loss_fn.ids, loss_fn.w, loss_fn.mods):
            li = lf(batch, model, epoch, epochs_tot)
            parts[k] = li.detach()
            total = total + w * li
        return total, parts
    # simple loss
    total = loss_fn(batch, model, epoch, epochs_tot)
    return total, {}


def compute_ld_flow_stats(model: nn.Module, x: torch.Tensor, eps: float | None = None):
    """Return mean/std of log‑det contributed by RealNVP layers (excludes bijector)."""
    with torch.no_grad():
        z, tot_ld = model.forward(x)
        e = eps if eps is not None else getattr(model, "epsilon", 1e-6)
        x_cl = x.clamp(0, 1)
        xs = x_cl * (1 - 2*e) + e
        ld_bij = ( torch.log(torch.tensor(1 - 2*e, device=x.device))
                   - torch.log(xs) - torch.log1p(-xs) ).sum(dim=1)
        ld_flow = tot_ld - ld_bij
        return ld_flow.mean().item(), ld_flow.std().item()

# ----------------------------
# Training / Evaluation Loops
# ----------------------------

def _to_batch(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    return {'x': x} if mask is None else {'x': x, 'mask': mask}


def train_model_modular(model: nn.Module,
                        train_loader: Iterable,
                        val_loader: Iterable,
                        epochs: int,
                        lr: float,
                        writer,  # may be None
                        device: torch.device,
                        model_dir: str,
                        name_model: str,
                        loss_keys: str = 'N',
                        loss_weights: Optional[Dict[str, float]] = None,
                        loss_cfg: Optional[Dict[str, dict]] = None,
                        patience: int = 20,
                        weight_decay: float = 0.0,
                        trial=None,
                        plot_every: int = 0,
                        return_history: bool = False):
    """Modular training with weighted, pluggable losses.

    If `return_history=True`, returns `(model, history)` where `history` is a dict:
      history = {
        'train_total': [...], 'val_total': [...],
        'train_parts': {key: [...]}, 'val_parts': {key: [...]},
        'ld_flow_mean': [...], 'ld_flow_std': [...],
      }
    """
    loss_fn = build_combined_loss(loss_keys, loss_weights, loss_cfg)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_val = float('inf')
    epochs_no_improve = 0

    hist = {
        'train_total': [], 'val_total': [],
        'train_parts': defaultdict(list), 'val_parts': defaultdict(list),
        'ld_flow_mean': [], 'ld_flow_std': []
    }

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tr_sum = 0.0
        tr_parts_acc = defaultdict(float)
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                x, mask = batch
                b = _to_batch(x.to(device), mask.to(device))
            elif isinstance(batch, dict):
                b = {k: v.to(device) for k, v in batch.items()}
            else:
                b = {'x': batch.to(device)}

            optimizer.zero_grad()
            loss, parts = compute_loss_and_parts(loss_fn, b, model, epoch, epochs)
            loss.backward()
            optimizer.step()

            tr_sum += float(loss)
            for k, v in parts.items():
                tr_parts_acc[k] += float(v)

        ntr = max(1, len(train_loader))
        train_loss = tr_sum / ntr
        hist['train_total'].append(train_loss)
        for k in getattr(loss_fn, 'ids', []):
            hist['train_parts'][k].append(tr_parts_acc.get(k, 0.0) / ntr)

        # ---- Validation ----
        model.eval()
        vl_sum = 0.0
        vl_parts_acc = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, mask = batch
                    b = _to_batch(x.to(device), mask.to(device))
                elif isinstance(batch, dict):
                    b = {k: v.to(device) for k, v in batch.items()}
                else:
                    b = {'x': batch.to(device)}
                vloss, vparts = compute_loss_and_parts(loss_fn, b, model, epoch, epochs)
                vl_sum += float(vloss)
                for k, v in vparts.items():
                    vl_parts_acc[k] += float(v)
        nvl = max(1, len(val_loader))
        val_loss = vl_sum / nvl
        hist['val_total'].append(val_loss)
        for k in getattr(loss_fn, 'ids', []):
            hist['val_parts'][k].append(vl_parts_acc.get(k, 0.0) / nvl)

        # ---- Jacobian stats on a small validation batch ----
        try:
            xb = next(iter(val_loader))
            x = xb[0].to(device) if isinstance(xb,(list,tuple)) else (xb['x'].to(device) if isinstance(xb,dict) else xb.to(device))
            mu_ld, sd_ld = compute_ld_flow_stats(model, x)
            hist['ld_flow_mean'].append(mu_ld)
            hist['ld_flow_std'].append(sd_ld)
        except Exception:
            hist['ld_flow_mean'].append(float('nan'))
            hist['ld_flow_std'].append(float('nan'))

        logger.info(f"[{epoch}/{epochs}] Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        if writer is not None:
            writer.add_scalars("Loss/Total", {"Train": train_loss, "Val": val_loss}, epoch)
            for k in getattr(loss_fn, 'ids', []):
                writer.add_scalars(f"Loss/{k}", {"Train": hist['train_parts'][k][-1], "Val": hist['val_parts'][k][-1]}, epoch)
            writer.add_scalars("Jacobian/ld_flow", {"mean": hist['ld_flow_mean'][-1], "std": hist['ld_flow_std'][-1]}, epoch)

        # Optuna pruning
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping & checkpoint
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_dir}/{name_model}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (best=%.4f)", epoch, best_val)
                break

        # Optional inline plot in notebooks
        if plot_every and (epoch % plot_every == 0):
            try:
                plot_loss_history(hist, ids=getattr(loss_fn, 'ids', []))
            except Exception:
                pass

    # Load best
    model.load_state_dict(torch.load(f"{model_dir}/{name_model}.pt", map_location=device))
    logger.info("Training complete. Best validation loss: %.4f", best_val)
    if return_history:
        return model, hist
    return model

# ----------------------------
# Plot helpers
# ----------------------------

def plot_loss_history(history: Dict, ids: Iterable[str] = ()):  # simple, notebook-friendly
    """Plot total Train/Val and each component's Train/Val curves."""
    T = len(history['train_total'])
    xs = list(range(1, T+1))
    n_parts = len(list(ids))
    rows = 1 + (1 if n_parts else 0) + (1)  # total + (parts) + (ld stats)
    fig, axes = plt.subplots(rows, 1, figsize=(7, 3*rows), sharex=True)
    if rows == 1:
        axes = [axes]

    # Total
    ax = axes[0]
    ax.plot(xs, history['train_total'], label='Train')
    ax.plot(xs, history['val_total'],   label='Val')
    ax.set_ylabel('Total loss'); ax.set_title('Total Loss'); ax.grid(True, alpha=0.3); ax.legend()

    # Parts
    if n_parts:
        ax = axes[1]
        for k in ids:
            ax.plot(xs, history['train_parts'][k], label=f'Train {k}')
            ax.plot(xs, history['val_parts'][k],   label=f'Val {k}', linestyle='--')
        ax.set_ylabel('Loss parts'); ax.set_title('Per‑component losses'); ax.grid(True, alpha=0.3); ax.legend(ncol=max(1, n_parts))

    # Jacobian stats
    ax = axes[-1]
    ax.plot(xs, history['ld_flow_mean'], label='ld_flow mean')
    ax.plot(xs, history['ld_flow_std'],  label='ld_flow std')
    ax.set_ylabel('LD stats'); ax.set_xlabel('Epoch'); ax.set_title('RealNVP log‑det (flow only)'); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout(); plt.show()

# ----------------------------
# Latent helpers (unchanged API)
# ----------------------------

def collect_latents(model, loader, device):
    """Collect latent representations and masks from a data loader.
    Returns (zs, ys)."""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch['x'], batch.get('mask', None)
                if y is None:
                    raise ValueError('loader dict must include a "mask" key for guided metrics')
            else:
                raise ValueError('loader must yield (x, mask) or dict with keys x/mask')
            x = x.to(device)
            z, _ = model.forward(x)
            zs.append(z.cpu())
            ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def plot_latent_dims(zs, ys, prior):
    """Plot marginal histograms of each latent dimension, with component means."""
    if isinstance(zs, torch.Tensor):
        zs = zs.cpu().detach().numpy()
    if isinstance(ys, torch.Tensor):
        ys = ys.cpu().detach().numpy()
    ys = ys.astype(bool)

    N, D = zs.shape
    fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
    if D == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(zs[~ys, i], bins=30, alpha=0.5, density=True, label='lvl1', color='C0')
        ax.hist(zs[ ys, i], bins=30, alpha=0.5, density=True, label='lvl2', color='C1')
        for k in range(prior.K):
            mu_ki = prior.mu[k, i].item()
            ax.axvline(mu_ki, linestyle='--', color=f'C{k}', label=f'μ_{k}, dim{i}')
        ax.set_xlabel(f'z[{i}]'); ax.set_ylabel('Density'); ax.set_title(f'Latent Dimension #{i}')
        ax.legend(fontsize='small')
    plt.tight_layout(); plt.show()


def latent_metrics(zs, ys, prior):
    """Compute simple latent metrics (silhouette; Mahalanobis for K=2)."""
    try:

        HAVE_SK = True
    except Exception:
        HAVE_SK = False

    if isinstance(ys, torch.Tensor):
        labels = ys.numpy()
    else:
        labels = ys
    if isinstance(zs, torch.Tensor):
        data = zs.numpy()
    else:
        data = zs

    sil = None
    if HAVE_SK:
        try:
            sil = float(__import__('sklearn.metrics').metrics.silhouette_score(data, labels))
        except Exception:
            sil = None

    maha = None
    if getattr(prior, 'K', 0) == 2:
        diff = (prior.mu[0] - prior.mu[1]).unsqueeze(0)
        cov = (prior.log_sig.exp()[0]**2 + prior.log_sig.exp()[1]**2).mean()
        maha = (diff.norm() / math.sqrt(cov)).item()

    return sil, maha
