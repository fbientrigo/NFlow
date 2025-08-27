"""
utils/metrics.py

Lightweight, NumPy/Matplotlib‑only metrics and plots for evaluating flow models,
plus a consolidated **post‑training smoke report** you can call from the notebook
instead of a long cell.

Expected usage (call sites)
---------------------------
1) Notebook "smoke" diagnostics (after training):
   - Call `post_training_smoke_report(model, val_loader, history=history)` to
     compute A1/A2, show a compact table, and (optionally) plot loss curves via
     the provided `plot_loss_history_fn` (e.g., `utils.training.plot_loss_history`).
   - Use `ks_stat` / `ks_2samp_np` / `wasserstein1d` and `plot_qq` for per‑feature checks.

2) Evaluation cells (A/B tests):
   - Compute KS/W1 per feature and control FDR with `fdr_bh`.

Notes
-----
- Only depends on NumPy, Matplotlib, and (for the smoke report) PyTorch.
- Functions are robust to ties and work with flattened arrays.
"""
from __future__ import annotations

from typing import Optional, Tuple, Sequence, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Torch is only used by the post‑training report
import torch

__all__ = [
    "ks_stat",
    "ks_2samp_np",
    "wasserstein1d",
    "plot_qq",
    "fdr_bh",
    "compute_ld_flow_stats",
    "post_training_smoke_report",
]

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Two‑sample Kolmogorov–Smirnov statistic (no p‑value), NumPy‑only.

    This computes D = sup_t |F_x(t) - F_y(t)| on the pooled grid of both
    samples using right‑continuous ECDFs. Works with ties and different sizes.
    """
    xs = np.sort(np.asarray(x).ravel())
    ys = np.sort(np.asarray(y).ravel())
    grid = np.concatenate([xs, ys])
    Fx = np.searchsorted(xs, grid, side="right") / xs.size
    Fy = np.searchsorted(ys, grid, side="right") / ys.size
    return float(np.max(np.abs(Fx - Fy)))


def ks_2samp_np(x: np.ndarray, y: np.ndarray, B: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
    """Two‑sample KS with permutation p‑value, NumPy‑only.

    Parameters
    ----------
    x, y : array_like
        Samples to compare (flattened internally).
    B : int, default 500
        Number of label permutations for the null distribution. Set B=0 to
        skip p‑value computation (returns ``(stat, nan)``).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    stat : float
        KS statistic.
    p_boot : float
        Permutation p‑value in [0, 1]. If ``B==0`` returns ``np.nan``.
    """
    xs = np.sort(np.asarray(x).ravel())
    ys = np.sort(np.asarray(y).ravel())
    grid = np.concatenate([xs, ys])
    Fx = np.searchsorted(xs, grid, side="right") / xs.size
    Fy = np.searchsorted(ys, grid, side="right") / ys.size
    stat = float(np.max(np.abs(Fx - Fy)))

    if B <= 0:
        return stat, float("nan")

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([xs, ys])
    n = xs.size
    ge = 0
    for _ in range(B):
        rng.shuffle(pooled)
        xb = np.sort(pooled[:n]); yb = np.sort(pooled[n:])
        gridb = np.concatenate([xb, yb])
        Fxb = np.searchsorted(xb, gridb, side="right") / xb.size
        Fyb = np.searchsorted(yb, gridb, side="right") / yb.size
        stat_b = np.max(np.abs(Fxb - Fyb))
        if stat_b >= stat - 1e-12:
            ge += 1
    return stat, ge / B


def wasserstein1d(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein (Earth‑Mover) distance, NumPy‑only.

    Sorts both samples and averages absolute pairwise differences up to the
    shorter length (robust and fast for 1D comparisons).
    """
    xs = np.sort(np.asarray(x).ravel())
    ys = np.sort(np.asarray(y).ravel())
    m = min(xs.size, ys.size)
    if m == 0:
        return float("nan")
    return float(np.mean(np.abs(xs[:m] - ys[:m])))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_qq(real: np.ndarray,
            gen: np.ndarray,
            name: str,
            qs: Optional[Sequence[float]] = None,
            n_boot: int = 200,
            ci: float = 0.95,
            seed: int = 0) -> None:
    """Q–Q plot with optional bootstrap band (over generated sample).

    Parameters
    ----------
    real, gen : array_like
        Samples to compare.
    name : str
        Label/title for the plot.
    qs : sequence or None
        Quantile levels in [0,1]. If None, uses 0.01..0.99 evenly.
    n_boot : int
        Number of bootstrap resamples for the band. Set 0 to disable.
    ci : float
        Confidence level for the band.
    seed : int
        RNG seed for bootstrap.
    """
    real = np.asarray(real).ravel(); gen = np.asarray(gen).ravel()
    if qs is None:
        qs = np.linspace(0.01, 0.99, 99)
    qr = np.quantile(real, qs)
    qg = np.quantile(gen,  qs)

    lo = hi = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.stack([
            np.quantile(rng.choice(gen, size=gen.size, replace=True), qs)
            for _ in range(n_boot)
        ], axis=0)
        lo = np.quantile(boots, (1-ci)/2, axis=0)
        hi = np.quantile(boots, 1-(1-ci)/2, axis=0)

    m = min(qr.min(), qg.min()); M = max(qr.max(), qg.max())
    plt.figure(figsize=(4.2, 4.2))
    plt.plot(qr, qg, 'o', alpha=0.6, label='Q–Q')
    plt.plot([m, M], [m, M], 'k--', lw=1, label='y=x')
    if lo is not None:
        plt.fill_between(qr, lo, hi, alpha=0.2, label=f'{int(ci*100)}% band')
    plt.xlabel('Real quantiles'); plt.ylabel('Gen quantiles')
    plt.title(f'Q–Q {name}')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# Multiple testing helper
# ---------------------------------------------------------------------------

def fdr_bh(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR mask.

    Parameters
    ----------
    pvals : sequence of float
        p‑values for multiple tests.
    alpha : float
        Desired FDR level.

    Returns
    -------
    np.ndarray of bool
        Mask of rejected hypotheses after FDR control.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m+1) / m)
    sig = ranked <= thresh
    mask = np.zeros(m, dtype=bool)
    if sig.any():
        kmax = np.max(np.where(sig))
        mask[order[:kmax+1]] = True
    return mask

# ---------------------------------------------------------------------------
# Flow‑specific diagnostics used after training (A1/A2 consolidation)
# ---------------------------------------------------------------------------

def compute_ld_flow_stats(model: torch.nn.Module, x: torch.Tensor, eps: float | None = None) -> Tuple[float, float, float, float]:
    """Return mean/std/min/max of the log‑det **from RealNVP layers only**.

    This subtracts the bijector (logit) contribution from the model's total
    log‑det so you can monitor how much volume change the flow is using.
    """
    with torch.no_grad():
        z, tot_ld = model.forward(x)
        e = eps if eps is not None else getattr(model, "epsilon", 1e-6)
        x_cl = x.clamp(0, 1)
        xs = x_cl * (1 - 2*e) + e
        ld_bij = (
            torch.log(torch.tensor(1 - 2*e, device=x.device))
            - torch.log(xs)
            - torch.log1p(-xs)
        ).sum(dim=1)
        ld_flow = tot_ld - ld_bij
        return (
            float(ld_flow.mean().item()),
            float(ld_flow.std(unbiased=False).item()),
            float(ld_flow.min().item()),
            float(ld_flow.max().item()),
        )


def post_training_smoke_report(
    model: torch.nn.Module,
    val_loader,
    history: Optional[Dict[str, Any]] = None,
    plot_loss_history_fn: Optional[Any] = None,
    sample_N: Optional[int] = None,
    eps: Optional[float] = None,
    device: Optional[torch.device] = None,
    temperature: Optional[float] = None,
    return_table: str = "pandas",
) -> Tuple[Dict[str, float], Any]:
    """Compute and display the post‑training smoke metrics (A1/A2) and plots.

    This function replicates the notebook cell you used after training but
    centralizes the logic here. It prints a compact table and returns the
    metrics dictionary plus, optionally, a pandas DataFrame if available.

    Parameters
    ----------
    model : nn.Module
        Trained flow model (with attributes `layers`, optional `pre_bijector`,
        and `ensure_base_dist(device)`).
    val_loader : DataLoader
        Validation loader to draw a small batch for A2 (log‑det stats).
    history : dict or None
        Training history as returned by `train_model_modular(..., return_history=True)`.
        If provided and `plot_loss_history_fn` is not None, loss curves are plotted.
    plot_loss_history_fn : callable or None
        Function with signature `plot_loss_history(history, ids)` to draw curves.
        You can pass `utils.training.plot_loss_history`.
    sample_N : int or None
        Number of samples for A1 (saturation check). Defaults to 8×batch or 20000 max.
    eps : float or None
        Epsilon used by the logit bijector. Defaults to `model.epsilon` if present.
    device : torch.device or None
        Device to run on. Defaults to the model's device.
    temperature : float or None
        Optional temperature to scale latent samples for A1 visualization.
    return_table : {"pandas", "dict"}
        If "pandas" and pandas is available, additionally return a DataFrame.

    Returns
    -------
    metrics : dict
        Dictionary with keys: 'ld_mean','ld_std','ld_min','ld_max',
        'frac_|y|>8','frac_x<=1e-3','frac_x>=1-1e-3','N'.
    table : pandas.DataFrame or None
        If return_table=="pandas" and pandas is installed, a small table; else None.
    """
    model.eval()
    dev = device or next(model.parameters()).device

    # --- pull one validation batch for A2 ---
    xb = next(iter(val_loader))
    if isinstance(xb, (tuple, list)):
        x_val = xb[0].to(dev)
    elif isinstance(xb, dict):
        x_val = xb['x'].to(dev)
    else:
        x_val = xb.to(dev)

    # A2: log‑det stats from flow only
    ld_mean, ld_std, ld_min, ld_max = compute_ld_flow_stats(model, x_val, eps=eps)

    # A1: saturation before scaler — sample z, invert RealNVP only, then bijector
    with torch.no_grad():
        N_default = min(20000, 8 * x_val.shape[0])
        N = int(sample_N or N_default)
        # ensure base dist
        if hasattr(model, 'ensure_base_dist'):
            model.ensure_base_dist(dev)
            z_samp = model._base_dist.sample((N,)).to(dev)
        elif hasattr(model, '_base_dist') and model._base_dist is not None:
            z_samp = model._base_dist.sample((N,)).to(dev)
        else:
            # fallback
            D = getattr(model, 'input_dim', x_val.shape[1])
            z_samp = torch.randn(N, D, device=dev)
        if temperature is not None:
            z_samp = z_samp * float(temperature)
        # invert through RealNVP layers only
        y = z_samp.clone()
        for layer in reversed(model.layers):
            y, _ = layer(y, reverse=True)
        # apply bijector inverse into (0,1)
        if getattr(model, 'pre_bijector', None) is not None:
            x_scaled, _ = model.pre_bijector(y, reverse=True)
        else:
            # if no bijector, treat y as x
            x_scaled = y

    y_np = y.detach().cpu().numpy().ravel()
    x_np = x_scaled.detach().cpu().numpy().ravel()
    frac_y_big = float(np.mean(np.abs(y_np) > 8.0))
    frac_lo  = float(np.mean(x_np < 1e-3))
    frac_hi  = float(np.mean(x_np > 1 - 1e-3))

    metrics = {
        'ld_mean': ld_mean,
        'ld_std': ld_std,
        'ld_min': ld_min,
        'ld_max': ld_max,
        'frac_|y|>8': frac_y_big,
        'frac_x<=1e-3': frac_lo,
        'frac_x>=1-1e-3': frac_hi,
        'N': int(y_np.size),
    }

    # Pretty table (if pandas available)
    table = None
    if return_table == "pandas":
        try:
            import pandas as pd
            table = pd.DataFrame({
                'stat': ['ld_mean','ld_std','ld_min','ld_max','frac_|y|>8','frac_x<=1e-3','frac_x>=1-1e-3','N'],
                'value': [ld_mean, ld_std, ld_min, ld_max, frac_y_big, frac_lo, frac_hi, int(y_np.size)]
            })
            # pretty print
            with pd.option_context('display.precision', 6):
                print(table)
        except Exception:
            table = None

    # Plot losses if available
    if (history is not None) and (plot_loss_history_fn is not None):
        try:
            ids = list(history.get('train_parts', {}).keys())
            plot_loss_history_fn(history, ids=ids)
        except Exception:
            pass

    # Console summary
    print(f"A2 · ld_flow stats: mean={ld_mean:.3f} std={ld_std:.3f} min={ld_min:.3f} max={ld_max:.3f}")
    print(f"A1 · |y|>8: {frac_y_big:.3%}  |  x<=1e-3: {frac_lo:.3%}  x>=1-1e-3: {frac_hi:.3%}")

    return metrics, table
