import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def ks_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute a simple two-sample Kolmogorov–Smirnov (KS) statistic in 1D.

    This implementation builds empirical CDFs by matching the sorted order
    statistics of both samples and measuring the maximum absolute difference
    between their CDF values. It is lightweight and dependency-free, but it is
    an approximation when sample sizes differ (it aligns the first `m =
    min(nx, ny)` order statistics rather than evaluating on the pooled grid).

    Parameters
    ----------
    x : np.ndarray
        First sample (any shape; will be flattened).
    y : np.ndarray
        Second sample (any shape; will be flattened).

    Returns
    -------
    float
        KS statistic D in [0, 1]. Larger values indicate larger distributional
        discrepancy. No p-value is computed.

    Notes
    -----
    - For exact two-sample KS on the pooled grid and p-values, use a more
      complete routine (e.g., SciPy or a custom pooled-ECDF implementation).
    - This function is intended for quick, visual/diagnostic comparisons.
    """
    x = np.sort(x.ravel()); y = np.sort(y.ravel())
    nx, ny = len(x), len(y)
    m = min(nx, ny)
    Fx = (np.arange(1, m+1)) / nx
    Fy = (np.arange(1, m+1)) / ny
    return float(np.max(np.abs(Fx - Fy)))


def fd_bins(a: np.ndarray, min_bins=20, max_bins=80) -> int:
    """
    Suggest a histogram bin count using the Freedman–Diaconis rule.

    The bin width is `h = 2 * IQR * n^(-1/3)`, where IQR is the interquartile
    range (Q3 - Q1) of the data and `n` is the number of samples. The suggested
    number of bins is `(max(a) - min(a)) / h`, clamped to `[min_bins, max_bins]`.

    Parameters
    ----------
    a : np.ndarray
        Sample data (any shape; will be flattened).
    min_bins : int, optional
        Lower bound for the returned number of bins (default 20).
    max_bins : int, optional
        Upper bound for the returned number of bins (default 80).

    Returns
    -------
    int
        Recommended number of bins for a 1D histogram.

    Notes
    -----
    - If the IQR is zero (e.g., nearly constant data), the function falls back
      to `min_bins`.
    - The returned value is an integer and may be clipped by the provided bounds.
    """
    a = a.ravel()
    q25, q75 = np.quantile(a, [0.25, 0.75])
    iqr = max(q75 - q25, 1e-9)
    h = 2 * iqr * (len(a) ** (-1/3))
    if h <= 0:
        return min_bins
    bins = int(np.ceil((a.max() - a.min()) / h))
    return int(np.clip(bins, min_bins, max_bins))


def prange(a: np.ndarray, b: np.ndarray, qlo=0.01, qhi=0.99):
    """
    Compute a robust plotting range from the combined percentiles of two arrays.

    The lower bound is the minimum of the `qlo`-quantiles of `a` and `b`; the
    upper bound is the maximum of the `qhi`-quantiles. If the resulting range is
    invalid (NaN/Inf or zero width), the function falls back to the min/max of
    both arrays, and if still degenerate, expands by ±1.0 around the constant.

    Parameters
    ----------
    a : np.ndarray
        First dataset (any shape; will be flattened).
    b : np.ndarray
        Second dataset (any shape; will be flattened).
    qlo : float, optional
        Lower percentile in [0, 1] (default 0.01 → 1st percentile).
    qhi : float, optional
        Upper percentile in [0, 1] (default 0.99 → 99th percentile).

    Returns
    -------
    (float, float)
        Tuple `(lo, hi)` defining a robust range suitable for histogram/plot axes.

    Notes
    -----
    - Using percentiles mitigates the influence of outliers on axis limits.
    - Choose tighter percentiles (e.g., 0.05–0.95) for very heavy-tailed data.
    """
    lo = min(np.quantile(a, qlo), np.quantile(b, qlo))
    hi = max(np.quantile(a, qhi), np.quantile(b, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        if lo == hi: lo, hi = lo-1.0, hi+1.0
    return lo, hi


def inverse_scale(t: torch.Tensor, scaler_mother : MinMaxScaler = MinMaxScaler()) -> torch.Tensor:
    if 'scaler_mother' in globals() and scaler_mother is not None:
        return scaler_mother.inverse_transform(t.detach().cpu().numpy())
    else:
        return t.detach().cpu().numpy()


def numerical_comparison(val_real, gen_real, feat_names=['px','py','pz','energy']):
    N, D = val_real.shape
    feat_idx = list(range(min(4, D)))
    # Numerical Comparison Results
    for i, name in zip(feat_idx, feat_names):
        r = val_real[:, i]; g = gen_real[:, i]
    print(f"{name:>6s} | μ_real={np.mean(r):+.3e} μ_gen={np.mean(g):+.3e} | σ_real={np.std(r):.3e} σ_gen={np.std(g):.3e} | KS={ks_1d(r,g):.3f}")


def plot_comparison(val_real, gen_real, feat_names=['px','py','pz','energy'], rows=2, cols=2):
    """
    Plot a side by side comparison between generated and real data.

    Args:
        val_real (np.ndarray): Real data.
        gen_real (np.ndarray): Generated data.
        feat_names (list): List of feature names.
    """
    N, D = val_real.shape
    feat_idx = list(range(min(4, D)))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5), constrained_layout=True)
    axes = axes.ravel()

    for ax, i, name in zip(axes, feat_idx, feat_names):
        r = val_real[:, i]; g = gen_real[:, i]
        ks = ks_1d(r, g)
        rng = prange(r, g, 0.01, 0.99)
        bins = max(fd_bins(r), fd_bins(g))

        ax.hist(r, bins=bins, range=rng, density=True, alpha=0.45, label='Real')
        ax.hist(g, bins=bins, range=rng, density=True, alpha=0.45, label='Gen')
        mu_r, mu_g = np.mean(r), np.mean(g)
        sd_r, sd_g = np.std(r),  np.std(g)
        ax.set_title(f"{name} | KS={ks:.3e} | μΔ={mu_g-mu_r:+.2e} | σ×={sd_g/(sd_r+1e-12):.2e}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(name); ax.set_ylabel("Density")

    axes[0].legend(frameon=False)
    plt.show()

