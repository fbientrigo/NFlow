import logging
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils.data_handling import compute_angles

logger = logging.getLogger(__name__)

def plot_feature_histograms(true_data, generated_data, run_dir, space_tag="scaled", bins=50, verbose=True):

    # Check number of features
    if true_data.shape[1] not in [3, 4]:
        raise ValueError("Expected 3 or 4 features in data.")
    
    feature_names = ['px', 'py', 'pz']
    if true_data.shape[1] == 4:
        feature_names.append('energy')
    
    os.makedirs(run_dir, exist_ok=True)
    
    rangedict = {
        'original': {'px': [-10., 10.], 'py': [-7.5, 7.5], 'pz': [0., 30.], 'energy': [0., 30.]},
        'scaled':   {'px': [-10., 10.], 'py': [-7.5, 7.5], 'pz': [-5., 5.],  'energy': [-5., 5.]}
    }
    units = ['[GeV]']

    # Plot for each feature: basic and with error bars
    for i, feature in enumerate(feature_names):
        histrange = rangedict.get(space_tag, {}).get(feature, [-10., 10.])
        _plot_basic_histogram(true_data, generated_data, feature, i, bins, histrange, units, space_tag, run_dir, verbose)
        _plot_histogram_with_error_bars(true_data, generated_data, feature, i, bins, histrange, units, space_tag, run_dir, verbose)
    
    # Angular distributions
    theta_true, phi_true = compute_angles(true_data[:, 0], true_data[:, 1], true_data[:, 2], handle_zero=True)
    theta_gen, phi_gen = compute_angles(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], handle_zero=True)
    _plot_angular_histogram(theta_true, theta_gen, bins, [0, (np.pi)/2], r"$\theta$ [rad]", "hist_theta.png", run_dir, verbose,
                            xticks=[0, np.pi/4, np.pi/2],
                            xtick_labels=['0', r'$\pi/4$', r'$\pi/2$'])
    _plot_angular_histogram(phi_true, phi_gen, bins, [-np.pi, np.pi], r"$\phi$ [rad]", "hist_phi.png", run_dir, verbose,
                            xticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                            xtick_labels=[r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    return


def _plot_basic_histogram(true_data, generated_data, feature, feature_index, bins, hist_range, units, space_tag, run_dir, verbose):

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.hist(true_data[:, feature_index], bins=bins, alpha=0.6, density=True,
            label='True', color='tab:blue', range=hist_range)
    ax.hist(generated_data[:, feature_index], bins=bins, alpha=0.6, density=True,
            label='Generated', color='tab:orange', range=hist_range)

    ax.set_xlabel(f"{feature} {units[0]}", fontsize=15)
    ax.set_ylabel("a.u.", fontsize=15)
    ax.legend(frameon=False, fontsize=15)

    ax.minorticks_on()
    ax.ticklabel_format(axis='both', style='sci')
    ax.tick_params('both', direction='in', length=8, width=1, which='major', top=True, right=True)
    ax.tick_params('both', direction='in', length=4, width=0.5, which='minor', top=True, right=True)
    plt.setp(ax.spines.values(), lw=1.25)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, f"hist_{feature}_{space_tag}.png")
    plt.savefig(plot_path)
    plt.close()

    if verbose:
        logger.info(f"Saved histogram plot: {plot_path}")
    return


def _plot_histogram_with_error_bars(true_data, generated_data, feature, feature_index,
                                    bins, hist_range, units, space_tag, run_dir, verbose):

    true_counts, bin_edges = np.histogram(true_data[:, feature_index],
                                          bins=bins, range=hist_range)
    gen_counts, _ = np.histogram(generated_data[:, feature_index],
                                 bins=bins, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Bin uncertainty
    true_errors = np.sqrt(true_counts) / (np.sum(true_counts) * 1.0)
    gen_errors  = np.sqrt(gen_counts)  / (np.sum(gen_counts)  * 1.0)

    fig = plt.figure(dpi=150)
    gs  = gridspec.GridSpec(2, 1, height_ratios = [8,2])
    axes = plt.subplot(gs[0]), plt.subplot(gs[1])

    sigma_panel = np.sqrt(true_errors**2 + gen_errors**2)
    panel     = (true_counts - gen_counts)/sigma_panel
    panel     = [0 if np.isnan(elem) else elem for elem in panel]

    # Bar charts for normalized counts
    axes[0].bar(bin_centers, true_counts/np.sum(true_counts),
             width=bin_widths, alpha=0.6, label='True',
             color='tab:blue', align='center')
    axes[0].bar(bin_centers, gen_counts/np.sum(gen_counts),
             width=bin_widths, alpha=0.6, label='Generated',
             color='tab:orange', align='center')

    # Error bars
    axes[0].errorbar(bin_centers, true_counts/np.sum(true_counts),
                  yerr=true_errors, color='tab:blue', capsize=2, fmt=' ', linestyle=None)
    axes[0].errorbar(bin_centers, gen_counts/np.sum(gen_counts),
                  yerr=gen_errors, color='tab:orange', capsize=2, fmt=' ', linestyle=None)

    # Axes labels and styling
    axes[0].set_xlabel(f"{feature} {str(units[0])}", fontsize=15)
    axes[0].set_ylabel("a.u.", fontsize=15)
    axes[0].legend(frameon=False, fontsize=15)
    axes[0].minorticks_on()
    axes[0].ticklabel_format(axis='both', style='sci')
    axes[0].tick_params('both', direction='in', length=8, width=1, which='major', top=True, right=True)
    axes[0].tick_params('both', direction='in', length=4, width=0.5, which='minor', top=True, right=True)

    xmin = bin_edges[0]
    xmax = bin_edges[-1]
    axes[1].hist(bin_edges[:-1], bin_edges, weights=panel, color='green', alpha=0.4)
    axes[1].hlines(-2.5, xmin, xmax, color='r', linestyles='--', alpha=0.5, linewidth=1.25)
    axes[1].hlines( 2.5, xmin, xmax, color='r', linestyles='--', alpha=0.5, linewidth=1.25)
    axes[1].hlines(-5.,  xmin, xmax, color='r', linestyles='-', alpha=0.5, linewidth=1.25)
    axes[1].hlines( 5.,  xmin, xmax, color='r', linestyles='-', alpha=0.5, linewidth=1.25)
    axes[1].set_ylabel('Pull', fontsize=15)
    axes[1].set_xlim((xmin,xmax))
    axes[1].ticklabel_format( axis='both', style='sci', scilimits=(0,2))

    pull_lim = max([abs(j) for j in panel])
    axes[1].set_ylim((-pull_lim-2, pull_lim+2))

    plt.setp(axes[0].spines.values(), lw=1.25)
    plt.setp(axes[1].spines.values(), lw=1.25)
    plt.tight_layout()

    # Save figure
    plot_path_error = os.path.join(run_dir, f"hist_{feature}_{space_tag}_error.png")
    plt.savefig(plot_path_error)
    plt.close()
    if verbose:
        logger.info(f"Saved: {plot_path_error}")
    return


def _plot_angular_histogram(data_true, data_gen, bins, range_, label, output_name, run_dir, verbose, xticks=None, xtick_labels=None):

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.hist(data_true, bins=bins, alpha=0.6, density=True, label='True', color='tab:blue', range=range_)
    ax.hist(data_gen, bins=bins, alpha=0.6, density=True, label='Generated', color='tab:orange', range=range_)
    ax.set_xlabel(label, fontsize=15)
    ax.set_ylabel("a.u.", fontsize=15)
    ax.legend(frameon=False, fontsize=15)

    ax.minorticks_on()
    ax.ticklabel_format(axis='both', style='sci')
    ax.tick_params('both', direction='in', length=8, width=1, which='major', top=True, right=True)
    ax.tick_params('both', direction='in', length=4, width=0.5, which='minor', top=True, right=True)
    if xticks is not None and xtick_labels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

    plt.setp(ax.spines.values(), lw=1.25)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, output_name)
    plt.savefig(plot_path)
    plt.close()
    if verbose:
        logger.info(f"Saved angular histogram: {plot_path}")
    return


# ---- For plotting at colab ----

# ---------- helpers ----------
def pT(px, py): return np.hypot(px, py)

def qrange(data, qlo=0.01, qhi=0.99):
    lo = np.quantile(data, qlo); hi = np.quantile(data, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(data)), float(np.max(data))
        if lo == hi: lo, hi = lo - 1.0, hi + 1.0
    return lo, hi

def joint_range(a, b, qlo=0.01, qhi=0.99):
    lo1, hi1 = qrange(a, qlo, qhi)
    lo2, hi2 = qrange(b, qlo, qhi)
    return min(lo1, lo2), max(hi1, hi2)



# ---------- mask based in quantiles of real set ----------
def make_lvl2_mask(x, q_pz=0.8, q_pt=0.8):
    """
    This just assumes a quantile way of defining high impacticity
    """
    px, py, pz = x[:,0], x[:,1], x[:,2]
    pt = pT(px, py)
    thr_pz = np.quantile(pz, q_pz)
    thr_pt = np.quantile(pt, q_pt)
    return (pz >= thr_pz) & (pt >= thr_pt), thr_pz, thr_pt

# ---------- plotting Muon Data ----------
from .data_handling import filter_by_id


def _weighted_kde(x, y, w, xlabel, ylabel, title, out_path, cmap):
    """Generate the weighted 2D KDE plot and save to file."""
    plt.figure(figsize=(6, 5))
    sns.kdeplot(
        x=x, y=y, weights=w, fill=True, cmap=cmap, cbar=True, thresh=0.01
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_muon_maps(data : np.ndarray,
                   muon_id: int | None = None,
                   output_dir: str = "figs_muons",
                   cmap: str = "mako",
                   max_points: int | None = None   # ← NUEVO argumento opcional
                   ) -> None:
    """
    Filters by particle id and generate four 2D maps, saved as PNG files:
    - x vs y
    - sqrt(x²+y²) vs z
    - px vs py
    - sqrt(px²+py²) vs pz
    Each map is a weighted KDE plot using the weights from the data.

    Parameters
    ----------
    data : np.array
    muon_id : int | None
    output_dir : str
    cmap : str
    max_points : int | None
        Maximum number of events to plot (random sample).
    """
    os.makedirs(output_dir, exist_ok=True)
    #data = filter_by_id(load_muon_data(file_path), muon_id)
    data = filter_by_id(data, muon_id)

    # --- NUEVO: muestreo aleatorio si se especifica ---
    if max_points is not None and len(data) > max_points:
        idx = np.random.choice(len(data), size=max_points, replace=False)
        data = data[idx]

    px, py, pz, x, y, z, _, w = [data[:, i] for i in range(8)]
    r_xy = np.sqrt(x**2 + y**2)
    pt   = np.sqrt(px**2 + py**2)

    plots = [
        (x, y,           "x [m]",     "y [m]",        "Map x vs y",        "x_vs_y.png"),
        (r_xy, z,        "r_xy [m]",  "z [m]",        "Map sqrt(x²+y²) vs z", "r_xy_vs_z.png"),
        (px, py,         "px [GeV/c]","py [GeV/c]",   "Map px vs py",      "px_vs_py.png"),
        (pt, pz,         "pt [GeV/c]","pz [GeV/c]",   "Map sqrt(px²+py²) vs pz","pt_vs_pz.png"),
    ]

    for xdat, ydat, xlabel, ylabel, title, fname in plots:
        out_path = os.path.join(output_dir, fname)
        _weighted_kde(xdat, ydat, w, xlabel, ylabel, title, out_path, cmap)


def plot_muon_histograms(
    data: np.ndarray,
    muon_id: int | None = None,
    output_dir: str = "figs_muons",
    cmap: str = "mako",
    max_points: int | None = None
) -> None:
    """
    Generate weighted 1D histograms of physical variables:
    p_T, p_z, x, y, z, and relativistic energy E.

    Parameters
    ----------
    data: np.ndarray
        File with loaded muon data.
    muon_id : int | None
        PDG code for filtering particle.
    output_dir : str
        Output directory for saving plots.
    cmap : str
        Color palette seaborn/matplotlib.
    max_points : int | None
        Maximum number of events to sample to avoid memory overload.
    """
    os.makedirs(output_dir, exist_ok=True)
    # data = filter_by_id(load_muon_data(file_path), muon_id)
    data = filter_by_id(data, muon_id)

    # Muestreo opcional para no sobrecargar memoria
    if max_points is not None and len(data) > max_points:
        idx = np.random.choice(len(data), size=max_points, replace=False)
        data = data[idx]

    # Separar columnas
    px, py, pz, x, y, z, _, w = [data[:, i] for i in range(8)]
    pT = np.sqrt(px**2 + py**2)

    # Masa del muón en GeV/c^2
    m_mu = 0.105658
    p_abs = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_abs**2 + m_mu**2)

    variables = [
        (pT, "p_T [GeV/c]", "Histogram p_T", "hist_pT.png"),
        (pz, "p_z [GeV/c]", "Histogram p_z", "hist_pz.png"),
        (x,  "x [m]",       "Histogram x",   "hist_x.png"),
        (y,  "y [m]",       "Histogram y",   "hist_y.png"),
        (z,  "z [m]",       "Histogram z",   "hist_z.png"),
        (E,  "E [GeV]",     "Histogram Energy", "hist_E.png"),
    ]

    for values, xlabel, title, fname in variables:
        # Aseguramos que `values` y `w` tengan la misma longitud
        assert len(values) == len(w), \
            f"Longitudes distintas: {len(values)} vs {len(w)}"

        plt.figure(figsize=(6, 4))
        sns.histplot(
            x=values,                # <─ forzamos modo long-form
            weights=w,               # pesos del mismo tamaño
            bins=50,
            kde=False,
            color=sns.color_palette(cmap, n_colors=1)[0]
        )
        plt.xlabel(xlabel)
        plt.ylabel("Eventos ponderados")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
