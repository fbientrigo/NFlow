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