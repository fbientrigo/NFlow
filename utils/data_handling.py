"""
data_handling.py

Data loading and preprocessing utilities for muon kinematic datasets.

This module provides functions to cache and load preprocessed data, open HDF5 files,
prepare PyTorch DataLoaders, compute kinematic angles, and scale muon features for flow models.
"""

import logging
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def cache_processed_data(cache_path: str, data):
    """
    Cache processed data to a file using PyTorch serialization.

    Parameters
    ----------
    cache_path : str
        Path to the file where `data` will be saved.
    data : any
        Data object to cache (e.g., preprocessed tensors or arrays).

    Returns
    -------
    None

    Notes
    -----
    Uses `torch.save` under the hood and logs the cache location.
    """
    torch.save(data, cache_path)
    logger.info(f"Processed data cached at: {cache_path}")


def load_cached_data(cache_path: str):
    """
    Load cached data if it exists.

    Parameters
    ----------
    cache_path : str
        Path to the cached data file.

    Returns
    -------
    data : any or None
        The loaded data if the file exists; otherwise, `None`.

    Examples
    --------
    >>> data = load_cached_data("cache.pt")
    >>> if data is None:
    ...     data = preprocess(raw)
    ...     cache_processed_data("cache.pt", data)
    """
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False) # changed to weights_only as 0
        logger.info(f"Loaded cached data from: {cache_path}")
        return data
    else:
        return None


def prepare_dataloader(scaled_data: np.ndarray, batch_size: int = 128):
    """
    Create a PyTorch DataLoader from a NumPy array.

    Parameters
    ----------
    scaled_data : np.ndarray of shape (n_samples, n_features)
        Array of preprocessed feature vectors.
    batch_size : int, default=128
        Number of samples per batch.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Shuffled DataLoader that yields batches of shape (batch_size, n_features).
    """
    tensor_data = torch.tensor(scaled_data, dtype=torch.float)
    dataset = TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader


def open_hdf(inpath: str, key: str) -> np.ndarray:
    """
    Load a dataset from an HDF5 file.

    Parameters
    ----------
    inpath : str
        Path to the HDF5 file.
    key : str
        Dataset key to read within the HDF5 file.

    Returns
    -------
    data : np.ndarray
        Loaded data array.

    Raises
    ------
    RuntimeError
        If the file cannot be opened or the key does not exist.
    """
    try:
        with h5py.File(inpath, 'r') as hf:
            logger.info(f'-- Opening input file at {inpath}')
            data = hf[str(key)][:]
    except Exception as e:
        logger.error(f"Error loading {key}: {e} with path {inpath}")
        raise RuntimeError(f"Error loading data from {inpath} key {key}")
    return data


def open_datafiles(datapath_mothers: str, datapath_daughters: str = None) -> np.ndarray:
    """
    Load mother (and optionally daughter) muon data from HDF5 files.

    Parameters
    ----------
    datapath_mothers : str
        Path to the HDF5 file containing mother muon data under the key "mothers_data".
    datapath_daughters : str, optional
        Path to the HDF5 file containing daughter muon data under the key "daughters_data".
        Currently not used.

    Returns
    -------
    mothers : np.ndarray
        Array of mother muon data.

    Notes
    -----
    The `daughters_data` loading is currently commented out.
    """
    mothers = open_hdf(inpath=datapath_mothers, key="mothers_data")
    # daughters = open_hdf(inpath=datapath_daughters, key="daughters_data") if datapath_daughters else None
    return mothers


def compute_angles(px, py, pz, handle_zero: bool = True):
    """
    Compute polar (theta) and azimuthal (phi) angles from momentum components.

    Parameters
    ----------
    px, py, pz : array-like or scalar
        Momentum components along x, y, and z axes.
    handle_zero : bool, default=True
        If True, set phi to zero where transverse momentum is zero,
        and theta to NaN where total momentum is zero.

    Returns
    -------
    theta : array-like or scalar
        Polar angle computed as arctan2(sqrt(px^2 + py^2), pz).
    phi : array-like or scalar
        Azimuthal angle computed as arctan2(py, px).

    Examples
    --------
    >>> theta, phi = compute_angles(np.array([1,0]), np.array([0,1]), np.array([1,1]))
    """
    mag_pt = np.hypot(px, py)
    theta = np.arctan2(mag_pt, pz)
    phi = np.arctan2(py, px)

    if handle_zero:
        # Handle zero transverse momentum
        if np.all(mag_pt == 0):
            phi = np.zeros_like(px) if isinstance(px, np.ndarray) else 0.0
        # Handle zero total momentum
        mag_p = np.sqrt(px**2 + py**2 + pz**2)
        if np.all(mag_p == 0):
            theta = np.full_like(px, np.nan) if isinstance(px, np.ndarray) else np.nan

    return theta, phi


def scale_muon_data(mothers_df: np.ndarray,
                    plotdir: str,
                    scaler_mother: MinMaxScaler):
    """
    Compute energy, apply scaling to muon features, and save diagnostic plots.

    Parameters
    ----------
    mothers_df : np.ndarray of shape (n_samples, 3)
        Raw muon momentum data columns [px, py, pz].
    plotdir : str
        Directory where pre/post scaling histograms will be saved.
    scaler_mother : QuantileTransformer
        Scaler instance (e.g., QuantileTransformer or StandardScaler) to fit and apply.

    Returns
    -------
    mother_scaled : np.ndarray of shape (n_samples, 4)
        Scaled features [px, py, pz, energy].
    scaler_mother : QuantileTransformer
        Fitted scaler instance.
    raw_features : np.ndarray of shape (n_samples, 4)
        Original features before scaling.

    Notes
    -----
    - Energy is computed as sqrt(px^2 + py^2 + pz^2 + mass_muon^2).
    - Saves pre- and post-scaling histograms for each feature in `plotdir`.
    """
    mass_muon = 0.1134289259  # GeV
    raw_features = mothers_df[:, :3]
    energy = np.sqrt((raw_features**2).sum(axis=1) + mass_muon**2)
    raw_features = np.column_stack((raw_features, energy))

    # Fit and transform
    mother_scaled = scaler_mother.fit_transform(raw_features)

    # Plot diagnostics
    bins = 100
    rangevar = [-10.0, 10.0]
    feature_names = ['px', 'py', 'pz', 'energy']
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(8, 5), dpi=150)
        plt.hist(raw_features[:, i], bins=bins, alpha=0.5,
                 label='Original', density=True, range=rangevar)
        plt.hist(mother_scaled[:, i], bins=bins, alpha=0.5,
                 label='Scaled', density=True, range=rangevar)
        plt.xlabel(f"{feature} [GeV]")
        plt.ylabel("Density")
        plt.legend(frameon=False)
        plt.title(f"Mother {feature} pre and post scaling")
        plt.tight_layout()
        plot_path = os.path.join(plotdir, f"{feature}_mother_pre_post_scaling.pdf")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved scaling plot: {plot_path}")

    logger.info("Done plotting pre and post scaling features for mothers.")
    return mother_scaled, scaler_mother, raw_features
