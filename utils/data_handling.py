import logging
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def cache_processed_data(cache_path, data):
    torch.save(data, cache_path)
    logger.info(f"Processed data cached at: {cache_path}")
    return

def load_cached_data(cache_path):
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        logger.info(f"Loaded cached data from: {cache_path}")
        return data
    else:
        return None

def prepare_dataloader(scaled_data, batch_size=128):
    tensor_data = torch.tensor(scaled_data, dtype=torch.float)
    dataset = TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def open_hdf(inpath, key):
    try:
        with h5py.File(inpath, 'r') as hf:
            logger.info(f'-- Opening input file at {inpath}')
            data = hf[str(key)][:]
    except Exception as e:
        logger.error(f"Error loading {key}: {e} with path {inpath}")
        raise RuntimeError('Error loading data')

    return data


def open_datafiles(datapath_mothers, datapath_daughters=None):

    # Mother (muon)
    mothers = open_hdf(inpath=datapath_mothers, key="mothers_data")

    # Daughters
    # daughters = open_hdf(inpath=datapath_daughters, key="daughters_data")
    
    return mothers


def compute_angles(px, py, pz, handle_zero=True):
# Compute polar (theta) and azimuthal (phi) angles from spatial momentum components

    mag_pt = np.hypot(px, py)
    theta = np.arctan2(mag_pt, pz)  # polar angle
    phi = np.arctan2(py, px)        # azimuthal angle

    if handle_zero:
        if np.all(mag_pt == 0):
            phi = np.zeros_like(px) if isinstance(px, np.ndarray) else 0.0

        mag_p = np.sqrt(px**2 + py**2 + pz**2)
        if np.all(mag_p == 0):
            theta = np.full_like(px, np.nan) if isinstance(px, np.ndarray) else np.nan

    return theta, phi


def scale_muon_data(mothers_df, plotdir):
#  Scaling input features using QuantileTransformer

    mass_muon = 0.1134289259 # units GeV
    raw_features = mothers_df[:, :3] # px, py, pz
    energy = np.sqrt(np.sum(raw_features**2, axis=1) + mass_muon**2)
    raw_features = np.column_stack((raw_features, energy))

    scaler_mother = QuantileTransformer(output_distribution='normal', random_state=42)
    mother_scaled = scaler_mother.fit_transform(raw_features)
    
    bins = 100
    rangevar = [-10., 10.]
    feature_names = ['px', 'py', 'pz', 'energy']
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(8, 5), dpi=150)
        plt.hist(raw_features[:, i], bins=bins, alpha=0.5, label='Original', density=True, range=rangevar)
        plt.hist(mother_scaled[:, i], bins=bins, alpha=0.5, label='Scaled', density=True, range=rangevar)
        plt.xlabel(f"{feature} [GeV]")
        plt.ylabel("a.u.")
        plt.legend(frameon=False)
        plt.title(f"Mother {feature} pre and post scaling")
        plt.tight_layout()
        plot_path = os.path.join(plotdir, f"{feature}_mother_pre_post_scaling.pdf")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved scaling plot: {plot_path}")
    
    logger.info('Done plotting pre and post scaling features for mothers.')

    return mother_scaled, scaler_mother, raw_features
