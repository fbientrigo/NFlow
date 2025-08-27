# Synthesis of Data for Testing
# This module provides functions to synthesize data for testing purposes.

import numpy as np
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer

# cross dependencies

from utils.data_handling import scale_muon_data, open_hdf
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger(__name__)


def generate_mothers_data(n_events: int,
                          cluster_frac: float = 0.7,
                          centers: list = [(-2,0,0), (2,0,0)],
                          sigma_cluster: float = 0.5,
                          ring_frac: float = 0.3,
                          r0: float = 3.0,
                          sigma_r: float = 0.2,
                          sigma_z: float = 0.3,
                          seed: int = 42) -> np.ndarray:
    """
    Genera n_events de momentum (px,py,pz) con:
     - cluster_frac % en 2 gaussianas (en centers, sigma_cluster)
     - ring_frac % en un anillo de radio r0 con sigma_r en (px,py) y pz~N(0,sigma_z)
    """
    np.random.seed(seed)
    nc = int(n_events * cluster_frac)
    nr = n_events - nc

    # 1) Clusters Gaussianos
    clusters = []
    per_cluster = nc // len(centers)
    for cx, cy, cz in centers:
        pts = np.random.randn(per_cluster, 3) * sigma_cluster
        pts += np.array([cx, cy, cz])[None, :]
        clusters.append(pts)
    # Si hay “sobrantes” por división entera:
    if sum(len(c) for c in clusters) < nc:
        extra = nc - sum(len(c) for c in clusters)
        cx, cy, cz = centers[0]
        extra_pts = np.random.randn(extra, 3) * sigma_cluster + np.array([cx, cy, cz])
        clusters.append(extra_pts)
    cluster_data = np.vstack(clusters)

    # 2) Donut/ring en px-py
    # ángulos uniformes en [0,2π)
    thetas = np.random.rand(nr) * 2 * np.pi
    # radios alrededor de r0
    rs = np.random.randn(nr) * sigma_r + r0
    px_ring = rs * np.cos(thetas)
    py_ring = rs * np.sin(thetas)
    # pz gaussiano
    pz_ring = np.random.randn(nr) * sigma_z
    ring_data = np.stack([px_ring, py_ring, pz_ring], axis=1)

    # 3) Mezcla final y barajado
    data = np.vstack([cluster_data, ring_data]).astype(np.float32)
    np.random.shuffle(data)
    return data



def filter_deep_inelastic(data: np.ndarray,
                          q_pz: float = 0.8,
                          q_pt: float = 0.8) -> np.ndarray:
    """
    Filtra eventos con pz y pt (sqrt(px^2+py^2) ) superiores a los cuantiles dados.
    q_pz, q_pt deben estar en (0,1).
    """
    px = data[:, 0]
    py = data[:, 1]
    pz = data[:, 2]
    pt = np.sqrt(px**2 + py**2)
    thresh_pz = np.quantile(pz, q_pz)
    thresh_pt = np.quantile(pt, q_pt)

    mask = (pz >= thresh_pz) & (pt >= thresh_pt)
    return data[mask]

def save_hdf5(data: np.ndarray,
              filename: str,
              dataset_name: str = 'mothers_data') -> None:
    """
    Guarda un array en un archivo HDF5 bajo el nombre dataset_name.
    """
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(dataset_name, data=data)


# -- Guided Section ---

def create_lvl2_mask(data: np.ndarray,
                     q_pz: float = 0.8,
                     q_pt: float = 0.8) -> np.ndarray:
    """
    Devuelve máscara booleana (shape = n_events) indicando
    qué filas cumplen criterio de deep inelastic scattering.
    """
    px, py, pz = data[:,0], data[:,1], data[:,2]
    pt = np.sqrt(px**2 + py**2)
    thresh_pz = np.quantile(pz, q_pz)
    thresh_pt = np.quantile(pt, q_pt)
    mask = (pz >= thresh_pz) & (pt >= thresh_pt)
    logger.info(f"Nivel2 mask: {mask.sum()}/{len(mask)} eventos")
    return mask

def prepare_guided_dataloaders(path_lvl1: str,
                               batch_size: int = 128,
                               q_pz: float = 0.8,
                               q_pt: float = 0.8,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               plotdir: str = None,
                               scaler_mother: QuantileTransformer = QuantileTransformer()):
    """
    Loads data from lvl1
    1) Carga datos nivel1 desde HDF5.
    2) Escala con QuantileTransformer.
    3) Crea máscara lvl2 reaplicando el criterio.
    4) Separa train/val y retorna DataLoaders que devuelven (x, mask_lvl2).
    """
    # Loading
    data = open_hdf(inpath=path_lvl1, key="mothers_data")

    # Scaling
    scaled, scaler, _ = scale_muon_data(data, plotdir, scaler_mother)

    # lvl 2 mask
    mask = create_lvl2_mask(data, q_pz=q_pz, q_pt=q_pt)

    # Train/Val split
    X_tr, X_va, m_tr, m_va = train_test_split(
        scaled, mask, test_size=test_size,
        stratify=mask, random_state=random_state
    )

    # DataLoaders
    def build_loader(X, m):
        t_X = torch.tensor(X, dtype=torch.float)
        t_m = torch.tensor(m, dtype=torch.bool)
        ds = TensorDataset(t_X, t_m)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    train_loader = build_loader(X_tr, m_tr)
    val_loader   = build_loader(X_va, m_va)

    return train_loader, val_loader, scaler

