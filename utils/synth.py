# Synthesis of Data for Testing
# This module provides functions to synthesize data for testing purposes.

import numpy as np
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer

import math
import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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

# --- GPU Cache in case of having an idle system ----



# ---- Dataset iterable que sube bloques a GPU (como antes) ----
class _GPUBlockBatchedDataset(IterableDataset):
    def __init__(self, X_cpu, M_cpu, batch_size, device, target_gb=5.0, safety=0.6, shuffle=True, seed=42):
        super().__init__()
        self.X = X_cpu.contiguous()
        self.M = M_cpu.contiguous()
        self.bs = int(batch_size)
        self.dev = device
        self.shuffle = bool(shuffle)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        N, D = self.X.shape
        bytes_per_sample = D * 4  # float32
        target_bytes = target_gb * (1024**3) * safety
        block_elems = max(self.bs, int(target_bytes // bytes_per_sample))
        self.block_size = max(self.bs, min(N, block_elems))

    def __iter__(self):
        N = self.X.shape[0]
        idx = torch.randperm(N, generator=self.rng) if self.shuffle else torch.arange(N)
        for s in range(0, N, self.block_size):
            e = min(N, s + self.block_size)
            ib = idx[s:e]
            # non_blocking aprovecha pin_memory() si estuvo disponible
            x_block = self.X[ib].to(self.dev, non_blocking=True)
            m_block = self.M[ib].to(self.dev, non_blocking=True)
            B = x_block.shape[0]
            for bs in range(0, B, self.bs):
                be = min(B, bs + self.bs)
                yield (x_block[bs:be], m_block[bs:be])
    def __len__(self):
        N = self.X.shape[0]
        return max(1, math.ceil(N / self.bs))  # nº de mini-batches por época


def prepare_guided_dataloaders_gpu_cached(
    path_lvl1: str,
    batch_size: int = 8192,
    q_pz: float = 0.8,
    q_pt: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42,
    plotdir: str | None = None,
    scaler_mother: QuantileTransformer = QuantileTransformer(),
    device: str | torch.device = "cuda",
    target_gpu_cache_gb: float = 5.0,
    safety: float = 0.6,
    shuffle: bool = True,
):
    # 1) Carga + 2) Escala + 3) Máscara
    data = open_hdf(inpath=path_lvl1, key="mothers_data")
    scaled, scaler, _ = scale_muon_data(data, plotdir, scaler_mother)
    mask = create_lvl2_mask(data, q_pz=q_pz, q_pt=q_pt)

    # 4) Split
    X_tr, X_va, m_tr, m_va = train_test_split(
        scaled, mask, test_size=test_size, stratify=mask, random_state=random_state
    )

    # 5) Tensores CPU (primero) y pin_memory DESPUÉS
    #    Nota: usar from_numpy evita copias extra
    X_tr_t = torch.from_numpy(X_tr).to(dtype=torch.float32)
    X_va_t = torch.from_numpy(X_va).to(dtype=torch.float32)
    m_tr_t = torch.from_numpy(m_tr).to(dtype=torch.bool)
    m_va_t = torch.from_numpy(m_va).to(dtype=torch.bool)

    have_cuda = torch.cuda.is_available()
    dev = torch.device(device if have_cuda else "cpu")

    if have_cuda:
        # pinnear solo features (lo que pesa de verdad)
        try:
            X_tr_t = X_tr_t.pin_memory()
            X_va_t = X_va_t.pin_memory()
            # m_tr_t / m_va_t es pequeño; no es necesario pinnearlo
        except RuntimeError:
            # si por alguna razón no se puede, seguimos sin pin
            pass
    else:
        # Sin GPU: devolvemos loaders estándar
        def _mk_loader(X, m):
            ds = TensorDataset(X, m)
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        return _mk_loader(X_tr_t, m_tr_t), _mk_loader(X_va_t, m_va_t), scaler

    ds_train = _GPUBlockBatchedDataset(
        X_tr_t, m_tr_t, batch_size=batch_size, device=dev,
        target_gb=target_gpu_cache_gb, safety=safety, shuffle=shuffle, seed=random_state
    )
    ds_val = _GPUBlockBatchedDataset(
        X_va_t, m_va_t, batch_size=batch_size, device=dev,
        target_gb=max(1.0, target_gpu_cache_gb * 0.25), safety=safety, shuffle=False, seed=random_state
    )

    train_loader = DataLoader(ds_train, batch_size=None, shuffle=False, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=None, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, scaler