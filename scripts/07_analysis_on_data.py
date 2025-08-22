#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# 1) Rutas locales — apunta aquí a tu carpeta de descargas
data_dir = r"C:\Users\Asus\Downloads\Muons"
files = {
    'FullMC_pre':  os.path.join(data_dir, "muons_FullMC.pkl"),
    'GAN_pre':     os.path.join(data_dir, "muons_GAN.pkl"),
    'FullMC_post': os.path.join(data_dir, "muonsFullMC_afterMS.pkl"),
    'GAN_post':    os.path.join(data_dir, "muonsGAN_afterMS.pkl"),
}

# 2) Crea carpeta de resultados
out_dir = os.path.join(data_dir, "eda_results")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

# * `px, py, pz` – momentum components in **GeV / c**  
#* `x, y, z`    – position of the muon at the scoring plane, expressed in **metres**  
#* `id`         – PDG code
#* `w`          – event weight

col_names = ['px', 'py', 'pz', 'x', 'y', 'z', 'pdg code', 'event weight']

# Función de carga robusta
def load_as_df(path):
    """Carga un pickle (posiblemente gzip) y retorna un DataFrame."""
    # Detecta compresión gzip
    try:
        obj = pd.read_pickle(path, compression='gzip')
    except (ValueError, EOFError):
        obj = pd.read_pickle(path)  # pickle sin compresión

    print(f"  → Tipo detectado: {type(obj)}")
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, np.ndarray):
        # # ndarray estructurado: tiene nombres en dtype.names
        # if obj.dtype.names:
        #     cols = list(obj.dtype.names)
        # else:
        #     # ndarray genérico: asigna nombres col0, col1, ...
        #     #cols = [f"col{i}" for i in range(obj.shape[1])]
        df = pd.DataFrame(obj, columns=col_names)
    else:
        raise ValueError(f"Tipo no soportado: {type(obj)}")
    return df

# 3) Carga e inspección
dfs = {}
for label, path in files.items():
    print(f"Cargando {label} desde {path} ...")
    df = load_as_df(path)
    print(f"  → shape: {df.shape}")
    print(f"  → columnas: {list(df.columns)}")
    print(f"  → nulos:\n{df.isna().sum()}\n")
    dfs[label] = df

# 4) Variables de interés (excluye ID si existe)
variables = [c for c in dfs['FullMC_pre'].columns if 'id' not in c.lower()]

# 5) Estadísticas descriptivas
for label, df in dfs.items():
    stats = df[variables].describe().T
    stats.to_csv(os.path.join(out_dir, f"{label}_stats.csv"))
    print(f"Estadísticas de {label} guardadas en {label}_stats.csv")

# 6) Hístogramas y KS-test
for var in variables:
    # Pre-Shield
    plt.figure()
    sns.histplot(dfs['FullMC_pre'][var], stat='density', bins=80, alpha=0.5, label='FullMC_pre')
    sns.histplot(dfs['GAN_pre'][var],    stat='density', bins=80, alpha=0.5, label='GAN_pre')
    plt.legend(); plt.title(f"{var} — Pre-Shield")
    plt.savefig(os.path.join(out_dir, "figures", f"{var}_pre.png"))
    plt.close()
    # Post-Shield
    plt.figure()
    sns.histplot(dfs['FullMC_post'][var], stat='density', bins=80, alpha=0.5, label='FullMC_post')
    sns.histplot(dfs['GAN_post'][var],    stat='density', bins=80, alpha=0.5, label='GAN_post')
    plt.legend(); plt.title(f"{var} — Post-Shield")
    plt.savefig(os.path.join(out_dir, "figures", f"{var}_post.png"))
    plt.close()
    # KS-test
    ks1 = ks_2samp(dfs['FullMC_pre'][var],  dfs['GAN_pre'][var])
    ks2 = ks_2samp(dfs['FullMC_post'][var], dfs['GAN_post'][var])
    print(f"{var}: KS_pre = {ks1.statistic:.3f} (p={ks1.pvalue:.1e}), "
          f"KS_post = {ks2.statistic:.3f} (p={ks2.pvalue:.1e})")

print("\n¡EDA completado! Revisa la carpeta 'eda_results' para estadísticas y figuras.")
