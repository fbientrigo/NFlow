# NFlow

Normalizing Flows for high-/non-high-impact (HI/non-HI) muon distributions with a **modular training** API (N/G/C/J losses), single-run training, and A/B testing.


## Features

- **RealNVP** flow with optional bounded-data bijector (LogitTransform).
- **Modular losses**:
  - `N` = Negative log-likelihood
  - `G` = Guided latent regularization for lvl2 mask (two Gaussian scales)
  - `C` = Correlation-structure matching
  - `J` = Jacobian regularization (stabilizes flow log-det)
- **Training modes**:
  - `deepflow.py`: single training run + full artifact snapshot
  - `deepflow_abtest.py`: grid A/B testing across loss configs
- **Reproducible outputs** per run: checkpoint, scaler, training history/plots, smoke metrics (A1/A2), and manifest
- **Diagnostics**: KS/Wasserstein/Q–Q, log-det stats, and “smoke” checks to catch failure modes early


## Requirements

- Python 3.9+ (tested on 3.10/3.11)
- PyTorch (CUDA optional)
- NumPy, matplotlib, scikit-learn
- (Optional) TensorBoard and pandas for richer logs/tables

Install (example):
```bash
pip install torch numpy matplotlib scikit-learn tensorboard pandas
```


## Repository Structure (key parts)

```
.
├── deepflow.py                    # single-run trainer
├── deepflow_abtest.py             # A/B testing runner
├── config.yaml                    # config for deepflow.py
├── config_ab.yaml                 # config for deepflow_abtest.py
└── utils/
    ├── flow_models.py             # RealNVP + bijector
    ├── training.py                # modular training loop + plotting
    ├── metrics.py                 # KS/W1/Q–Q, smoke reports, etc.
    ├── data_handling.py           # loaders + guided dataloaders
    ├── run_management.py          # run folders, counters
    ├── logging_config.py          # file/console logging
    └── synth.py                   # (optional) synthetic data helpers
```


## Data

The scripts expect an HDF5 file with “lvl1” (mother) features. Set the path via `datapath_mothers` in the config.
For quick experiments you can generate a synthetic dataset using the utilities in `utils/synth.py` (see the Colab linked at the end for an example).

> **Dimensionality**: the loader infers `input_dim` from the HDF5 dataset.
> **Guided mask**: lvl2 selection is defined via momentum quantiles `q_pz` and `q_pt` (configurable).


## Quick Start — Single Run

1. Edit `config.yaml`:

```yaml
run_base_dir: outputs
device: "cuda"          # or "cpu"
batch_size: 2048
datapath_mothers: "mom_lvl1.h5"
data:
  val_ratio: 0.2
  q_pz: 0.8
  q_pt: 0.8
model:
  name: "RealNVP_Flow"
  hidden_dim: 200
  n_layers: 20
training:
  learning_rate: 0.00004
  epochs: 300
  patience: 10
  weight_decay: 0.0
  # modular loss:
  loss_keys: "NGCJ"
  loss_weights: { N: 0.55, G: 0.20, C: 0.15, J: 0.10 }
  loss_cfg:
    G: { sigma1: 1.0, sigma2: 0.35, lambda1: 0.10, lambda2: 0.30 }
    J: { alpha: 1.0e-3, beta: 1.0e-3 }
  # optional warmup→hold schedule for G.lambda2:
  schedulers: { lambda2: "cosine_hold" }
  scheduler_cfg:
    lambda2: { max_v: 0.7, warmup_frac: 0.5, start_frac: 0.2 }
logging:
  level: "INFO"
  file: "logs/project.log"
```

2. Run:

```bash
python deepflow.py --config config.yaml
```

3. Outputs (per run) in `outputs/run_XXXX/`:

* `*_final.pt` — CPU `state_dict` checkpoint
* `*_scaler.pkl` — scaler for inverse transform
* `*_history.(npz|json)` — per-epoch curves and parts (if returned)
* `*_loss_curves.png` — training/validation total and parts + ld stats
* `*_smoke_metrics.(csv|json)` — A1/A2 snapshot (tails & log-det)
* `*_loss_cfg.json` — exact loss keys/weights/config used
* `manifest.json` — reproducibility info (versions, files, device)
* `config.yaml` — the config copied to the run folder

> **TensorBoard**: if enabled in the training code, run `tensorboard --logdir outputs` and open the provided URL (I haven't gone to try if it works yet)


## A/B Testing

Use `deepflow_abtest.py` with a declarative grid in `config_ab.yaml`:

```yaml
run_base_dir: outputs_ab
device: "cuda"
batch_size: 2048
datapath_mothers: "mom_lvl1.h5"
data: { val_ratio: 0.2, q_pz: 0.8, q_pt: 0.8 }
model: { name: "RealNVP_Flow", hidden_dim: 200, n_layers: 20 }
training: { learning_rate: 0.00004, epochs: 60, patience: 8, weight_decay: 0.0 }

ab:
  combos:
    - keys: "N"
    - keys: "NC"
      C: { n_samples: 1024 }
    - keys: "NG"
      G:
        sigma1: 1.0
        lambda1: 0.5
        lambda2: 0.7
        grid:
          sigma2: [0.30, 0.35]
          lambda2: [0.5, 0.7]
    - keys: "NGC"
      G:
        sigma1: 1.0
        lambda1: 0.5
        lambda2: 0.7
        grid:
          sigma2: [0.30, 0.35]
          lambda2: [0.5, 0.7]
      C: { n_samples: 1024 }
```

Run:

```bash
python deepflow_abtest.py --config config_ab.yaml
```

What you get in `outputs_ab/run_XXXX/`:

* `ab_results.json` — list of all trials (tag, keys, cfg, val loss, time, checkpoint path)
* `ab_summary.json` — best trial summary with paths
* `*_BEST.pt` — checkpoint of the best model + `*_history.npz/png`, `*_best_loss_cfg.json`, `*_smoke_metrics.json`
* `config_ab.yaml` — the AB config copied to the run folder

> **Selection metric**: last `val_total` loss by default (consistent a

## Restoring a Model

```python
import torch
from utils.flow_models import NormalizingFlow

# Rebuild the same architecture
model = NormalizingFlow(input_dim=..., hidden_dim=..., n_layers=...)
state = torch.load("outputs/run_XXXX/RealNVP_Flow_final.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

If you saved the scaler:

```python
import pickle
with open("outputs/run_XXXX/RealNVP_Flow_scaler.pkl","rb") as f:
    scaler = pickle.load(f)
```


## Interpreting Diagnostics (at a glance)

* **Smoke metrics**:

  * `A1`: tail mass in pre-bijector latents and saturation ratio in post-bijector x; large values can signal instability.
  * `A2`: stats of flow-only log-det (`ld_flow mean/std/min/max`); exploding variance often correlates with bad samples.
* **KS/Wasserstein**: 1D marginal comparisons per feature; lower is better. p-values are permutation-based (bootstrapped).
* **Q–Q plots**: deviations from the diagonal reveal quantile mismatches (especially in tails).


## Tips

* Start with `loss_keys: "N"` to sanity-check NLL convergence.
* Add `G` gently: small `lambda2`, **warmup→hold** scheduler to get a “U-shape” schedule that stops growing after warmup.
* Add `J` (Jacobian reg) with small `alpha/beta` if you observe spiky log-det or tail blow-ups.
* Use AB testing to sweep `sigma2` and `lambda2` for `G`.



## Google Colab (reference notebook)

A companion Colab notebook that demonstrates the full pipeline (data synth, training, metrics, A/B testing) is available here:

▶ [https://colab.research.google.com/drive/1waxd59e1sKGU-1elieIT\_lu7DxP9bXm3?usp=sharing](https://colab.research.google.com/drive/1waxd59e1sKGU-1elieIT_lu7DxP9bXm3?usp=sharing)


