#!/usr/bin/env python3
# deepflow.py
"""
Single-run training entrypoint for the modular RealNVP flow (N/G/C/J).
- Loads config (YAML) and sets up run directory & logging
- Builds dataloaders (guided)
- Trains with train_model_modular (utils.training)
- Saves: checkpoint (CPU), scaler, history (npz/json), loss plot, smoke metrics (csv/json),
         loss config used, manifest; y deja todo en run_dir
"""

from __future__ import annotations
import argparse, os, sys, json, shutil, time, datetime as dt
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

# --- repo utils ---
from utils.config import load_config
from utils.data_handling import prepare_guided_dataloaders
from utils.flow_models import NormalizingFlow
from utils.logging_config import setup_logging
from utils.run_management import get_next_attempt_number
from utils.training import train_model_modular, plot_loss_history
from utils.metrics import post_training_smoke_report

# ---------------------------
# Helper: schedulers (capped)
# ---------------------------
def make_cosine_hold(max_v: float = 0.7, warmup_frac: float = 0.5, start_frac: float = 0.0):
    """Cosine warmup to max_v, then hold."""
    import math
    def sched(e: int, E: int) -> float:
        t = e / float(E)
        if t <= start_frac:
            return 0.0
        a = (t - start_frac) / max(1e-8, warmup_frac)
        a = max(0.0, min(1.0, a))
        return max_v * (0.5 - 0.5 * math.cos(math.pi * a))
    return sched

# ---------------------------
# Save artifacts (reusable)
# ---------------------------
def _to_cpu_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(v) for v in obj]
    if callable(obj): return getattr(obj, "__name__", "callable")
    if torch.is_tensor(obj): return obj.detach().cpu().tolist()
    return obj

def save_artifacts(*,
                   model: torch.nn.Module,
                   history: Optional[Dict[str, list]],
                   scaler_mother,
                   loss_keys: Optional[str],
                   loss_weights: Optional[Dict[str, float]],
                   loss_cfg: Optional[Dict[str, dict]],
                   run_dir: str,
                   artifact_stem: str,
                   val_loader) -> Dict[str, str]:
    import pickle
    import matplotlib.pyplot as plt

    os.makedirs(run_dir, exist_ok=True)
    paths = {
        "ckpt":          os.path.join(run_dir, f"{artifact_stem}_final.pt"),
        "scaler":        os.path.join(run_dir, f"{artifact_stem}_scaler.pkl"),
        "history_npz":   os.path.join(run_dir, f"{artifact_stem}_history.npz"),
        "history_json":  os.path.join(run_dir, f"{artifact_stem}_history.json"),
        "loss_plot":     os.path.join(run_dir, f"{artifact_stem}_loss_curves.png"),
        "metrics_csv":   os.path.join(run_dir, f"{artifact_stem}_smoke_metrics.csv"),
        "metrics_json":  os.path.join(run_dir, f"{artifact_stem}_smoke_metrics.json"),
        "loss_cfg":      os.path.join(run_dir, f"{artifact_stem}_loss_cfg.json"),
        "manifest":      os.path.join(run_dir, f"{artifact_stem}_manifest.json"),
    }

    # 1) checkpoint (CPU)
    torch.save(_to_cpu_state_dict(model), paths["ckpt"])

    # 2) scaler
    if scaler_mother is not None:
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler_mother, f)

    # 3) history + plot
    if isinstance(history, dict) and history:
        np.savez_compressed(paths["history_npz"], **{k: np.asarray(v) for k, v in history.items() if hasattr(v, "__len__")})
        with open(paths["history_json"], "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items() if isinstance(v, (list, tuple))}, f, indent=2)
        try:
            fig = plot_loss_history(history, ids=list(history.get("train_parts", {}).keys()))
            if fig is not None:
                fig.savefig(paths["loss_plot"], bbox_inches="tight", dpi=150)
                plt.close(fig)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not save loss plot: %s", e)

    # 4) smoke metrics snapshot (A1/A2) + CSV/JSON
    try:
        metrics_dict, table = post_training_smoke_report(
            model=model,
            val_loader=val_loader,
            history=history,
            plot_loss_history_fn=plot_loss_history,
            sample_N=None,
            eps=None,
            device=None,
            return_table="pandas",
        )
        try:
            table.to_csv(paths["metrics_csv"], index=False)
        except Exception:
            import csv
            with open(paths["metrics_csv"], "w", newline="") as f:
                w = csv.writer(f); w.writerow(["stat","value"])
                for k,v in metrics_dict.items(): w.writerow([k, v])
        with open(paths["metrics_json"], "w") as f:
            json.dump(metrics_dict, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning("Smoke metrics failed: %s", e)

    # 5) loss config
    with open(paths["loss_cfg"], "w") as f:
        json.dump(_jsonable({"loss_keys": loss_keys, "loss_weights": loss_weights, "loss_cfg": loss_cfg}), f, indent=2)

    # 6) manifest
    manifest = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "artifact_stem": artifact_stem,
        "device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "checkpoint": paths["ckpt"],
        "scaler": paths["scaler"] if os.path.exists(paths["scaler"]) else None,
        "history_npz": paths["history_npz"] if os.path.exists(paths["history_npz"]) else None,
        "loss_plot": paths["loss_plot"] if os.path.exists(paths["loss_plot"]) else None,
        "metrics_csv": paths["metrics_csv"] if os.path.exists(paths["metrics_csv"]) else None,
        "metrics_json": paths["metrics_json"] if os.path.exists(paths["metrics_json"]) else None,
        "loss_cfg": paths["loss_cfg"],
        "config_yaml": os.path.join(run_dir, "config.yaml"),
    }
    with open(paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    return paths

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a modular RealNVP flow (single run).")
    ap.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    ap.add_argument("--no_tb", action="store_true", help="Disable TensorBoard writer.")
    args = ap.parse_args()

    # Load config (utils.config.load_config may accept path; fall back to cwd)
    try:
        config = load_config(args.config)
    except TypeError:
        config = load_config()

    # Setup run dir & logging
    run_base_dir = config.get("run_base_dir", "outputs")
    attempt_number, run_dir = get_next_attempt_number(run_base_dir)
    os.makedirs(run_dir, exist_ok=True)
    # copy config used
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
    setup_logging(config.get("logging", {}), run_dir)
    logger = logging.getLogger(__name__)
    logger.info("Run %d → %s", attempt_number, run_dir)

    # Device
    dev = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", dev)

    # Data
    from sklearn.preprocessing import QuantileTransformer
    q_pz = float(config.get("data", {}).get("q_pz", 0.8))
    q_pt = float(config.get("data", {}).get("q_pt", 0.8))
    batch_size = int(config.get("batch_size", 256))
    val_ratio  = float(config.get("data", {}).get("val_ratio", 0.2))
    train_loader, val_loader, scaler_mother = prepare_guided_dataloaders(
        path_lvl1=config.get("datapath_mothers"),
        batch_size=batch_size,
        q_pz=q_pz,
        q_pt=q_pt,
        test_size=val_ratio,
        random_state=42,
        plotdir=run_dir,
        scaler_mother=QuantileTransformer()
    )

    # Model
    input_dim  = next(iter(train_loader))[0].shape[1]
    model_cfg  = config.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 160))
    n_layers   = int(model_cfg.get("n_layers", 10))
    model = NormalizingFlow(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    model.to(dev)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Loss config (modular)
    train_cfg = config.get("training", {})
    epochs   = int(train_cfg.get("epochs", 300))
    lr       = float(train_cfg.get("learning_rate", 4e-4))
    patience = int(train_cfg.get("patience", 20))
    wd       = float(train_cfg.get("weight_decay", 0.0))

    loss_keys    = train_cfg.get("loss_keys", "N")
    loss_weights = train_cfg.get("loss_weights", None)
    loss_cfg     = train_cfg.get("loss_cfg", {})
    # Optional: named schedulers for G.lambda2_schedule
    sched_name   = (train_cfg.get("schedulers", {}) or {}).get("lambda2", None)
    if loss_cfg.get("G") and sched_name:
        if sched_name == "cosine_hold":
            scfg = train_cfg.get("scheduler_cfg", {}).get("lambda2", {})
            loss_cfg["G"]["lambda2_schedule"] = make_cosine_hold(
                max_v=float(scfg.get("max_v", 0.7)),
                warmup_frac=float(scfg.get("warmup_frac", 0.5)),
                start_frac=float(scfg.get("start_frac", 0.0)),
            )

    # TensorBoard writer (optional)
    writer = None
    if not args.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    # Train
    t0 = time.time()
    model_best, history = train_model_modular(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        writer=writer,
        device=str(dev),
        model_dir=run_dir,
        name_model=model_cfg.get("name", "RealNVP_Flow"),
        loss_keys=loss_keys,
        loss_weights=loss_weights,
        loss_cfg=loss_cfg,
        patience=patience,
        weight_decay=wd,
        trial=None,
        plot_every=0,
        return_history=True,
    )
    dt = time.time() - t0
    logger.info("Training complete in %.1fs", dt)

    # Save artifacts
    artifact_stem = model_cfg.get("name", "RealNVP_Flow")
    paths = save_artifacts(
        model=model_best,
        history=history,
        scaler_mother=scaler_mother,
        loss_keys=loss_keys,
        loss_weights=loss_weights,
        loss_cfg=loss_cfg,
        run_dir=run_dir,
        artifact_stem=artifact_stem,
        val_loader=val_loader,
    )
    logger.info("Artifacts saved: %s", json.dumps(paths, indent=2))

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
