#!/usr/bin/env python3
# deepflow_abtest.py
"""
A/B testing entrypoint for the modular RealNVP flow.
- Loads AB config describing combos and simple grids for G/C/J sublosses
- Trains each variant from the same initial weights, logs val loss
- Keeps a JSON/CSV of results and saves a checkpoint & artifacts for the BEST
"""

from __future__ import annotations
import argparse, os, sys, json, shutil, time
from copy import deepcopy
from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# repo utils
from utils.config import load_config
from utils.data_handling import prepare_guided_dataloaders
from utils.flow_models import NormalizingFlow
from utils.logging_config import setup_logging
from utils.run_management import get_next_attempt_number
from utils.training import train_model_modular, plot_loss_history
from utils.metrics import post_training_smoke_report

# reuse save_artifacts from deepflow.py (keep a local copy here)
def _to_cpu_state_dict(model): return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k,v in model.state_dict().items()}
def _jsonable(obj):
    if isinstance(obj, dict): return {str(k): _jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(v) for v in obj]
    if callable(obj): return getattr(obj, "__name__", "callable")
    if torch.is_tensor(obj): return obj.detach().cpu().tolist()
    return obj

def save_artifacts_best(*,
                        model: torch.nn.Module,
                        history: Dict[str, list],
                        scaler_mother,
                        loss_keys: str,
                        loss_weights: Dict[str, float] | None,
                        loss_cfg: Dict[str, dict] | None,
                        run_dir: str,
                        artifact_stem: str,
                        val_loader):
    import pickle, json, numpy as np, matplotlib.pyplot as plt
    from utils.metrics import post_training_smoke_report

    paths = {
        "ckpt":          os.path.join(run_dir, f"{artifact_stem}_BEST.pt"),
        "scaler":        os.path.join(run_dir, f"{artifact_stem}_scaler.pkl"),
        "history_npz":   os.path.join(run_dir, f"{artifact_stem}_history.npz"),
        "loss_plot":     os.path.join(run_dir, f"{artifact_stem}_loss_curves.png"),
        "results_json":  os.path.join(run_dir, f"{artifact_stem}_ab_results.json"),
        "best_cfg_json": os.path.join(run_dir, f"{artifact_stem}_best_loss_cfg.json"),
        "metrics_json":  os.path.join(run_dir, f"{artifact_stem}_smoke_metrics.json"),
    }
    os.makedirs(run_dir, exist_ok=True)
    torch.save(_to_cpu_state_dict(model), paths["ckpt"])
    if scaler_mother is not None:
        with open(paths["scaler"], "wb") as f: pickle.dump(scaler_mother, f)
    if isinstance(history, dict) and history:
        np.savez_compressed(paths["history_npz"], **{k: np.asarray(v) for k,v in history.items() if hasattr(v, "__len__")})
        try:
            fig = plot_loss_history(history, ids=list(history.get("train_parts", {}).keys()))
            if fig is not None:
                fig.savefig(paths["loss_plot"], bbox_inches="tight", dpi=150)
                plt.close(fig)
        except Exception: pass
    with open(paths["best_cfg_json"], "w") as f:
        json.dump(_jsonable({"loss_keys": loss_keys, "loss_weights": loss_weights, "loss_cfg": loss_cfg}), f, indent=2)
    try:
        metrics_dict, _ = post_training_smoke_report(model=model, val_loader=val_loader, history=history,
                                                     plot_loss_history_fn=plot_loss_history, return_table=None)
        with open(paths["metrics_json"], "w") as f:
            json.dump(metrics_dict, f, indent=2)
    except Exception: pass
    return paths

def expand_combos(base: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, dict]]]:
    """
    Input schema (see config_ab.yaml):
      ab.combos: list of dicts like:
        - keys: "N"
        - keys: "NG"
          G:
            sigma1: 1.0
            sigma2: 0.35
            lambda1: 0.5
            lambda2: 0.7
            grid: { sigma2: [0.30,0.35], lambda2: [0.5,0.7] }
        - keys: "NC"
          C: { n_samples: 1024 }
    Returns list of (tag, keys, per_loss_cfg)
    """
    out = []
    for d in base:
        keys = d["keys"]
        # collect per-loss base cfg (no grid)
        per = {k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in d.items() if k in ("G","C","J")}
        # build grid (cartesian product) if present for any subloss
        grids = []
        names = []
        for k, v in d.items():
            if k in ("G","C","J") and isinstance(v, dict) and "grid" in v and isinstance(v["grid"], dict):
                grid_spec = v["grid"]
                names.append((k, list(grid_spec.keys())))
                grids.append([ (k, param, val) for param, vals in grid_spec.items() for val in vals ])
        if not grids:
            tag = keys
            out.append((tag, keys, per))
        else:
            # product over parameter *choices* but we need to group by param; do a nested product manually
            # gather dict of lists for each subloss
            spec_lists: Dict[str, Dict[str, List[Any]]] = {}
            for k, v in d.items():
                if k in ("G","C","J") and isinstance(v, dict) and "grid" in v:
                    spec_lists[k] = {param: vals for param, vals in v["grid"].items()}
            # cartesian per subloss
            # build a list of dicts for each subloss combinations, then combine if multiple sublosses have grid
            per_sloss_choices: List[List[Tuple[str, Dict[str, Any]]]] = []
            for sloss, pmap in spec_lists.items():
                keys_params = list(pmap.keys())
                vals_lists  = [pmap[p] for p in keys_params]
                combos = []
                for tpl in product(*vals_lists):
                    combos.append( (sloss, dict(zip(keys_params, tpl))) )
                per_sloss_choices.append(combos)
            # global product across sublosses
            for tpl in product(*per_sloss_choices):
                cfg = {k: dict(per.get(k, {})) for k in per.keys()}
                tag_bits = [keys]
                for (sloss, upd) in tpl:
                    cfg.setdefault(sloss, {}).update(upd)
                    for kk, vv in upd.items():
                        tag_bits.append(f"{sloss}-{kk}{str(vv).replace('.','p')}")
                tag = "-".join(tag_bits)
                out.append((tag, keys, cfg))
    return out

def main():
    ap = argparse.ArgumentParser(description="A/B testing for modular RealNVP flow.")
    ap.add_argument("--config", default="config_ab.yaml", help="Path to AB YAML config.")
    ap.add_argument("--no_tb", action="store_true", help="Disable TensorBoard writer for inner runs.")
    args = ap.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except TypeError:
        from pathlib import Path
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Setup run dir & logging
    run_base_dir = config.get("run_base_dir", "outputs_ab")
    attempt_number, run_dir = get_next_attempt_number(run_base_dir)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "config_ab.yaml"))
    setup_logging(config.get("logging", {}), run_dir)
    logger = logging.getLogger(__name__)
    logger.info("AB Run %d → %s", attempt_number, run_dir)

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

    # Model template & initial weights
    input_dim  = next(iter(train_loader))[0].shape[1]
    model_cfg  = config.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 160))
    n_layers   = int(model_cfg.get("n_layers", 10))
    base_model = NormalizingFlow(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(dev)
    init_sd    = deepcopy(base_model.state_dict())

    # AB space
    ab_cfg = config.get("ab", {})
    combos_cfg = ab_cfg.get("combos", [{"keys":"N"}])
    combos = expand_combos(combos_cfg)
    logger.info("Expanded to %d combos.", len(combos))

    # Train hyperparams
    train_cfg = config.get("training", {})
    epochs   = int(train_cfg.get("epochs", 60))
    lr       = float(train_cfg.get("learning_rate", 4e-4))
    patience = int(train_cfg.get("patience", 10))
    wd       = float(train_cfg.get("weight_decay", 0.0))

    results: List[Dict[str, Any]] = []
    best_val = float("inf")
    best = None

    # Loop over combos
    for tag, keys, per_loss_cfg in combos:
        model = NormalizingFlow(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(dev)
        model.load_state_dict(init_sd)

        t0 = time.time()
        model_tmp, history = train_model_modular(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            writer=None if args.no_tb else None,  # keep None to avoid TB overhead
            device=str(dev),
            model_dir=run_dir,
            name_model=f"{model_cfg.get('name','RealNVP_Flow')}_{tag}",
            loss_keys=keys,
            loss_weights=None,
            loss_cfg=per_loss_cfg,
            patience=patience,
            weight_decay=wd,
            trial=None,
            plot_every=0,
            return_history=True,
        )
        dt = time.time() - t0
        val_loss = float(history["val_total"][-1])
        info = {
            "tag": tag, "loss_keys": keys, "val_loss": val_loss, "time_s": dt,
            "epochs": epochs, "cfg": per_loss_cfg,
            "checkpoint": os.path.join(run_dir, f"{model_cfg.get('name','RealNVP_Flow')}_{tag}.pt")
        }
        results.append(info)
        logger.info("✔ %s: val=%.4f time=%.1fs", tag, val_loss, dt)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best = (model_tmp, history, keys, per_loss_cfg, info)

    # Save ranking
    results_path = os.path.join(run_dir, "ab_results.json")
    with open(results_path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)

    # Save best artifacts
    if best is not None:
        best_model, best_hist, best_keys, best_cfg, best_info = best
        paths = save_artifacts_best(
            model=best_model,
            history=best_hist,
            scaler_mother=scaler_mother,
            loss_keys=best_keys,
            loss_weights=None,
            loss_cfg=best_cfg,
            run_dir=run_dir,
            artifact_stem=model_cfg.get("name","RealNVP_Flow"),
            val_loader=val_loader,
        )
        # write small summary
        summary = {
            "best_tag": best_info["tag"],
            "best_val": best_info["val_loss"],
            "best_cfg": _jsonable(best_cfg),
            "paths": paths,
            "results_json": results_path,
        }
        with open(os.path.join(run_dir, "ab_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
    else:
        print("No best model (empty AB space?).")

if __name__ == "__main__":
    main()
