import os
import torch
import yaml
import os, json, pickle, math, time, datetime as dt

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _to_cpu_state_dict(model):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}

def _jsonable(obj):
    # convierte estructuras con callables / tensores a strings/valores básicos
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if callable(obj):
        return getattr(obj, "__name__", "callable")
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj