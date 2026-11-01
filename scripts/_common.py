"""Shared utilities for experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sgba.training.protocol import TrainingConfig
from sgba.utils.seeds import set_seed, get_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_dirs():
    for d in [RESULTS_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2, default=str)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def parse_args():
    args = {}
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                args[key] = sys.argv[i + 1]
            else:
                args[key] = True
    return args


def setup_experiment(seed: int = 42, device: str | None = None):
    set_seed(seed)
    ensure_dirs()
    dev = get_device() if device is None else torch.device(device)
    return dev


def load_result_json(site, model, seed, results_dir=None):
    """Load a result JSON from results directory."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    path = results_dir / f'{site}_{model}_seed{seed}_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_metric(site, model, seed, metric='RMSE', results_dir=None):
    """Get a metric value from result JSON."""
    r = load_result_json(site, model, seed, results_dir)
    if r is None:
        return None
    tm = r.get('test_metrics', {})
    for key in [metric, metric.upper(), metric.lower()]:
        if key in tm:
            return tm[key]
        if key in r:
            return r[key]
    return None


def make_synthetic_dataset(n_days=200, site='dkasc_site7', seed=42):
    """Create a synthetic PV dataset for quick testing without DKASC data."""
    rng = np.random.default_rng(seed)
    n_hours = n_days * 12
    ts = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    hour = ts.hour + ts.minute / 60.0
    elevation = np.maximum(0, 60 * np.sin(np.pi * (hour - 6) / 12))
    power = 5.0 * (elevation / 60.0) * (1 + 0.1 * rng.standard_normal(n_hours))
    power = np.maximum(0, power)
    ghi = 1000.0 * (elevation / 60.0) * (1 + 0.05 * rng.standard_normal(n_hours))
    ghi = np.maximum(0, ghi)
    dhi = 0.15 * ghi
    temp = 25 + 10 * np.sin(2 * np.pi * ts.dayofyear / 365.0) + rng.standard_normal(n_hours) * 2
    hum = 50 + 20 * np.sin(2 * np.pi * ts.dayofyear / 365.0 + np.pi) + rng.standard_normal(n_hours) * 5
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": ts,
        "power": power,
        "temp": temp,
        "humidity": np.clip(hum, 0, 100),
        "ghi": ghi,
        "dhi": dhi,
        "elevation": elevation,
    })
    return df


def split_chronological(df, train=0.7, val=0.15):
    import pandas as pd
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_val].copy(), df.iloc[n_train + n_val:].copy()
