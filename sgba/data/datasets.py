"""PyTorch datasets for PV forecasting — matches notebook data format.

Data layout (from Kaggle/Colab notebooks):
    data/processed/dkasc/{site7,site10,site2}/
        X_{train,val,test}.npy    — (N, 24, 5) z-scored input features
        y_{train,val,test}.npy    — (N,) z-scored power target
        hour_{train,val,test}.npy — (N, 24) hour-of-day [0, 24)
        elev_{train,val,test}.npy — (N, 24) solar elevation in degrees
        ts_{train,val,test}.npy   — (N,) timestamps (for multi-horizon lookup)
        normalization_params.json — {means: {...}, stds: {...}}
        split_info.json           — {n_train, n_val, n_test}

    data/processed/pvgis_{singapore,oslo,madrid}/
        Same layout but with 4 channels (no DHI) and different stats files

    configs/geometry_normalization_stats.json
        {elevation_stats: {site: {mean, std}},
         ghi_envelope_stats: {site: {mean, std}}}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Standard channel ordering used across all datasets.
CHANNEL_NAMES = ["power", "temp", "humidity", "ghi", "dhi"]


class PVDatasetV2(Dataset):
    """PV forecasting dataset returning (x, y, hour, elevation).

    Reads pre-processed .npy files from a site directory.
    """
    def __init__(self, site_dir, split='train'):
        self.site_dir = Path(site_dir)
        self.X = np.load(self.site_dir / f'X_{split}.npy')
        self.y = np.load(self.site_dir / f'y_{split}.npy')
        self.hour = np.load(self.site_dir / f'hour_{split}.npy')
        self.elev = np.load(self.site_dir / f'elev_{split}.npy')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.hour[idx], dtype=torch.float32),
            torch.tensor(self.elev[idx], dtype=torch.float32),
        )


class MultiHorizonDataset(Dataset):
    """Dataset for multi-horizon forecasting. Reuses existing X arrays with custom y targets."""
    def __init__(self, X, y_h, hour, elev):
        self.X = X
        self.y = y_h
        self.hour = hour
        self.elev = elev

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.hour[idx], dtype=torch.float32),
            torch.tensor(self.elev[idx], dtype=torch.float32),
        )


def build_multihorizon_targets(ts, y, horizon):
    """Build h-step-ahead targets by timestamp lookup.

    For window i with h=1 target at ts[i]:
      h-step target = y[j] where ts[j] = ts[i] + (horizon - 1) hours

    Windows where the target timestamp doesn't exist (nighttime gap
    or out of range) are dropped.

    Returns: (y_h, valid_mask) where valid_mask[i]=True means window i
    has a valid h-step target.
    """
    ts_pd = pd.to_datetime(ts)
    ts_to_idx = {}
    for i, t in enumerate(ts_pd):
        ts_to_idx[t] = i

    y_h = np.full(len(y), np.nan, dtype=np.float32)
    for i, t in enumerate(ts_pd):
        target_ts = t + pd.Timedelta(hours=horizon - 1)
        if target_ts in ts_to_idx:
            j = ts_to_idx[target_ts]
            y_h[i] = y[j]

    valid_mask = ~np.isnan(y_h)
    return y_h, valid_mask


def create_dataloaders(site_key, data_root, batch_size=128, num_workers=2, seed=42):
    """Create train/val/test loaders for a DKASC or PVGIS site.

    Args:
        site_key: 'site7', 'site10', 'site2', 'singapore', 'oslo', 'madrid'
        data_root: Path to the data root (e.g. 'data/processed')
        batch_size: batch size
        num_workers: number of DataLoader workers
        seed: random seed for shuffling

    Returns:
        train_loader, val_loader, test_loader, norm_stats
        norm_stats = {
            'ghi_envelope_mean': float,
            'ghi_envelope_std': float,
            'elev_mean': float,
            'elev_std': float,
        }
    """
    data_root = Path(data_root)

    # Determine site directory (DKASC sites are under dkasc/, PVGIS under pvgis_)
    if site_key.startswith('site'):
        site_dir = data_root / 'dkasc' / site_key
    else:
        site_dir = data_root / f'pvgis_{site_key}'

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_ds = PVDatasetV2(site_dir, 'train')
    val_ds = PVDatasetV2(site_dir, 'val')
    test_ds = PVDatasetV2(site_dir, 'test')

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Load normalization stats from configs/geometry_normalization_stats.json
    norm_path = data_root.parent / 'configs' / 'geometry_normalization_stats.json'
    if not norm_path.exists():
        # Fallback: try data_root / 'configs' / ...
        norm_path = data_root / 'configs' / 'geometry_normalization_stats.json'

    if norm_path.exists():
        with open(norm_path) as f:
            norm_data = json.load(f)

        # Handle both site7 and dkasc_site7 key formats
        elev_key = site_key if site_key in norm_data.get('elevation_stats', {}) else f'dkasc_{site_key}'
        ghi_key = site_key if site_key in norm_data.get('ghi_envelope_stats', {}) else f'dkasc_{site_key}'

        norm_stats = {
            'ghi_envelope_mean': norm_data['ghi_envelope_stats'].get(ghi_key, {}).get('mean', 0.0),
            'ghi_envelope_std': norm_data['ghi_envelope_stats'].get(ghi_key, {}).get('std', 1.0),
            'elev_mean': norm_data['elevation_stats'].get(elev_key, {}).get('mean', 0.0),
            'elev_std': norm_data['elevation_stats'].get(elev_key, {}).get('std', 1.0),
        }
    else:
        # Fallback: compute from training data
        print(f"Warning: geometry_normalization_stats.json not found at {norm_path}")
        print(f"  Computing stats from training data elevation...")
        elev_train = np.load(site_dir / 'elev_train.npy')
        norm_stats = {
            'ghi_envelope_mean': 0.0,
            'ghi_envelope_std': 1.0,
            'elev_mean': float(elev_train.mean()),
            'elev_std': float(elev_train.std() + 1e-8),
        }

    return train_loader, val_loader, test_loader, norm_stats


def load_site_normalization(site_key, data_root):
    """Load normalization_params.json for a site.

    Returns:
        dict with 'means' and 'stds' for each channel.
    """
    data_root = Path(data_root)
    if site_key.startswith('site'):
        site_dir = data_root / 'dkasc' / site_key
    else:
        site_dir = data_root / f'pvgis_{site_key}'

    norm_path = site_dir / 'normalization_params.json'
    if norm_path.exists():
        with open(norm_path) as f:
            return json.load(f)
    return None


def compute_metrics(y_true_norm, y_pred_norm, site_key, data_root):
    """All metrics in original kW scale using site-specific normalization.

    Args:
        y_true_norm: normalized true values
        y_pred_norm: normalized predicted values
        site_key: site identifier (e.g. 'site7')
        data_root: Path to processed data root

    Returns:
        dict with RMSE, MAE, R2, MBE, n
    """
    import math

    norm = load_site_normalization(site_key, data_root)
    if norm is None:
        # Fallback: return normalized metrics
        y_true = np.asarray(y_true_norm).flatten()
        y_pred = np.asarray(y_pred_norm).flatten()
    else:
        pm, ps = norm['means']['power'], norm['stds']['power']
        y_true = np.asarray(y_true_norm) * ps + pm
        y_pred = np.asarray(y_pred_norm) * ps + pm
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mbe = np.mean(y_pred - y_true)

    return {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2),
            'MBE': float(mbe), 'n': int(len(y_true))}
