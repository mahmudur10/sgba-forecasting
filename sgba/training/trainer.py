"""Unified training loop and evaluator.

Matches the notebook's train_sgba_control function:
- 100 epochs, no early stopping
- Adam optimizer, lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- Gradient clipping (max norm 1.0)
- MSE loss
- Best validation loss checkpoint selection
- Checkpoint resume support
"""

from __future__ import annotations

import os
import time
import math
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sgba.training.protocol import TrainingConfig
from sgba.utils.seeds import set_seed


class Trainer:
    """Unified training loop applying the paper's protocol.

    Supports both geometry-based models (SGBA, SGBAControl) that take
    (x, hour, elevation) and simple baselines (PatchTST, DLinear, etc.)
    that take (x,) only.
    """

    def __init__(self, config: TrainingConfig | None = None,
                 device: str | None = None,
                 checkpoint_dir: str | Path | None = None):
        self.config = config or TrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        data_root: str | Path | None = None,
        site_key: str | None = None,
        exp_id: str | None = None,
        use_geometry: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Train the model and return training history.

        Args:
            model: the model to train
            train_loader: training DataLoader yielding (x, y, hour, elev)
            val_loader: validation DataLoader
            data_root: path to processed data (for compute_metrics)
            site_key: site identifier (e.g. 'site7')
            exp_id: experiment ID for checkpointing
            use_geometry: if True, model takes (x, hour, elev); if False, takes (x,)
            verbose: print progress every 10 epochs

        Returns:
            dict with 'history', 'best_epoch', 'best_val_loss', 'model'
        """
        cfg = self.config
        model = model.to(self.device)

        # Checkpoint paths
        best_path = None
        resume_path = None
        if self.checkpoint_dir and exp_id:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_path = self.checkpoint_dir / f'{exp_id}_best.pt'
            resume_path = self.checkpoint_dir / f'{exp_id}_resume.pt'

        start_epoch = 0
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
        best_val = float('inf')
        best_epoch = -1
        best_state = None

        # Resume support
        if resume_path and resume_path.exists():
            try:
                ck = torch.load(resume_path, map_location=self.device, weights_only=False)
                if ck.get('epochs') == cfg.epochs:
                    model.load_state_dict(ck['model_state'])
                    start_epoch = ck['epoch'] + 1
                    history = ck['history']
                    best_val = ck['best_val_loss']
                    best_epoch = ck['best_epoch']
                    if verbose:
                        print(f"  Resumed at epoch {start_epoch}/{cfg.epochs}")
            except Exception:
                pass

        if start_epoch >= cfg.epochs:
            if best_path and best_path.exists():
                bk = torch.load(best_path, map_location=self.device, weights_only=False)
                model.load_state_dict(bk['model_state'])
                best_val = bk.get('val_loss', best_val)
                best_epoch = bk.get('epoch', best_epoch)
            return {'model': model, 'history': history, 'best_epoch': best_epoch,
                    'best_val_loss': best_val, 'n_params': sum(p.numel() for p in model.parameters())}

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.lr_min)
        criterion = nn.MSELoss()

        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Training '{exp_id or 'model'}' | {cfg.epochs} epochs | params: {n_params:,}")

        t0 = time.time()

        for epoch in range(start_epoch, cfg.epochs):
            # TRAIN
            model.train()
            losses = []
            for batch in train_loader:
                x_b, y_b = batch[0], batch[1]
                x_b = x_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)

                if use_geometry:
                    hour_b = batch[2].to(self.device, non_blocking=True)
                    elev_b = batch[3].to(self.device, non_blocking=True)
                    pred = model(x_b, hour_b, elev_b)
                else:
                    pred = model(x_b)

                optimizer.zero_grad()
                loss = criterion(pred.squeeze(-1), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
                optimizer.step()
                losses.append(loss.item())

            avg_tr = float(np.mean(losses))

            # VALIDATE
            model.eval()
            vlosses = []
            with torch.no_grad():
                for batch in val_loader:
                    x_b, y_b = batch[0], batch[1]
                    x_b = x_b.to(self.device, non_blocking=True)
                    y_b = y_b.to(self.device, non_blocking=True)

                    if use_geometry:
                        hour_b = batch[2].to(self.device, non_blocking=True)
                        elev_b = batch[3].to(self.device, non_blocking=True)
                        pred = model(x_b, hour_b, elev_b)
                    else:
                        pred = model(x_b)

                    vlosses.append(criterion(pred.squeeze(-1), y_b).item())

            avg_va = float(np.mean(vlosses))
            history['train_loss'].append(avg_tr)
            history['val_loss'].append(avg_va)
            scheduler.step(avg_va)

            if avg_va < best_val:
                best_val = avg_va
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if best_path:
                    torch.save({'model_state': best_state, 'epoch': epoch,
                                'val_loss': best_val, 'exp_id': exp_id}, best_path)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1):
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1:3d}/{cfg.epochs}: tr={avg_tr:.5f} va={avg_va:.5f} "
                      f"best={best_val:.5f}@{best_epoch+1} ({elapsed:.0f}s)")

            # Checkpoint every 5 epochs
            if resume_path and ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1):
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': epoch, 'epochs': cfg.epochs,
                    'history': history, 'best_val_loss': best_val,
                    'best_epoch': best_epoch
                }, resume_path)

        train_time = time.time() - t0

        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose:
            print(f"  Done in {train_time:.0f}s | best val {best_val:.6f} @ epoch {best_epoch+1}")

        return {'model': model, 'history': history, 'best_epoch': best_epoch,
                'best_val_loss': best_val, 'train_time': train_time,
                'n_params': sum(p.numel() for p in model.parameters())}

    def evaluate(self, model: nn.Module, loader: DataLoader,
                 data_root: str | Path | None = None,
                 site_key: str | None = None,
                 use_geometry: bool = True) -> dict:
        """Evaluate model on a loader; returns metrics + predictions."""
        model = model.to(self.device)
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in loader:
                x_b, y_b = batch[0], batch[1]
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)

                if use_geometry:
                    hour_b = batch[2].to(self.device)
                    elev_b = batch[3].to(self.device)
                    p = model(x_b, hour_b, elev_b)
                else:
                    p = model(x_b)

                preds.append(p.squeeze(-1).cpu().numpy())
                trues.append(y_b.cpu().numpy())

        yp = np.concatenate(preds)
        yt = np.concatenate(trues)

        # Compute metrics if we have site info
        metrics = {}
        if data_root and site_key:
            from sgba.data.datasets import compute_metrics
            metrics = compute_metrics(yt, yp, site_key, data_root)

        # Denormalize if possible
        y_pred_kw = yp
        y_true_kw = yt
        if data_root and site_key:
            from sgba.data.datasets import load_site_normalization
            norm = load_site_normalization(site_key, data_root)
            if norm:
                ps, pm = norm['stds']['power'], norm['means']['power']
                y_pred_kw = yp * ps + pm
                y_true_kw = yt * ps + pm

        return {'metrics': metrics,
                'y_pred_kw': y_pred_kw,
                'y_true_kw': y_true_kw,
                'y_pred_norm': yp,
                'y_true_norm': yt}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig | None = None,
    device: str | None = None,
    data_root: str | Path | None = None,
    site_key: str | None = None,
    exp_id: str | None = None,
    use_geometry: bool = True,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = False,
) -> dict:
    """Convenience function: train, evaluate, optionally checkpoint."""
    trainer = Trainer(config=config, device=device, checkpoint_dir=checkpoint_dir)
    history = trainer.train(model, train_loader, val_loader, data_root=data_root,
                           site_key=site_key, exp_id=exp_id,
                           use_geometry=use_geometry, verbose=verbose)
    eval_results = trainer.evaluate(model, test_loader, data_root=data_root,
                                    site_key=site_key, use_geometry=use_geometry)
    return {
        'model': model,
        'history': history['history'],
        'best_epoch': history['best_epoch'],
        'best_val_loss': history['best_val_loss'],
        'metrics': eval_results['metrics'],
        'predictions': eval_results['y_pred_kw'],
        'errors': eval_results['y_pred_kw'] - eval_results['y_true_kw'],
        'train_time_sec': history.get('train_time', 0),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | None = None,
    data_root: str | Path | None = None,
    site_key: str | None = None,
    use_geometry: bool = True,
) -> dict:
    """Evaluate only — no training. Returns predictions + metrics."""
    trainer = Trainer(device=device)
    return trainer.evaluate(model, loader, data_root=data_root,
                            site_key=site_key, use_geometry=use_geometry)
