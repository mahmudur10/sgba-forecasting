"""Evaluation metrics: RMSE, MAE, R^2, MAPE, MBE.

Statistical tests:
  - Diebold-Mariano test with Newey-West HAC standard errors
  - Block bootstrap confidence intervals on RMSE difference
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


# ============================================================
# Basic forecasting metrics
# ============================================================


def _to_numpy(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    return y_pred.flatten(), y_true.flatten()


def rmse(y_pred, y_true) -> float:
    y_pred, y_true = _to_numpy(y_pred, y_true)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred, y_true) -> float:
    y_pred, y_true = _to_numpy(y_pred, y_true)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_pred, y_true) -> float:
    y_pred, y_true = _to_numpy(y_pred, y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_pred, y_true, eps: float = 0.1) -> float:
    """MAPE computed only on samples with |y| > eps (paper convention)."""
    y_pred, y_true = _to_numpy(y_pred, y_true)
    mask = np.abs(y_true) > eps
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mbe(y_pred, y_true) -> float:
    """Mean Bias Error: positive means over-forecast."""
    y_pred, y_true = _to_numpy(y_pred, y_true)
    return float(np.mean(y_pred - y_true))


def all_metrics(y_pred, y_true) -> dict:
    return {
        "RMSE": rmse(y_pred, y_true),
        "MAE": mae(y_pred, y_true),
        "R2": r2_score(y_pred, y_true),
        "MAPE": mape(y_pred, y_true),
        "MBE": mbe(y_pred, y_true),
    }


# ============================================================
# Diebold-Mariano test with Newey-West HAC variance
# ============================================================


def _newey_west_hac_variance(x: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC variance estimator (Bartlett kernel)."""
    n = len(x)
    mean = np.mean(x)
    deviations = x - mean
    # Gamma_0 (variance)
    gamma_0 = np.mean(deviations ** 2)
    # Sum of weighted autocovariances
    weighted_sum = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.mean(deviations[:n - lag] * deviations[lag:])
        weighted_sum += 2 * weight * gamma_lag
    return (gamma_0 + weighted_sum) / n


def diebold_mariano_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    max_lag: int = 24,
    loss: str = "squared",
) -> Tuple[float, float]:
    """Diebold-Mariano test with Newey-West HAC standard errors.

    Tests H0: equal predictive accuracy vs H1: different predictive accuracy.

    Args:
        errors_model1: forecast errors (y_pred - y_true) from model 1
        errors_model2: forecast errors (y_pred - y_true) from model 2
        max_lag: maximum lag for Newey-West HAC variance (default 24 hours)
        loss: 'squared' (default, matches RMSE) or 'absolute' (matches MAE)

    Returns:
        (dm_stat, p_value): DM statistic (standard normal under null) and
        two-sided p-value. Negative DM statistic means model 1 has lower loss.
    """
    e1 = np.asarray(errors_model1).flatten()
    e2 = np.asarray(errors_model2).flatten()
    assert len(e1) == len(e2), f"length mismatch: {len(e1)} vs {len(e2)}"

    if loss == "squared":
        l1 = e1 ** 2
        l2 = e2 ** 2
    elif loss == "absolute":
        l1 = np.abs(e1)
        l2 = np.abs(e2)
    else:
        raise ValueError(f"unknown loss: {loss}")

    d = l1 - l2  # loss differential
    n = len(d)
    mean_d = np.mean(d)
    var_d = _newey_west_hac_variance(d, max_lag)
    if var_d <= 0:
        return 0.0, 1.0
    se_d = np.sqrt(var_d / n)
    dm_stat = mean_d / se_d

    # Two-sided p-value from standard normal
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ============================================================
# Block bootstrap CI on RMSE difference
# ============================================================


def block_bootstrap_ci(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    n_bootstrap: int = 1000,
    block_len: int = 24,
    seed: int = 42,
    loss: str = "squared",
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Moving-block bootstrap CI on the difference in mean loss.

    Args:
        errors_model1, errors_model2: forecast errors from two models
        n_bootstrap: number of bootstrap resamples (default 1000)
        block_len: block length preserving the daily cycle (default 24)
        seed: random seed for reproducibility
        loss: 'squared' (RMSE-matching) or 'absolute' (MAE-matching)
        confidence: confidence level (default 0.95)

    Returns:
        (mean_diff, ci_lo, ci_hi, std): mean of (loss1 - loss2), lower and
        upper bounds of the CI, and the std of the bootstrap distribution.
    """
    e1 = np.asarray(errors_model1).flatten()
    e2 = np.asarray(errors_model2).flatten()
    n = len(e1)
    rng = np.random.default_rng(seed)

    if loss == "squared":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss == "absolute":
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError(f"unknown loss: {loss}")

    diffs = np.zeros(n_bootstrap)
    n_blocks = int(np.ceil(n / block_len))
    for b in range(n_bootstrap):
        # Sample block start indices
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        # Concatenate blocks
        sampled = np.concatenate([l1[s: s + block_len] for s in starts])[:n]
        sampled2 = np.concatenate([l2[s: s + block_len] for s in starts])[:n]
        diffs[b] = np.mean(sampled) - np.mean(sampled2)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    alpha = 1 - confidence
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return mean_diff, ci_lo, ci_hi, std_diff
