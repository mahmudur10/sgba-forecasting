"""Training utilities: trainer, metrics, protocol."""

from sgba.training.trainer import train_model, evaluate_model, Trainer
from sgba.training.metrics import (
    rmse, mae, r2_score, mape, mbe,
    diebold_mariano_test, block_bootstrap_ci,
)
from sgba.data.datasets import compute_metrics
from sgba.training.protocol import TrainingConfig, DEFAULT_CONFIG

__all__ = [
    "train_model",
    "evaluate_model",
    "Trainer",
    "rmse",
    "mae",
    "r2_score",
    "mape",
    "mbe",
    "diebold_mariano_test",
    "block_bootstrap_ci",
    "compute_metrics",
    "TrainingConfig",
    "DEFAULT_CONFIG",
]
