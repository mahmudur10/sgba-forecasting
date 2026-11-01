"""Training protocol configuration (paper Section 3.5, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Unified training protocol. Identical for all models, sites, and seeds."""

    epochs: int = 100
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_schedule: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6
    batch_size: int = 128
    grad_clip_max_norm: float = 1.0
    loss: str = "mse"
    seeds: tuple = (42, 123, 7)

    # Architecture defaults (SGBA)
    d_model: int = 96
    n_heads: int = 4
    n_blocks: int = 2
    d_ff: int = 384
    d_geo: int = 32
    alpha: float = 0.9
    seq_len: int = 24
    input_channels: int = 5
    horizon: int = 1
    dropout: float = 0.1

    # Data
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    ghi_threshold: float = 10.0  # W/m^2 nighttime cutoff
    interp_max_gap: int = 12  # 1 hour at 5-min resolution

    # Statistical testing
    dm_max_lag: int = 24
    bootstrap_n: int = 1000
    bootstrap_block_len: int = 24
    bootstrap_seed: int = 42

    # Device
    device: str = "cuda"

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "lr_schedule": self.lr_schedule,
            "lr_factor": self.lr_factor,
            "lr_patience": self.lr_patience,
            "lr_min": self.lr_min,
            "batch_size": self.batch_size,
            "grad_clip_max_norm": self.grad_clip_max_norm,
            "loss": self.loss,
            "seeds": list(self.seeds),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_blocks": self.n_blocks,
            "d_geo": self.d_geo,
            "alpha": self.alpha,
            "seq_len": self.seq_len,
            "input_channels": self.input_channels,
            "horizon": self.horizon,
            "dropout": self.dropout,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "ghi_threshold": self.ghi_threshold,
            "interp_max_gap": self.interp_max_gap,
            "dm_max_lag": self.dm_max_lag,
            "bootstrap_n": self.bootstrap_n,
            "bootstrap_block_len": self.bootstrap_block_len,
            "device": self.device,
        }


DEFAULT_CONFIG = TrainingConfig()
