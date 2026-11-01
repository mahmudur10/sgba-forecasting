#!/usr/bin/env python3
"""01_train_sgba.py — Train SGBA-Elev (headline model) on one site.

Usage:
    python scripts/01_train_sgba.py --site site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import create_dataloaders
from scripts._common import parse_args, setup_experiment, save_json, RESULTS_DIR, DATA_DIR


def main():
    args = parse_args()
    site = args.get("site", "site7")
    seed = int(args.get("seed", 42))
    device = setup_experiment(seed=seed)

    print(f"=== Training SGBA-Elev on {site} (seed {seed}) ===")

    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        site, DATA_DIR, batch_size=128, seed=seed,
    )

    model = SGBA(
        geometry_mode="elev",
        elev_mean=norm_stats["elev_mean"],
        elev_std=norm_stats["elev_std"],
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    results = train_model(
        model, train_loader, val_loader, test_loader,
        config=TrainingConfig(epochs=100, batch_size=128),
        data_root=DATA_DIR, site_key=site,
        use_geometry=True, verbose=True,
    )
    print(f"\nTest metrics: {results['metrics']}")
    print(f"Training time: {results['train_time_sec']:.1f}s")

    save_json({
        "site": site, "seed": seed, "model": "SGBA-Elev",
        "params": sum(p.numel() for p in model.parameters()),
        "metrics": results["metrics"],
        "history": results["history"],
        "norm_stats": norm_stats,
    }, RESULTS_DIR / f"01_sgba_elev_{site}_seed{seed}.json")

    print(f"\nSaved: {RESULTS_DIR / f'01_sgba_elev_{site}_seed{seed}.json'}")


if __name__ == "__main__":
    main()
