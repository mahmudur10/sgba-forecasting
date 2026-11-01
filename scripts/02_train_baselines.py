#!/usr/bin/env python3
"""02_train_baselines.py — Train all baselines on one site.

Usage:
    python scripts/02_train_baselines.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import (
    CTNet, PatchTST, DLinear, LSTM, GRU, MLP,
    PVClientAug, iTransformerLSTMKANAug,
)
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seed = int(args.get("seed", 42))
    device = setup_experiment(seed=seed)

    print(f"=== Training all baselines on {site} (seed {seed}) ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    baselines = {
        "DLinear": DLinear(),
        "PatchTST": PatchTST(),
        "LSTM": LSTM(),
        "GRU": GRU(),
        "MLP": MLP(),
        "CT-Net": CTNet(),
        "PV-Client-Aug": PVClientAug(),
        "iTransformer-LSTM-KAN-Aug": iTransformerLSTMKANAug(),
    }

    all_results = {}
    for name, model in baselines.items():
        print(f"\n--- {name} ({model.count_parameters():,} params) ---")
        results = train_model(model, train_loader, val_loader, test_loader, verbose=False)
        print(f"  Test RMSE: {results['metrics']['RMSE']:.4f}")
        all_results[name] = {
            "params": model.count_parameters(),
            "metrics": results["metrics"],
            "history": results["history"],
        }

    save_json({
        "site": site,
        "seed": seed,
        "baselines": all_results,
        "elev_mean": meta["elev_mean"],
        "elev_std": meta["elev_std"],
    }, RESULTS_DIR / f"02_baselines_{site}_seed{seed}.json")
    print(f"\nSaved: {RESULTS_DIR / f'02_baselines_{site}_seed{seed}.json'}")


if __name__ == "__main__":
    main()
