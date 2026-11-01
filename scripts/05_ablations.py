#!/usr/bin/env python3
"""05_ablations.py — Expanded ablation with 6 bias types x 3 positional encodings.

Usage:
    python scripts/05_ablations.py --site dkasc_site7
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBAControl
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


BIAS_TYPES = ["bilinear", "sin_cos_only", "elev_only", "shuffled", "alibi", "none"]
PE_TYPES = ["none", "sinusoidal", "learned"]


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    setup_experiment(seed=42)
    print(f"=== Expanded ablation on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=42)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    results = {}
    for bias in BIAS_TYPES:
        for pe in PE_TYPES:
            key = f"{bias}__{pe}"
            model = SGBAControl(
                bias_type=bias, pe_type=pe,
                elev_mean=meta["elev_mean"], elev_std=meta["elev_std"],
            )
            r = train_model(model, train_loader, val_loader, test_loader, verbose=False)
            results[key] = {"RMSE": r["metrics"]["RMSE"], "params": model.count_parameters()}
            print(f"  {key}: RMSE={r['metrics']['RMSE']:.4f}")

    save_json({"site": site, "ablation": results},
              RESULTS_DIR / f"05_ablation_{site}.json")


if __name__ == "__main__":
    main()
