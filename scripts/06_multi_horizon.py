#!/usr/bin/env python3
"""06_multi_horizon.py — Multi-horizon forecasting at h in {1, 6, 12, 24}.

Usage:
    python scripts/06_multi_horizon.py --site dkasc_site7
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA, PatchTST, DLinear
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


HORIZONS = [1, 6, 12, 24]


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    setup_experiment(seed=42)
    print(f"=== Multi-horizon evaluation on {site} ===")

    df = make_synthetic_dataset(n_days=400, site=site, seed=42)
    df_train, df_val, df_test = split_chronological(df)

    results = {}
    for h in HORIZONS:
        train_loader, val_loader, test_loader, meta = make_dataloaders(
            df_train, df_val, df_test, site=site, horizon=h,
        )
        results[h] = {"n_test": meta["n_test"]}
        for name, mk in [
            ("SGBA-Elev", lambda: SGBA(geometry_mode="elev",
                                       elev_mean=meta["elev_mean"],
                                       elev_std=meta["elev_std"])),
            ("PatchTST", lambda: PatchTST()),
            ("DLinear", lambda: DLinear()),
        ]:
            r = train_model(mk(), train_loader, val_loader, test_loader, verbose=False)
            results[h][name] = r["metrics"]["RMSE"]
            print(f"  h={h}  {name}: RMSE={r['metrics']['RMSE']:.4f}")

    save_json({"site": site, "horizons": results},
              RESULTS_DIR / f"06_multi_horizon_{site}.json")


if __name__ == "__main__":
    main()
