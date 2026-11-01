#!/usr/bin/env python3
"""08_efficiency.py — FLOPs/latency/parameter analysis.

Usage:
    python scripts/08_efficiency.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sgba.models import (
    SGBA, PatchTST, DLinear, LSTM, GRU, MLP,
    PVClientAug, iTransformerLSTMKANAug,
)
from scripts._common import setup_experiment, save_json, RESULTS_DIR


def count_flops_thop(model, sample_input):
    """FLOPs via thop. Falls back to None if thop not installed."""
    try:
        from thop import profile
        flops, params = profile(model, inputs=sample_input, verbose=False)
        return int(flops)
    except Exception:
        return None


def measure_latency(model, x, hour, elev, n_warmup=10, n_trials=100):
    """Single-sample inference latency in ms."""
    model.eval()
    device = next(model.parameters()).device
    x, hour, elev = x.to(device), hour.to(device), elev.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                model(x, hour, elev)
            except TypeError:
                model(x, hour)

        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_trials):
                try:
                    model(x, hour, elev)
                except TypeError:
                    model(x, hour)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / n_trials
        else:
            t0 = time.time()
            for _ in range(n_trials):
                try:
                    model(x, hour, elev)
                except TypeError:
                    model(x, hour)
            return (time.time() - t0) / n_trials * 1000


def main():
    setup_experiment(seed=42)
    print("=== Efficiency benchmark ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, L, C = 1, 24, 5
    x = torch.randn(B, L, C)
    hour = torch.rand(B, L) * 24
    elev = torch.rand(B, L) * 60

    models = {
        "DLinear": DLinear(),
        "PatchTST": PatchTST(),
        "LSTM": LSTM(),
        "GRU": GRU(),
        "MLP": MLP(),
        "PV-Client-Aug": PVClientAug(),
        "iTrans.-KAN-Aug": iTransformerLSTMKANAug(),
        "SGBA-Elev": SGBA(geometry_mode="elev"),
    }

    results = {}
    for name, model in models.items():
        model = model.to(device)
        params = model.count_parameters()
        # Try to count FLOPs
        try:
            flops = count_flops_thop(model, (x.to(device), hour.to(device), elev.to(device)))
        except Exception:
            flops = None
        # Latency
        try:
            latency = measure_latency(model, x, hour, elev)
        except Exception as e:
            latency = None

        results[name] = {
            "params": params,
            "flops": flops,
            "latency_ms": latency,
            "device": str(device),
        }
        print(f"  {name}: params={params:,}  flops={flops}  latency={latency:.3f}ms" if latency else
              f"  {name}: params={params:,}  flops={flops}  latency=N/A")

    save_json({"device": str(device), "models": results},
              RESULTS_DIR / "08_efficiency.json")


if __name__ == "__main__":
    main()
