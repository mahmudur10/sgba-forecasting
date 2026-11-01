# SGBA: Solar Geometry-Biased Attention for PV Power Forecasting

A physics-informed Transformer that injects solar geometry as a bilinear
pairwise bias on the attention score matrix, for short-term photovoltaic
power forecasting.

## Overview

SGBA (Solar Geometry-Biased Attention) modifies the standard Transformer
attention mechanism by adding a **bilinear pairwise bias** derived from
solar geometry:

```
Attn_SGBA(Q, K, V, G) = softmax(QK^T / sqrt(d) + B_geo) V
```

where `B_geo[i,j] = g_i^T M_h g_j` and `g_i` is a projected geometry
feature vector at position `i` containing:
- Sinusoidal hour-of-day encoding (sin, cos)
- Normalized solar elevation (or EWMA envelope of GHI)
- Squared elevation term

The bias matrix `M_h` is **zero-initialized**, so SGBA reduces exactly to
standard attention at the start of training. The model learns how much
geometry to inject.

## Installation

```bash
# From source (recommended)
pip install -e .

# Or with all optional dependencies (pvlib, matplotlib, xgboost, etc.)
pip install -e .[all]

# Or just the core dependencies
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0

## Quick Start

```python
import torch
from sgba.models import SGBA

# Create the headline model (true solar elevation mode)
model = SGBA(
    geometry_mode='elev',
    elev_mean=35.6,   # training-set mean of elevation
    elev_std=22.9,    # training-set std of elevation
)

# Forward pass
# x: (B, 24, 5) — [power, temp, humidity, ghi, dhi]
# hour: (B, 24) — hour of day [0, 24)
# elevation: (B, 24) — solar elevation in degrees
output = model(x, hour, elevation)  # (B, 1)
```

For a complete walkthrough (training, evaluation, statistical tests, and
ablation control comparison), see
[`notebooks/demo.ipynb`](notebooks/demo.ipynb).

## Package Structure

```
sgba-forecasting/
├── sgba/                     # importable package
│   ├── models/
│   │   ├── sgba.py           # SGBA, SGBABlock, AnomalyExtractor
│   │   ├── ablations.py      # SGBAControl (ShuffledElev, SinCosOnly, etc.)
│   │   ├── baselines.py      # CTNet, PatchTST, DLinear, LSTM, GRU, MLP
│   │   ├── pv_client.py      # PV-Client with augmented input
│   │   └── itransformer.py   # iTransformer-LSTM-KAN
│   ├── data/
│   │   ├── datasets.py       # PVDatasetV2, MultiHorizonDataset
│   │   └── solar.py          # pvlib solar position computation
│   ├── training/
│   │   ├── trainer.py        # Unified training loop
│   │   ├── metrics.py        # RMSE/MAE/R² + DM test + block bootstrap
│   │   └── protocol.py       # TrainingConfig dataclass
│   └── utils/
│       └── seeds.py          # set_seed()
├── scripts/                  # experiment runners (01-13)
│   ├── _common.py            # shared utilities
│   ├── 01_train_sgba.py      # train SGBA-Elev on one site
│   ├── 02_train_baselines.py # train all baselines on one site
│   ├── 03_multi_seed.py      # multi-seed experiment
│   ├── 04_cross_climate.py   # cross-climate validation
│   ├── 05_ablations.py       # expanded ablation (6 bias types × 3 PEs)
│   ├── 06_multi_horizon.py   # multi-horizon forecasting (h=1,6,12,24)
│   ├── 07_dm_tests.py        # Diebold-Mariano significance tests
│   ├── 08_efficiency.py      # FLOPs/latency/param analysis
│   ├── 09_generate_figures.py # all 12 publication figures
│   ├── 10_alpha_sensitivity.py # α sweep for SGBA-EWMA (fig6)
│   ├── 11_extract_predictions.py # save test preds + attention (figs 7,8,9)
│   ├── 12_capacity_matched.py  # capacity-matched variants (fig11)
│   └── 13_regime_analysis.py   # clear vs cloudy performance (fig12)
├── tests/                    # pytest smoke tests (52 tests)
│   ├── conftest.py
│   ├── test_models.py        # SGBA, SGBAControl, AnomalyExtractor
│   ├── test_baselines.py     # all baselines + RevIN
│   ├── test_data.py          # PVDatasetV2, MultiHorizonDataset, pvlib
│   ├── test_metrics.py       # RMSE/MAE/R², DM test, block bootstrap
│   └── test_training.py      # end-to-end training loop
├── notebooks/
│   └── demo.ipynb            # single clean demo
├── configs/
│   └── default.yaml          # unified training protocol
├── data/
│   └── README.md             # dataset instructions
├── pyproject.toml            # PEP 517/518 build config
├── requirements.txt
├── LICENSE                   # MIT
└── README.md                 # this file
```

## Available Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| SGBA-Elev | 252,289 | Headline model with true solar elevation from pvlib |
| SGBA-EWMA | 252,289 | EWMA envelope of GHI as geometry proxy |
| CT-Net | 13,081 | CNN-Transformer hybrid (Munsif et al., 2023) |
| PatchTST | 101,057 | Patch-based Transformer (Nie et al., 2023) |
| DLinear | 50 | Linear trend-seasonal decomposition |
| LSTM | 51,521 | Standard LSTM |
| GRU | 38,657 | Standard GRU |
| MLP | 23,809 | Multi-layer perceptron |
| PV-Client-Aug | configurable | Cross-variable attention (Gao et al., 2026) |
| iTransformer-LSTM-KAN-Aug | configurable | Four-module hybrid (Wu et al., 2024) |

## Training Protocol

All models are trained with an identical unified protocol:

| Parameter | Value |
|-----------|-------|
| Epochs | 100 (fixed, no early stopping) |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch Size | 128 |
| Gradient Clipping | max norm 1.0 |
| Loss | MSE |
| Seeds | 42, 123, 7 |

## Datasets

- **DKASC** (Alice Springs, Australia): 3 sites, desert climate, 2020–2023
- **PVGIS** (Singapore/Oslo/Madrid): 3 sites, tropical/cold/Mediterranean, 2016–2020

See [`data/README.md`](data/README.md) for download and processing
instructions.

## Reproducibility

To reproduce the paper's experiments end-to-end:

```bash
# 1. Install the package and test dependencies
pip install -e .[dev]
pytest tests/ -v               # 52 tests should pass

# 2. Download data (see data/README.md) into data/processed/

# 3. Run experiments in order
python scripts/01_train_sgba.py --site dkasc_site7 --seed 42
python scripts/02_train_baselines.py --site dkasc_site7 --seed 42
python scripts/03_multi_seed.py --site dkasc_site7
python scripts/04_cross_climate.py
python scripts/05_ablations.py --site dkasc_site7
python scripts/06_multi_horizon.py --site dkasc_site7
python scripts/07_dm_tests.py --site dkasc_site7 --seed 42
python scripts/08_efficiency.py
python scripts/10_alpha_sensitivity.py --site dkasc_site7 --seed 42
python scripts/11_extract_predictions.py --site dkasc_site7 --seed 42
python scripts/12_capacity_matched.py --site dkasc_site7 --seed 42
python scripts/13_regime_analysis.py --site dkasc_site7 --seed 42
python scripts/09_generate_figures.py  # produces all 12 figures
```

Each script saves its outputs to `results/` (JSON) and `checkpoints/`
(PyTorch checkpoints). The figure script reads the JSON files and
produces both PDF (for the paper) and PNG (for this README) versions.

## Statistical Tests

The package implements the Diebold-Mariano test with Newey-West HAC
standard errors. The test correctly handles autocorrelation in
stride-1 window errors, which would otherwise inflate the effective
sample size and produce uninterpretable p-values.

```python
from sgba.training import diebold_mariano_test, block_bootstrap_ci
import numpy as np

e1 = np.load('errors_sgba.npy')        # forecast errors from model 1
e2 = np.load('errors_baseline.npy')    # forecast errors from model 2

# DM test (Newey-West HAC)
dm_stat, p_value = diebold_mariano_test(e1, e2, max_lag=24)

# Block bootstrap CI on RMSE difference
mean_diff, ci_lo, ci_hi, std = block_bootstrap_ci(
    e1, e2, n_bootstrap=1000, block_len=24, seed=42,
)
```

## Citation

If you use this code, please cite:

```bibtex
@article{sgba2026,
  title={Solar Geometry-Biased Attention for Short-Term Photovoltaic
         Power Forecasting: A Physics-Informed Transformer Architecture},
  author={Labib, Mahmudur Rahman and Munna, Abdullah Ibne Abbas and
          Siam, Md Abid Hossain and Rahman, Rashedur M.},
  journal={SN Computer Science},
  year={2026}
}
```

## License

[MIT](LICENSE)
