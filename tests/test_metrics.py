"""Tests for metrics, DM test, block bootstrap."""
import numpy as np
import pytest
from sgba.training.metrics import rmse, mae, r2_score, diebold_mariano_test, block_bootstrap_ci


class TestBasicMetrics:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert np.isclose(rmse(y_pred, y_true), 1.0)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == 1.0


class TestDMTest:
    def test_significant_difference(self):
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(500) * 0.3
        e2 = rng.standard_normal(500) * 0.4 + 0.05
        dm_stat, p_val = diebold_mariano_test(e1, e2, max_lag=24)
        assert dm_stat < 0
        assert p_val < 0.05

    def test_no_difference(self):
        rng = np.random.default_rng(42)
        e = rng.standard_normal(500) * 0.3
        dm_stat, p_val = diebold_mariano_test(e, e, max_lag=24)
        assert abs(dm_stat) < 0.01
        assert p_val > 0.99


class TestBlockBootstrap:
    def test_returns_4_values(self):
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(500) * 0.3
        e2 = rng.standard_normal(500) * 0.4
        result = block_bootstrap_ci(e1, e2, n_bootstrap=10, block_len=24, seed=42)
        assert len(result) == 4

    def test_reproducibility(self):
        rng = np.random.default_rng(42)
        e1 = rng.standard_normal(500) * 0.3
        e2 = rng.standard_normal(500) * 0.4
        r1 = block_bootstrap_ci(e1, e2, n_bootstrap=50, block_len=24, seed=42)
        r2 = block_bootstrap_ci(e1, e2, n_bootstrap=50, block_len=24, seed=42)
        assert r1 == r2
