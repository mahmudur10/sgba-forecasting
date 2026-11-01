"""Pytest fixtures shared across tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.utils.seeds import set_seed


@pytest.fixture(autouse=True)
def fixed_seed():
    """Set a fixed seed before every test."""
    set_seed(42)


@pytest.fixture
def sample_batch():
    """Standard sample input: (B=4, L=24, C=5) with hour and elevation."""
    B, L, C = 4, 24, 5
    x = torch.randn(B, L, C)
    hour = torch.rand(B, L) * 24
    elevation = torch.rand(B, L) * 60
    return x, hour, elevation


@pytest.fixture
def sample_errors():
    """Two sets of forecast errors for DM test."""
    rng = np.random.default_rng(42)
    n = 500
    e1 = rng.standard_normal(n) * 0.3
    e2 = rng.standard_normal(n) * 0.4 + 0.05
    return e1, e2
