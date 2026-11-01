"""Tests for SGBA model, ablation controls, and baselines."""
import pytest
import torch
from sgba.models import SGBA, SGBAControl, PatchTST, CTNet, SimpleLSTM, SimpleGRU, SimpleMLP


class TestSGBA:
    def test_forward_elev(self):
        m = SGBA(geometry_mode='elev', elev_mean=35.0, elev_std=22.0)
        x = torch.randn(4, 24, 5)
        hour = torch.rand(4, 24) * 24
        elev = torch.rand(4, 24) * 60
        out = m(x, hour, elev)
        assert out.shape == (4, 1)

    def test_forward_ewma(self):
        m = SGBA(geometry_mode='ewma')
        x = torch.randn(4, 24, 5)
        hour = torch.rand(4, 24) * 24
        out = m(x, hour)
        assert out.shape == (4, 1)

    def test_zero_init(self):
        m = SGBA(geometry_mode='elev')
        x = torch.randn(4, 24, 5)
        hour = torch.rand(4, 24) * 24
        elev = torch.rand(4, 24) * 60
        out = m(x, hour, elev)
        assert out.abs().max().item() < 1e-6

    def test_param_count(self):
        m = SGBA(geometry_mode='elev')
        n = sum(p.numel() for p in m.parameters())
        assert 240_000 < n < 270_000, f"param count {n} not in expected range"


class TestSGBAControl:
    @pytest.mark.parametrize("bias_type", ["bilinear", "sin_cos_only", "elev_only", "shuffled", "alibi", "none"])
    @pytest.mark.parametrize("pe_type", [None, "sinusoidal", "learned"])
    def test_all_combos(self, bias_type, pe_type):
        m = SGBAControl(bias_type=bias_type, pe_type=pe_type, elev_mean=35.0, elev_std=22.0)
        x = torch.randn(4, 24, 5)
        hour = torch.rand(4, 24) * 24
        elev = torch.rand(4, 24) * 60
        out = m(x, hour, elev)
        assert out.shape == (4, 1)


class TestBaselines:
    def test_patchtst(self):
        m = PatchTST()
        x = torch.randn(4, 24, 5)
        out = m(x)
        assert out.shape == (4, 1)

    def test_ctnet(self):
        m = CTNet()
        x = torch.randn(4, 24, 5)
        out = m(x)
        assert out.shape == (4, 1)

    def test_lstm(self):
        m = SimpleLSTM()
        x = torch.randn(4, 24, 5)
        out = m(x)
        assert out.shape == (4, 1)

    def test_gru(self):
        m = SimpleGRU()
        x = torch.randn(4, 24, 5)
        out = m(x)
        assert out.shape == (4, 1)

    def test_mlp(self):
        m = SimpleMLP()
        x = torch.randn(4, 24, 5)
        out = m(x)
        assert out.shape == (4, 1)
