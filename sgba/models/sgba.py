"""SGBA model: Solar Geometry-Biased Attention for PV power forecasting.

Reference architecture (paper Section 3):
    - Input:        (B, 24, 5)  [power, temp, hum, ghi, dhi]
    - Geometry:     bilinear bias B[i,j] = g_i^T M_h g_j on attention scores
    - M_h is zero-initialized so untrained SGBA == standard attention
    - 2 SGBA blocks, 4 heads, d_model=96, head_dim=24
    - Single attention head output (zero-initialized final layer)
    - Total params: 252,289

Geometry feature vector g_t in R^32 (4 raw features projected nonlinearly):
    [sin(2*pi*hour/24), cos(2*pi*hour/24), tilde_e, tilde_e^2]  -> W2 ReLU W1 -> 32
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyExtractor(nn.Module):
    """Fixed EWMA envelope extraction with warm-start initialization.
    R2.2b FIX: e_0 = x_0 (first observed value) instead of e_0 = 0.
    """
    def __init__(self, alpha=0.9):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(float(alpha)))

    def forward(self, x):
        B, L, C = x.shape
        envelope = torch.zeros_like(x)
        env = x[:, 0, :].clone()  # warm-start
        for t in range(L):
            env = self.alpha * env + (1.0 - self.alpha) * x[:, t, :]
            envelope[:, t, :] = env
        return x - envelope, envelope


class SGBABlock(nn.Module):
    """Attention block with bilinear solar geometry bias.
    R2.1 FIX: B[i,j] = g_i^T M_h g_j (true pairwise coupling).
    M_h is zero-initialized so SGBA reduces to standard attention at init.
    """
    def __init__(self, d_model=96, n_heads=4, geo_dim=32, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Bilinear bias: B[i,j] = g_i^T M_h g_j for each head h
        self.geo_bias_M = nn.Parameter(
            torch.zeros(n_heads, geo_dim, geo_dim)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, tokens, geo_features):
        B, L, _ = tokens.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(tokens).view(B, L, H, Dh).transpose(1, 2)
        K = self.W_k(tokens).view(B, L, H, Dh).transpose(1, 2)
        V = self.W_v(tokens).view(B, L, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if geo_features is not None:
            proj = torch.einsum('blg,hgj->bhlj', geo_features, self.geo_bias_M)
            bias = torch.einsum('bhig,bjg->bhij', proj, geo_features)
            scores = scores + bias

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)

        h = self.norm1(tokens + self.dropout(out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h, attn


class SGBA(nn.Module):
    """Solar Geometry-Biased Attention (Corrected).
    No envelope path (R1.4/R2.5 fix). Attention head only.
    Zero-init final layer so SGBA starts with zero predictions.
    """
    def __init__(self, seq_len=24, n_features=5, d_model=96, n_heads=4,
                 n_blocks=2, geo_dim=32, dropout=0.1,
                 geometry_mode='ewma',
                 ghi_envelope_mean=0.0, ghi_envelope_std=1.0,
                 elev_mean=0.0, elev_std=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.power_idx = 0
        self.ghi_idx = 3
        self.geometry_mode = geometry_mode
        self.geo_dim = geo_dim

        self.register_buffer('ghi_envelope_mean', torch.tensor(float(ghi_envelope_mean)))
        self.register_buffer('ghi_envelope_std', torch.tensor(float(ghi_envelope_std)))
        self.register_buffer('elev_mean', torch.tensor(float(elev_mean)))
        self.register_buffer('elev_std', torch.tensor(float(elev_std)))

        if geometry_mode == 'ewma':
            self.anomaly_extractor = AnomalyExtractor(alpha=0.9)

        self.input_proj = nn.Linear(n_features, d_model)

        # Nonlinear geo_proj (R2.2f fix): Linear -> ReLU -> Linear
        self.geo_proj = nn.Sequential(
            nn.Linear(4, geo_dim), nn.ReLU(), nn.Linear(geo_dim, geo_dim)
        )

        self.blocks = nn.ModuleList([
            SGBABlock(d_model, n_heads, geo_dim, dropout)
            for _ in range(n_blocks)
        ])

        # Attention head only — NO envelope path (R1.4/R2.5 fix)
        self.attn_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        nn.init.zeros_(self.attn_head[-1].weight)
        nn.init.zeros_(self.attn_head[-1].bias)

    def compute_geo_features(self, x, hour, elevation=None, envelope=None):
        B, L, _ = x.shape
        # R2.2e: hour-of-day from timestamp, not window index
        pos_sin = torch.sin(2 * math.pi * hour / 24.0)
        pos_cos = torch.cos(2 * math.pi * hour / 24.0)

        if self.geometry_mode == 'ewma':
            if envelope is None:
                _, envelope = self.anomaly_extractor(x)
            geo_signal = envelope[:, :, self.ghi_idx]
            # R2.2c: global normalization (training-set stats)
            geo_signal_norm = (geo_signal - self.ghi_envelope_mean) / (self.ghi_envelope_std + 1e-6)
        else:  # 'elev'
            geo_signal = elevation
            geo_signal_norm = (geo_signal - self.elev_mean) / (self.elev_std + 1e-6)

        geo_raw = torch.stack([pos_sin, pos_cos, geo_signal_norm, geo_signal_norm ** 2], dim=-1)
        return self.geo_proj(geo_raw)

    def forward(self, x, hour, elevation=None):
        envelope = None
        if self.geometry_mode == 'ewma':
            _, envelope = self.anomaly_extractor(x)
        geo_features = self.compute_geo_features(x, hour, elevation, envelope)
        h = self.input_proj(x)
        for block in self.blocks:
            h, _ = block(h, geo_features)
        attn_repr = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return self.attn_head(attn_repr)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
