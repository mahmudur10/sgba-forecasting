"""Ablation controls for SGBA.

Six bias types × three positional encodings = 18 ablation configs per site.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGBABlockBase(nn.Module):
    """Base attention block — supports different bias types for ablation controls."""
    def __init__(self, d_model=96, n_heads=4, geo_dim=32, dropout=0.1, bias_type='bilinear'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bias_type = bias_type

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if bias_type in ('bilinear', 'sin_cos_only', 'elev_only', 'shuffled'):
            self.geo_bias_M = nn.Parameter(torch.zeros(n_heads, geo_dim, geo_dim))
        elif bias_type == 'alibi':
            self.alibi_slope = nn.Parameter(torch.zeros(n_heads))

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, tokens, geo_features, hour=None):
        B, L, _ = tokens.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(tokens).view(B, L, H, Dh).transpose(1, 2)
        K = self.W_k(tokens).view(B, L, H, Dh).transpose(1, 2)
        V = self.W_v(tokens).view(B, L, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.bias_type in ('bilinear', 'sin_cos_only', 'elev_only', 'shuffled'):
            proj = torch.einsum('blg,hgj->bhlj', geo_features, self.geo_bias_M)
            bias = torch.einsum('bhig,bjg->bhij', proj, geo_features)
            scores = scores + bias
        elif self.bias_type == 'alibi':
            positions = torch.arange(L, device=tokens.device).float()
            dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            bias = -self.alibi_slope.view(1, H, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
            scores = scores + bias

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(out)

        h = self.norm1(tokens + self.dropout(out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h, attn


class SGBAControl(nn.Module):
    """SGBA variant for ablation controls.
    
    bias_type options:
        'bilinear'      : full SGBA-Elev (headline)
        'sin_cos_only'  : bias uses only sin/cos (no elevation)
        'elev_only'     : bias uses only elevation (no sin/cos)
        'shuffled'      : bias uses FIXED-PERMUTATION elevation (null control)
        'alibi'         : ALiBi distance bias (no geometry)
        'none'          : no bias (standard attention)
    
    pe_type options (added to tokens, INDEPENDENT of bias):
        None            : no positional encoding on tokens
        'sinusoidal'    : standard sinusoidal PE added to tokens
        'learned'       : learned PE added to tokens
    
    FIX from original: 'shuffled' now uses a fixed permutation set at init
    time (deterministic), instead of random per forward pass. This makes
    evaluation reproducible. The permutation varies by global seed.
    """
    def __init__(self, seq_len=24, n_features=5, d_model=96, n_heads=4,
                 n_blocks=2, geo_dim=32, dropout=0.1,
                 bias_type='bilinear', pe_type=None,
                 ghi_envelope_mean=0.0, ghi_envelope_std=1.0,
                 elev_mean=0.0, elev_std=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.power_idx = 0
        self.ghi_idx = 3
        self.bias_type = bias_type
        self.pe_type = pe_type
        self.geo_dim = geo_dim

        self.register_buffer('ghi_envelope_mean', torch.tensor(float(ghi_envelope_mean)))
        self.register_buffer('ghi_envelope_std', torch.tensor(float(ghi_envelope_std)))
        self.register_buffer('elev_mean', torch.tensor(float(elev_mean)))
        self.register_buffer('elev_std', torch.tensor(float(elev_std)))

        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding on tokens (independent of bias)
        if pe_type == 'sinusoidal':
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe_sinusoidal', pe.unsqueeze(0))
        elif pe_type == 'learned':
            self.pe_learned = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.pe_learned, std=0.02)

        # Geo projection (only for bias types that use geo_features)
        if bias_type in ('bilinear', 'sin_cos_only', 'elev_only', 'shuffled'):
            self.geo_proj = nn.Sequential(
                nn.Linear(4, geo_dim), nn.ReLU(), nn.Linear(geo_dim, geo_dim)
            )

        # FIXED permutation for shuffled bias (set at init, deterministic)
        if bias_type == 'shuffled':
            perm = torch.randperm(seq_len)
            self.register_buffer('shuffle_perm', perm)

        self.blocks = nn.ModuleList([
            SGBABlockBase(d_model, n_heads, geo_dim, dropout, bias_type=bias_type)
            for _ in range(n_blocks)
        ])

        self.attn_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        nn.init.zeros_(self.attn_head[-1].weight)
        nn.init.zeros_(self.attn_head[-1].bias)

    def compute_geo_features(self, x, hour, elevation):
        B, L, _ = x.shape
        pos_sin = torch.sin(2 * math.pi * hour / 24.0)
        pos_cos = torch.cos(2 * math.pi * hour / 24.0)
        elev_norm = (elevation - self.elev_mean) / (self.elev_std + 1e-6)

        if self.bias_type == 'bilinear':
            geo_raw = torch.stack([pos_sin, pos_cos, elev_norm, elev_norm ** 2], dim=-1)
        elif self.bias_type == 'sin_cos_only':
            zeros = torch.zeros_like(pos_sin)
            geo_raw = torch.stack([pos_sin, pos_cos, zeros, zeros], dim=-1)
        elif self.bias_type == 'elev_only':
            zeros = torch.zeros_like(pos_sin)
            geo_raw = torch.stack([zeros, zeros, elev_norm, elev_norm ** 2], dim=-1)
        elif self.bias_type == 'shuffled':
            # FIXED permutation (deterministic) — null control
            elev_shuffled = elev_norm[:, self.shuffle_perm]
            geo_raw = torch.stack([pos_sin, pos_cos, elev_shuffled, elev_shuffled ** 2], dim=-1)
        else:
            geo_raw = torch.zeros(B, L, 4, device=x.device)

        return self.geo_proj(geo_raw)

    def forward(self, x, hour, elevation=None):
        h = self.input_proj(x)

        if self.pe_type == 'sinusoidal':
            h = h + self.pe_sinusoidal
        elif self.pe_type == 'learned':
            h = h + self.pe_learned

        geo_features = None
        if self.bias_type in ('bilinear', 'sin_cos_only', 'elev_only', 'shuffled'):
            geo_features = self.compute_geo_features(x, hour, elevation)

        for block in self.blocks:
            h, _ = block(h, geo_features, hour=hour)

        attn_repr = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return self.attn_head(attn_repr)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
