"""Baseline models: PatchTST, CTNet, DLinear, LSTM, GRU, MLP, PV-Client, iTransformer-LSTM-KAN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTST(nn.Module):
    """Simplified PatchTST for univariate forecasting."""
    def __init__(self, seq_len=24, pred_len=1, patch_len=4, stride=4,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1, power_idx=0):
        super().__init__()
        self.power_idx = power_idx
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x, hour=None, elevation=None):
        x_power = x[:, :, self.power_idx]  # (B, L) — univariate
        patches = x_power.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        embedded = self.patch_embedding(patches) + self.pos_embedding  # (B, n_patches, d_model)
        encoded = self.encoder(embedded)  # (B, n_patches, d_model)
        flattened = encoded.flatten(1)  # (B, n_patches * d_model)
        out = self.output_proj(flattened)  # (B, 1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNModule(nn.Module):
    def __init__(self, n_features=5, n_filters=12):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, n_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 3, padding=1)
        self.conv3 = nn.Conv1d(n_filters, n_filters, 3, padding=1)
        self.projection = nn.Conv1d(n_filters, n_filters, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        return self.projection(x).transpose(1, 2)


class CTNet(nn.Module):
    """CNN-Transformer hybrid (Munsif et al. 2023)."""
    def __init__(self, n_features=5, d_model=12, n_heads=4, n_encoder_blocks=3,
                 encoder_dropout=0.2, decoder_dropout=0.3):
        super().__init__()
        class EB(nn.Module):
            def __init__(s, nf, dm, nh, dp):
                super().__init__()
                s.cnn = CNNModule(nf, dm)
                s.mha = nn.MultiheadAttention(dm, nh, dropout=dp, batch_first=True)
                s.norm1 = nn.LayerNorm(dm)
                s.ffn = nn.Sequential(nn.Linear(dm, dm*4), nn.ReLU(),
                                      nn.Dropout(dp), nn.Linear(dm*4, dm))
                s.norm2 = nn.LayerNorm(dm); s.dp = nn.Dropout(dp)
            def forward(s, x):
                c = s.cnn(x); a, _ = s.mha(c, c, c)
                x = s.norm1(c + s.dp(a)); return s.norm2(x + s.dp(s.ffn(x)))
        class Dec(nn.Module):
            def __init__(s, dm, nh, dp):
                super().__init__()
                s.mha = nn.MultiheadAttention(dm, nh, dropout=dp, batch_first=True)
                s.ffd = nn.Sequential(nn.Linear(dm, 32), nn.ReLU(), nn.Dropout(dp),
                                      nn.Linear(32, 64), nn.ReLU(), nn.Dropout(dp), nn.Linear(64, 1))
            def forward(s, x):
                a, _ = s.mha(x, x, x); return s.ffd((x + a).mean(dim=1))
        self.encs = nn.ModuleList([EB(n_features if i==0 else d_model, d_model, n_heads, encoder_dropout)
                                    for i in range(n_encoder_blocks)])
        self.dec = Dec(d_model, n_heads, decoder_dropout)
    def forward(self, x, hour=None, elevation=None):
        for b in self.encs: x = b(x)
        return self.dec(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5, seq_len=24, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * seq_len, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, x, hour=None, elevation=None):
        return self.net(x.flatten(1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, hour=None, elevation=None):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, hour=None, elevation=None):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
LSTM = SimpleLSTM
GRU = SimpleGRU
MLP = SimpleMLP


class DLinear(nn.Module):
    """Linear trend-seasonal decomposition (Zeng et al. 2023)."""
    def __init__(self, seq_len=24, horizon=1, input_channels=5):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.trend_proj = nn.Linear(seq_len, horizon, bias=True)
        self.seasonal_proj = nn.Linear(seq_len, horizon, bias=True)

    def forward(self, x, hour=None, elevation=None):
        x_power = x[:, :, 0]  # (B, L)
        kernel_size = 5
        pad = (kernel_size - 1) // 2
        padded = F.pad(x_power.unsqueeze(1), (pad, pad + 1), mode="replicate").squeeze(1)
        trend = F.avg_pool1d(x_power.unsqueeze(1), kernel_size=kernel_size, stride=1).squeeze(1)
        seasonal = x_power - trend
        out = self.trend_proj(trend) + self.seasonal_proj(seasonal)
        return out.unsqueeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "normalize":
            self._get_mean_std(x)
            x = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denormalize":
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            return x * (self.std + self.eps) + self.mean
        else:
            raise ValueError(f"unknown mode: {mode}")

    def _get_mean_std(self, x: torch.Tensor):
        dim2reduce = 1
        self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.std = x.std(dim=dim2reduce, keepdim=True).detach()


class PVClient(nn.Module):
    """PV-Client: cross-variable attention (Gao et al. 2026)."""
    def __init__(self, input_channels=5, seq_len=24, d_model=128, n_heads=8,
                 n_blocks=2, d_ff=256, dropout=0.1, horizon=1, revin=True):
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.horizon = horizon
        self.revin = RevIN(num_features=input_channels) if revin else None
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_pos = nn.Parameter(torch.zeros(1, input_channels, d_model))
        nn.init.normal_(self.var_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.trend_proj = nn.Linear(seq_len, horizon)
        self.gate = nn.Linear(d_model + input_channels, input_channels)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x, hour=None, elevation=None):
        B, L, C = x.shape
        if self.revin is not None:
            x = self.revin(x, "normalize")
        x_var = x.transpose(1, 2)
        var_tokens = self.var_proj(x_var) + self.var_pos
        attended = self.encoder(var_tokens)
        attended_pooled = attended.mean(dim=1)
        power = x[:, :, 0]
        trend_out = self.trend_proj(power)
        gate_input = torch.cat([attended_pooled, x[:, 0, :]], dim=-1)
        gate_weights = torch.sigmoid(self.gate(gate_input))
        attended_out = self.head(attended_pooled)
        out = trend_out + attended_out.squeeze(-1) * gate_weights[:, 0]
        if self.revin is not None:
            out = out * (self.revin.std[:, 0, 0] + self.revin.eps) + self.revin.mean[:, 0, 0]
        return out.unsqueeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PVClientAug(PVClient):
    """PV-Client with augmented input channels (5 + sin/cos hour + norm elev = 8)."""
    def __init__(self, seq_len=24, d_model=128, n_heads=8, n_blocks=2,
                 d_ff=256, horizon=1, elev_mean=35.6, elev_std=22.9, dropout=0.1):
        super().__init__(input_channels=8, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, n_blocks=n_blocks, d_ff=d_ff,
                         horizon=horizon, dropout=dropout)
        self.elev_mean = elev_mean
        self.elev_std = elev_std

    def forward(self, x, hour, elevation=None):
        B, L, C = x.shape
        sin_h = torch.sin(2 * torch.pi * hour / 24.0).unsqueeze(-1)
        cos_h = torch.cos(2 * torch.pi * hour / 24.0).unsqueeze(-1)
        if elevation is not None:
            norm_e = ((torch.clamp(elevation, min=0.0) - self.elev_mean) /
                      (self.elev_std + 1e-8)).unsqueeze(-1)
        else:
            norm_e = torch.zeros(B, L, 1, device=x.device)
        x_aug = torch.cat([x, sin_h, cos_h, norm_e], dim=-1)
        return super().forward(x_aug, hour, elevation)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class KANLayer(nn.Module):
    """Simplified Kolmogorov-Arnold Network layer."""
    def __init__(self, in_dim: int, out_dim: int, n_basis: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis
        self.coeffs = nn.Parameter(torch.zeros(in_dim, out_dim, n_basis))
        nn.init.normal_(self.coeffs, std=0.02)
        self.residual = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bases = [torch.ones_like(x)]
        for k in range(1, self.n_basis):
            bases.append(bases[-1] * torch.tanh(x))
        bases = torch.stack(bases, dim=-1)
        out = torch.einsum("...ib,iob->...o", bases, self.coeffs)
        return out + F.silu(self.residual(x))


class iTransformerLSTMKAN(nn.Module):
    """iTransformer + LSTM + cross-attention + KAN (Wu et al. 2024)."""
    def __init__(self, input_channels=5, seq_len=24, d_model=64, n_heads=4,
                 d_ff=128, d_lstm=64, n_lstm_layers=2, kan_basis=8,
                 horizon=1, dropout=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.horizon = horizon
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_pos = nn.Parameter(torch.zeros(1, input_channels, d_model))
        nn.init.normal_(self.var_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.itransformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(
            input_size=input_channels, hidden_size=d_lstm,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0)
        self.lstm_proj = nn.Linear(d_lstm, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.kan = nn.Sequential(
            KANLayer(d_model, d_ff, n_basis=kan_basis),
            KANLayer(d_ff, horizon, n_basis=kan_basis))

    def forward(self, x, hour=None, elevation=None):
        B, L, C = x.shape
        x_var = x.transpose(1, 2)
        var_tokens = self.var_proj(x_var) + self.var_pos
        itrans_out = self.itransformer(var_tokens)
        lstm_out, (h, c) = self.lstm(x)
        lstm_last = self.lstm_proj(lstm_out[:, -1, :])
        query = lstm_last.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, itrans_out, itrans_out)
        x_norm = self.norm1(query + attn_out)
        x_norm = self.norm2(x_norm + self.ffn(x_norm))
        out = self.kan(x_norm.squeeze(1))
        return out.unsqueeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class iTransformerLSTMKANAug(iTransformerLSTMKAN):
    """iTransformer-LSTM-KAN with augmented input channels (8 instead of 5)."""
    def __init__(self, seq_len=24, d_model=64, n_heads=4, d_ff=128,
                 d_lstm=64, n_lstm_layers=2, horizon=1,
                 elev_mean=35.6, elev_std=22.9, dropout=0.1):
        super().__init__(input_channels=8, seq_len=seq_len, d_model=d_model,
                         n_heads=n_heads, d_ff=d_ff, d_lstm=d_lstm,
                         n_lstm_layers=n_lstm_layers, horizon=horizon, dropout=dropout)
        self.elev_mean = elev_mean
        self.elev_std = elev_std

    def forward(self, x, hour, elevation=None):
        B, L, C = x.shape
        sin_h = torch.sin(2 * torch.pi * hour / 24.0).unsqueeze(-1)
        cos_h = torch.cos(2 * torch.pi * hour / 24.0).unsqueeze(-1)
        if elevation is not None:
            norm_e = ((torch.clamp(elevation, min=0.0) - self.elev_mean) /
                      (self.elev_std + 1e-8)).unsqueeze(-1)
        else:
            norm_e = torch.zeros(B, L, 1, device=x.device)
        x_aug = torch.cat([x, sin_h, cos_h, norm_e], dim=-1)
        return super().forward(x_aug, hour, elevation)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
