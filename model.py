from __future__ import annotations
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer to inject sequence order.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, W, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TranADCore(nn.Module):
    """
    TranADCore: SOTA Two-Phase Adversarial Transformer for Time Series Anomaly Detection.
    Strictly implements the Tuli et al. (VLDB'22) mechanics.
    """
    def __init__(
        self,
        feature_dim: int,
        window_size: int = 30,
        hidden_dim: int = 64,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super(TranADCore, self).__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.d_model = hidden_dim
        
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=window_size)
        
        # Projection layer to map concatenated [Input, Condition] to Transformer hidden dimension
        self.input_proj = nn.Linear(2 * feature_dim, self.d_model)
        self.tgt_proj = nn.Linear(feature_dim, self.d_model)
        
        # Encoder: Extracts temporal dependencies mapping to Z
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Decoder 1 (Phase 1 - Baseline)
        decoder_layers1 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, n_layers)
        
        # Decoder 2 (Phase 2 - Refiner)
        decoder_layers2 = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, n_layers)
        
        # Final projection to map back to feature_dim (with Sigmoid for min-max scaled data)
        self.fcn = nn.Sequential(
            nn.Linear(self.d_model, feature_dim),
            nn.Sigmoid(),
        )

    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds input X and condition C, runs through Transformer Encoder.
        """
        # Concatenate true input and condition feature-wise
        src = torch.cat((x, c), dim=-1)  # Shape: [B, W, 2*D]
        src = self.input_proj(src)       # Shape: [B, W, d_model]
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        memory = self.transformer_encoder(src)
        
        # Prepare target for decoding (auto-regression baseline)
        tgt = self.tgt_proj(tgt)
        tgt = tgt * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        return memory, tgt

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sliding window of shape (B, W, D).
        Returns:
            x1: Baseline reconstruction (Phase 1)
            x2: Focus-conditioned refined reconstruction (Phase 2)
        """
        # Phase 1: Baseline Reconstruction (No condition)
        c_zero = torch.zeros_like(x)
        memory1, tgt1 = self.encode(x, c_zero, x)
        out1 = self.transformer_decoder1(tgt1, memory1)
        x1 = self.fcn(out1)  # Shape: (B, W, D)
        
        # Phase 2: Refined Reconstruction (Conditioned on focus score)
        # Focus score is the squared residual error of Phase 1
        focus_score = (x1 - x) ** 2 
        memory2, tgt2 = self.encode(x, focus_score, x)
        out2 = self.transformer_decoder2(tgt2, memory2)
        x2 = self.fcn(out2)  # Shape: (B, W, D)
        
        return x1, x2