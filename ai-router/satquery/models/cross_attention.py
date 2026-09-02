"""
satquery/models/cross_attention.py
───────────────────────────────────
Optical-SAR Cross-Attention Bottleneck.

Design:
    Q  ← optical token sequence  (B, N_opt, D)
    K  ← SAR spatial tokens      (B, N_sar, D)   (flattened from feature map)
    V  ← SAR spatial tokens      (B, N_sar, D)

Intuition: The optical image drives the "what am I looking at?" query.
The SAR image provides structural backscatter evidence as keys/values.
The attention output enriches optical tokens with SAR structural signals.

Architecture:
    SARSpatialAdapter  → flatten + linear project SAR feature map to token seq
    CrossAttentionBlock → nn.MultiheadAttention + LayerNorm + FFN residual
    FusionPooler        → pool attended tokens to a single (B, D) embedding
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from satquery.config import FUSION_D_MODEL, FUSION_HEADS


class SARSpatialAdapter(nn.Module):
    """
    Converts ConvNeXt SAR feature map (B, D, H', W') into a token sequence
    (B, H'*W', D) compatible with MultiheadAttention.

    A learned positional embedding is added so the model can reason about
    spatial layout within the SAR feature map.

    Args:
        max_spatial_tokens: Maximum number of SAR spatial positions expected.
            For IMG_SIZE=512 with ConvNeXt-Tiny stride-32: (512//32)^2 = 256.
    """

    def __init__(self, max_spatial_tokens: int = 256) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_spatial_tokens, FUSION_D_MODEL) * 0.02
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: (B, D, H', W')
        Returns:
            tokens  : (B, H'*W', D)
        """
        # Rearrange spatial dims into token dimension
        tokens = rearrange(feat_map, "b d h w -> b (h w) d")  # (B, N_sar, D)
        n = tokens.size(1)
        tokens = tokens + self.pos_embed[:, :n, :]
        return tokens


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block where optical queries attend to SAR keys/values.

    Includes:
    - Cross-attention (Q=optical, K=SAR, V=SAR)
    - Residual connection + LayerNorm
    - 2-layer FFN (4× expansion) + residual + LayerNorm
    - Dropout for regularisation

    Args:
        d_model     : Hidden dimension (must match FUSION_D_MODEL).
        num_heads   : Number of attention heads.
        ffn_ratio   : FFN expansion ratio (default 4×).
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        d_model: int = FUSION_D_MODEL,
        num_heads: int = FUSION_HEADS,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm_opt = nn.LayerNorm(d_model)
        self.norm_sar = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # expects (B, N, D) layout
        )

        ffn_dim = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        optical_tokens: torch.Tensor,  # (B, N_opt, D)
        sar_tokens: torch.Tensor,      # (B, N_sar, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            enriched_optical : (B, N_opt, D)  — optical tokens fused with SAR info
            attn_weights     : (B, N_opt, N_sar) — cross-attention map for visualisation
        """
        # Cross-attention: optical queries over SAR keys/values
        opt_normed = self.norm_opt(optical_tokens)
        sar_normed = self.norm_sar(sar_tokens)

        attended, attn_weights = self.cross_attn(
            query=opt_normed,
            key=sar_normed,
            value=sar_normed,
            need_weights=True,
            average_attn_weights=True,  # (B, N_opt, N_sar) averaged over heads
        )

        # Residual over original optical tokens
        optical_tokens = optical_tokens + attended

        # FFN with residual
        optical_tokens = optical_tokens + self.ffn(self.norm_ffn(optical_tokens))

        return optical_tokens, attn_weights


class FusionPooler(nn.Module):
    """
    Pools the attended optical token sequence to a single global embedding.

    Strategy:
    - If the sequence includes a CLS token (index 0 from ViT), use it directly.
    - Otherwise fall back to mean pooling.

    Returns a (B, D) vector used for classification and as the fused_embedding
    in the API contract.
    """

    def __init__(self, use_cls_token: bool = True) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D)
        Returns:
            embedding: (B, D)
        """
        if self.use_cls_token:
            return tokens[:, 0, :]   # CLS token from ViT
        return tokens.mean(dim=1)    # mean pooling fallback
