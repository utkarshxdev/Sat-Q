"""
satquery/models/encoders.py
────────────────────────────
Dual-encoder backbone setup.

OpticalEncoder  →  timm ViT-B/16 (or Swin-T) with SeCo pretrained weights
                   Input : (B, 3, H, W)
                   Output: (B, N_tokens, D) where N = (H/16)*(W/16)+1

SAREncoder      →  timm ConvNeXt-Tiny
                   Input : (B, 2, H, W)   [VV, VH channels]
                   Output: (B, D, H//32, W//32) — spatial feature map

Both encoders expose a `forward_features()` method that returns raw token /
feature-map tensors so the cross-attention module can consume them.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm

from satquery.config import DEVICE, FUSION_D_MODEL, SAR_CHANNELS


# ─── Optical Encoder (ViT-B/16 backbone) ─────────────────────────────────────

class OpticalEncoder(nn.Module):
    """
    Wraps timm ViT-B/16 as a pure feature extractor.

    The classification head is removed; instead we return the full sequence
    of patch tokens (including [CLS]) so the cross-attention bottleneck can
    attend over spatial positions.

    Args:
        pretrained (bool): Load ImageNet-21k weights. In production, replace
            with SeCo / Prithvi weights via a checkpoint load after init.
        freeze_stages (int): Number of ViT transformer blocks to freeze
            (counted from the stem). Freezing the first 8 of 12 blocks cuts
            Colab VRAM usage significantly.
    """

    def __init__(self, pretrained: bool = True, freeze_stages: int = 8) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,        # drop classification head
            global_pool="",       # return all tokens, not just CLS pool
        )
        self.d_model: int = self.backbone.embed_dim  # 768 for ViT-B

        # Freeze transformer blocks to preserve pretrained representations
        if freeze_stages >= 12:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_stages:
                    for param in block.parameters():
                        param.requires_grad = False

        # Projection to a shared d_model (identity if already 768)
        self.proj = nn.Linear(self.d_model, FUSION_D_MODEL) \
            if self.d_model != FUSION_D_MODEL else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  H=W=224 or 512 (ViT uses interpolated pos-embed)
        Returns:
            tokens: (B, N, FUSION_D_MODEL)
        """
        tokens = self.backbone.forward_features(x)   # (B, N, d_model)
        return self.proj(tokens)


# ─── SAR Encoder (ConvNeXt-Tiny backbone) ────────────────────────────────────

class SAREncoder(nn.Module):
    """
    Wraps timm ConvNeXt-Tiny as a spatial feature extractor for 2-channel SAR.

    The first conv layer is replaced to accept 2 input channels (VV, VH)
    instead of 3. ImageNet weights are *partially* reused by averaging
    the pretrained kernel weights across the missing channel dimension.

    Args:
        pretrained (bool): Load ImageNet-1k weights, then adapt first layer.
        freeze_stages (int): Freeze the first N ConvNeXt stages (0-3).
    """

    def __init__(self, pretrained: bool = True, freeze_stages: int = 2) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",     # return spatial feature map (B, C, H', W')
        )

        # ── Adapt stem conv: 3-ch → 2-ch ─────────────────────────────────────
        old_stem = self.backbone.stem[0]              # Conv2d(3, 96, 4, 4)
        new_stem = nn.Conv2d(
            SAR_CHANNELS,
            old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )
        if pretrained:
            # Average the 3-channel pretrained kernel to initialise 2-channel kernel
            with torch.no_grad():
                new_stem.weight.copy_(
                    old_stem.weight[:, :SAR_CHANNELS, :, :]
                    + old_stem.weight[:, SAR_CHANNELS - 1 : SAR_CHANNELS, :, :]
                )
                if old_stem.bias is not None:
                    new_stem.bias.copy_(old_stem.bias)
        self.backbone.stem[0] = new_stem

        # Freeze stages to preserve texture/structural representations
        if freeze_stages >= 4:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            stages_to_freeze = list(self.backbone.stages[:freeze_stages])
            for stage in stages_to_freeze:
                for param in stage.parameters():
                    param.requires_grad = False

        self.out_channels: int = 768  # ConvNeXt-Tiny final stage output channels

        # Project to shared d_model
        self.proj = nn.Conv2d(self.out_channels, FUSION_D_MODEL, kernel_size=1) \
            if self.out_channels != FUSION_D_MODEL else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: (B, 2, H, W)
        Returns:
            feat_map : (B, FUSION_D_MODEL, H', W')  where H'=H//32, W'=W//32
            spatial  : (H', W') for positional reconstruction downstream
        """
        feat = self.backbone.forward_features(x)     # (B, C, H', W')
        feat = self.proj(feat)                        # (B, FUSION_D_MODEL, H', W')
        _, _, h, w = feat.shape
        return feat, (h, w)
