"""
satquery/models/optical_sar_fusion.py
──────────────────────────────────────
Full Optical-SAR Fusion Model.

Architecture:
    OpticalEncoder  →  ViT-B/16 tokens  (B, N, D)
    SAREncoder      →  ConvNeXt feature map  →  SARSpatialAdapter  (B, N_sar, D)
    CrossAttentionBlock  →  enriched optical tokens  (B, N, D)
    FusionPooler    →  global embedding  (B, D)
    ClassHead       →  multi-label logits  (B, 19)  [BigEarthNet-19]

Training objective: BCEWithLogitsLoss on 19 land-cover labels.

The model supports two operating modes:
    mode='classify'  → returns (logits, embedding, attn_map)  [training]
    mode='embed'     → returns (embedding, attn_map)           [inference API]
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from satquery.config import FUSION_D_MODEL, NUM_BIGEARTHNET_CLASSES
from satquery.models.encoders import OpticalEncoder, SAREncoder
from satquery.models.cross_attention import (
    SARSpatialAdapter,
    CrossAttentionBlock,
    FusionPooler,
)


class OpticalSARFusionModel(nn.Module):
    """
    Dual-encoder cross-attention fusion model for optical + SAR remote-sensing.

    Args:
        num_classes     : Number of output labels (19 for BigEarthNet-19).
        opt_pretrained  : Use pretrained ViT-B/16 ImageNet weights.
        sar_pretrained  : Use pretrained ConvNeXt-Tiny ImageNet weights.
        freeze_opt_stages: Number of ViT blocks to freeze (default 8 of 12).
        freeze_sar_stages: Number of ConvNeXt stages to freeze (default 2 of 4).
        dropout         : Dropout rate in cross-attention and classifier.
    """

    def __init__(
        self,
        num_classes: int = NUM_BIGEARTHNET_CLASSES,
        opt_pretrained: bool = True,
        sar_pretrained: bool = True,
        freeze_opt_stages: int = 8,
        freeze_sar_stages: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────────────────
        self.optical_encoder = OpticalEncoder(
            pretrained=opt_pretrained,
            freeze_stages=freeze_opt_stages,
        )
        self.sar_encoder = SAREncoder(
            pretrained=sar_pretrained,
            freeze_stages=freeze_sar_stages,
        )

        # ── Cross-attention fusion ────────────────────────────────────────────
        self.sar_adapter = SARSpatialAdapter(max_spatial_tokens=256)
        self.cross_attn  = CrossAttentionBlock(
            d_model=FUSION_D_MODEL,
            dropout=dropout,
        )
        self.pooler = FusionPooler(use_cls_token=True)

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(FUSION_D_MODEL),
            nn.Dropout(dropout),
            nn.Linear(FUSION_D_MODEL, num_classes),
        )

    def forward(
        self,
        optical: torch.Tensor,   # (B, 3, H, W)
        sar: torch.Tensor,       # (B, 2, H, W)
        mode: Literal["classify", "embed"] = "classify",
    ) -> dict:
        """
        Args:
            optical : (B, 3, H, W)  ImageNet-normalised optical image.
            sar     : (B, 2, H, W)  channel-normalised SAR image.
            mode    : 'classify' returns logits; 'embed' skips classifier head.

        Returns:
            dict with keys:
                'embedding'   : (B, D)        global fused feature vector
                'logits'      : (B, C)        raw class logits  [classify mode only]
                'attn_weights': (B, N_opt, N_sar)  cross-attention map
        """
        # 1. Encode optical → patch tokens
        opt_tokens: torch.Tensor = self.optical_encoder(optical)  # (B, N, D)

        # 2. Encode SAR → spatial feature map → token sequence
        sar_feat_map, _ = self.sar_encoder(sar)                   # (B, D, H', W')
        sar_tokens: torch.Tensor = self.sar_adapter(sar_feat_map) # (B, N_sar, D)

        # 3. Cross-attention: optical attends to SAR
        fused_tokens, attn_weights = self.cross_attn(opt_tokens, sar_tokens)

        # 4. Pool to single embedding
        embedding = self.pooler(fused_tokens)  # (B, D)

        result = {
            "embedding": embedding,
            "attn_weights": attn_weights,
        }

        if mode == "classify":
            result["logits"] = self.classifier(embedding)   # (B, num_classes)

        return result

    def load_seco_backbone(self, checkpoint_path: str) -> None:
        """
        Load SeCo (Seasonal Contrast) pretrained weights into the optical encoder.
        SeCo checkpoints are ResNet-based; this helper maps compatible layer names.
        For ViT-based SeCo variants, load directly via timm's from_pretrained.

        Usage:
            model.load_seco_backbone("seco_resnet50.pth")
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        # Strip module. prefix if saved with DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.optical_encoder.backbone.load_state_dict(
            state, strict=False
        )
        print(f"[SeCo] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
