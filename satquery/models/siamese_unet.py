"""
satquery/models/siamese_unet.py
────────────────────────────────
Siamese U-Net for bi-temporal semantic change detection.

Design:
  • Shared ResNet34 encoder (weights shared — true Siamese).
  • Multi-scale absolute difference: F_diff[i] = |T1_feats[i] - T2_feats[i]|
    computed at each of the 4 ResNet skip-connection levels.
  • ChangeDecoder upsamples and predicts a binary probability map.

The backbone can be swapped for the SeCo pretrained ResNet18 or ResNet50
by changing `backbone_name`. The key requirement is it must expose
multi-scale intermediate feature maps.

Architecture overview:
    img_T1 ──┐
             ├── SharedResNet34 ──→ [f1, f2, f3, f4]  per time step
    img_T2 ──┘                      (T1 and T2 through the same weights)

    F_diff_i = |T1_feats_i - T2_feats_i|               (4 difference maps)

    ChangeDecoder([F_diff_4, F_diff_3, F_diff_2, F_diff_1])
         → prob_map (B, 1, H, W) ∈ [0, 1]
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except ImportError:
    _SMP_AVAILABLE = False

import timm

from satquery.models.change_decoder import ChangeDecoder


# ResNet34 layer output channels (before the global pool)
_RESNET34_CHANNELS = [64, 128, 256, 512]   # layer1 → layer4


class _ResNet34FeatureExtractor(nn.Module):
    """
    Wraps a timm ResNet34 backbone to return multi-scale feature maps
    from all 4 residual stages without the classification head.

    Returns features in order [layer1_out, layer2_out, layer3_out, layer4_out]
    i.e. spatial strides [4, 8, 16, 32] relative to input.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        backbone = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            features_only=True,     # expose intermediate feature maps
            out_indices=(1, 2, 3, 4),  # layer1..layer4
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            List of 4 tensors with channels [64, 128, 256, 512]
            and spatial sizes [H/4, H/8, H/16, H/32].
        """
        return self.backbone(x)   # [f1, f2, f3, f4]


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for bi-temporal change detection.

    The Siamese encoder uses SHARED weights — both time-step images pass
    through the identical parameter set. This ensures the feature space
    is view-consistent, so only genuine semantic changes produce large
    absolute differences.

    Args:
        in_channels     : Input image channels (default 3 for RGB).
        pretrained      : Use ImageNet pretrained ResNet34 weights.
        freeze_stages   : Freeze first N ResNet stages (0-3). Stage 0 = layer1.
                          Recommended: 2 (freeze layer1 + layer2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_stages: int = 2,
    ) -> None:
        super().__init__()

        # Shared encoder (same object, same parameters — true Siamese)
        self.encoder = _ResNet34FeatureExtractor(pretrained=pretrained)

        # Adapt first conv if in_channels != 3
        if in_channels != 3:
            old_conv = self.encoder.backbone.conv1 if hasattr(
                self.encoder.backbone, "conv1"
            ) else list(self.encoder.backbone.children())[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(self.encoder.backbone, "conv1"):
                self.encoder.backbone.conv1 = new_conv

        # Freeze early stages to preserve low-level pretrained features
        backbone_stages = [
            self.encoder.backbone.layer1,
            self.encoder.backbone.layer2,
            self.encoder.backbone.layer3,
            self.encoder.backbone.layer4,
        ] if hasattr(self.encoder.backbone, "layer1") else []

        for i, stage in enumerate(backbone_stages[:freeze_stages]):
            for param in stage.parameters():
                param.requires_grad = False

        # Decoder — expects diff maps ordered deepest→shallowest
        self.decoder = ChangeDecoder(
            encoder_channels=[512, 256, 128, 64],
            decoder_channels=[256, 128, 64, 32],
            final_upscale=True,
        )

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run one image through the shared encoder."""
        return self.encoder(x)   # [f1, f2, f3, f4]

    def _compute_difference_maps(
        self,
        feats_t1: List[torch.Tensor],
        feats_t2: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute per-scale absolute difference maps.
        F_diff[i] = |feats_t1[i] - feats_t2[i]|
        """
        return [torch.abs(f1 - f2) for f1, f2 in zip(feats_t1, feats_t2)]

    def forward(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_t1: (B, C, H, W) — pre-change image, normalised.
            img_t2: (B, C, H, W) — post-change image, normalised.
                    Must share identical spatial dimensions with img_t1.

        Returns:
            prob_map: (B, 1, H, W) ∈ [0, 1]
                      Values close to 1 indicate changed pixels.
        """
        if img_t1.shape != img_t2.shape:
            raise ValueError(
                f"img_t1 {img_t1.shape} and img_t2 {img_t2.shape} must match."
            )

        original_size = (img_t1.shape[2], img_t1.shape[3])

        # Siamese forward pass — SAME encoder object, called twice
        feats_t1 = self._extract_features(img_t1)  # [f1..f4]
        feats_t2 = self._extract_features(img_t2)  # [f1..f4]

        # Abs-difference at each scale: deepest first for decoder
        diff_maps = self._compute_difference_maps(feats_t1, feats_t2)
        diff_maps_reversed = diff_maps[::-1]  # [f4_diff, f3_diff, f2_diff, f1_diff]

        # Decode to full-resolution probability map
        prob_map = self.decoder(diff_maps_reversed, original_size=original_size)

        return prob_map   # (B, 1, H, W)

    def predict(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns (prob_map, binary_mask).
        Binary mask has values {0, 1} (float).

        Args:
            img_t1, img_t2: As in forward().
            threshold      : Decision boundary for change classification.

        Returns:
            (prob_map, binary_mask) both shape (B, 1, H, W)
        """
        with torch.no_grad():
            prob_map = self.forward(img_t1, img_t2)
        binary_mask = (prob_map >= threshold).float()
        return prob_map, binary_mask
