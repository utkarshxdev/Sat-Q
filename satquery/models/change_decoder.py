"""
satquery/models/change_decoder.py
──────────────────────────────────
U-Net decoder for binary change mask prediction.

Consumes the abs-difference feature maps produced by siamese_unet.py at
each encoder scale and progressively upsamples them to the original image
resolution, outputting a single-channel probability map in [0, 1].

Architecture per decoder stage:
    Upsample(×2) → Concat(skip from next stage) → ConvBnRelu × 2

Final head:
    Conv2d(C, 1, 1) → Sigmoid → probability map (B, 1, H, W)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Single U-Net decoder block.

    Upsample → concat with skip → 2× ConvBnRelu.

    Args:
        in_channels   : Channels of the upsampled feature map.
        skip_channels : Channels of the skip connection (difference map).
        out_channels  : Output channels after convolutions.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_channels + skip_channels, out_channels),
            ConvBnRelu(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        # Bilinear upsample to match skip spatial dims
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ChangeDecoder(nn.Module):
    """
    U-Net decoder that converts multi-scale difference feature maps into a
    binary change probability map.

    Expects 4 skip connections from the Siamese encoder, matching ResNet34
    layer output channels: [64, 128, 256, 512].

    Args:
        encoder_channels : List of skip-connection channel counts
            (from deepest to shallowest, i.e. [512, 256, 128, 64]).
        decoder_channels : Output channels at each decoder stage.
        final_upscale    : Whether to bilinear upsample back to input resolution.
    """

    def __init__(
        self,
        encoder_channels: List[int] = [512, 256, 128, 64],
        decoder_channels: List[int] = [256, 128, 64, 32],
        final_upscale: bool = True,
    ) -> None:
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels), (
            "encoder_channels and decoder_channels must have the same length."
        )
        self.final_upscale = final_upscale

        # Bottleneck conv on the deepest difference map
        self.bottleneck = ConvBnRelu(encoder_channels[0], encoder_channels[0])

        # Build decoder blocks: progressively shallower
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[0]
        for skip_ch, out_ch in zip(encoder_channels[1:], decoder_channels[:-1]):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # Final decoder block has no skip connection (it's the shallowest output)
        self.final_block = nn.Sequential(
            ConvBnRelu(in_ch, decoder_channels[-1]),
            ConvBnRelu(decoder_channels[-1], decoder_channels[-1]),
        )

        # Single-channel output head
        self.head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(
        self,
        diff_maps: List[torch.Tensor],
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            diff_maps: List of abs-difference feature maps ordered
                       [deepest, ..., shallowest], i.e.
                       [F_diff_4, F_diff_3, F_diff_2, F_diff_1]
                       Channels: [512, 256, 128, 64]
            original_size: (H, W) of the input image for final upscaling.
                           If None, output is at decoder resolution.

        Returns:
            prob_map: (B, 1, H, W) with values in [0, 1]  (after sigmoid)
        """
        x = self.bottleneck(diff_maps[0])   # deepest feature

        for block, skip in zip(self.blocks, diff_maps[1:]):
            x = block(x, skip)

        x = self.final_block(x)

        if self.final_upscale and original_size is not None:
            x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(self.head(x))   # (B, 1, H, W)
