"""
satquery/preprocessing/geo_interface.py
────────────────────────────────────────
Mock Geo Expert interface.

Accepts raw numpy arrays from the geospatial pipeline (C, H, W) and returns
validated, normalised torch.Tensor pairs ready for model inference.

The Geo Expert produces:
  • optical : np.ndarray float32  shape (C_opt, H, W)  range [0, 1] or raw DN
  • sar      : np.ndarray float32  shape (C_sar, H, W)  range [0, 1] or raw dB
  • img_t1   : np.ndarray float32  shape (C, H, W)      pre-change
  • img_t2   : np.ndarray float32  shape (C, H, W)      post-change

All outputs are returned as float32 tensors with values in [0, 1].
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from satquery.config import DEVICE, IMG_SIZE, OPTICAL_CHANNELS, SAR_CHANNELS


# ─── Normalisation statistics (Sentinel-2 RGB bands from BigEarthNet) ─────────
_OPT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet proxy
_OPT_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# SAR Sentinel-1 VV/VH after dB-to-linear conversion, typical range 0–1
_SAR_MEAN = np.array([0.33, 0.20], dtype=np.float32)
_SAR_STD  = np.array([0.17, 0.14], dtype=np.float32)


def _validate_array(arr: np.ndarray, expected_channels: int, name: str) -> None:
    """Raise a descriptive ValueError if the array doesn't meet contract."""
    if arr.ndim != 3:
        raise ValueError(
            f"{name}: expected 3-D array (C, H, W), got shape {arr.shape}"
        )
    c, h, w = arr.shape
    if c != expected_channels:
        raise ValueError(
            f"{name}: expected {expected_channels} channels, got {c}"
        )
    if h < 32 or w < 32:
        raise ValueError(
            f"{name}: spatial resolution ({h}×{w}) is below minimum 32×32"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: array contains NaN or Inf values")


def _clip_normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Clip to [0,1] then ImageNet-style channel normalise."""
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    # (C, H, W) → normalise per channel
    arr = (arr - mean[:, None, None]) / (std[:, None, None] + 1e-8)
    return arr


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """np.ndarray (C,H,W) → torch.Tensor (1,C,H,W) on DEVICE."""
    t = torch.from_numpy(arr.copy()).float().unsqueeze(0)
    return t.to(DEVICE)


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess_optical(optical: np.ndarray) -> torch.Tensor:
    """
    Validate and normalise a single optical image.

    Args:
        optical: float32 (3, H, W) in [0, 1].

    Returns:
        torch.Tensor (1, 3, H, W) on DEVICE, ImageNet-normalised.
    """
    _validate_array(optical, OPTICAL_CHANNELS, "optical")
    normed = _clip_normalize(optical, _OPT_MEAN, _OPT_STD)
    return _to_tensor(normed)


def preprocess_sar(sar: np.ndarray) -> torch.Tensor:
    """
    Validate and normalise a SAR image (VV, VH channels).

    Args:
        sar: float32 (2, H, W) in [0, 1] after dB conversion.

    Returns:
        torch.Tensor (1, 2, H, W) on DEVICE, channel-normalised.
    """
    _validate_array(sar, SAR_CHANNELS, "sar")
    normed = _clip_normalize(sar, _SAR_MEAN, _SAR_STD)
    return _to_tensor(normed)


def preprocess_bitemporal(
    img_t1: np.ndarray, img_t2: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and normalise a bi-temporal image pair for change detection.

    Both images must share the same spatial dimensions and channel count.

    Args:
        img_t1: float32 (C, H, W) – pre-change image in [0, 1].
        img_t2: float32 (C, H, W) – post-change image in [0, 1].

    Returns:
        Tuple of tensors (1, C, H, W) on DEVICE, ImageNet-normalised.
    """
    if img_t1.shape != img_t2.shape:
        raise ValueError(
            f"Bi-temporal shapes must match: "
            f"T1={img_t1.shape} vs T2={img_t2.shape}"
        )
    _validate_array(img_t1, img_t1.shape[0], "img_t1")
    _validate_array(img_t2, img_t2.shape[0], "img_t2")
    c = img_t1.shape[0]
    # Use optical normalisation stats regardless of channel count (generalises
    # to grayscale or pseudo-colour inputs from LEVIR-CD)
    mean = _OPT_MEAN[:c] if c <= 3 else np.full(c, 0.5, dtype=np.float32)
    std  = _OPT_STD[:c]  if c <= 3 else np.full(c, 0.25, dtype=np.float32)

    normed_t1 = _clip_normalize(img_t1, mean, std)
    normed_t2 = _clip_normalize(img_t2, mean, std)
    return _to_tensor(normed_t1), _to_tensor(normed_t2)
