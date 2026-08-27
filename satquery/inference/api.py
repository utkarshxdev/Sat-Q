"""
satquery/inference/api.py
──────────────────────────
PRIMARY DELIVERABLE — SatQuery AI ML Core API Contracts.

Two deterministic functions wrap the full ML pipeline:

  get_change_analysis(img_t1, img_t2)  →  change mask + metadata
  extract_fused_features(optical, sar) →  fused embedding + attention map

Both functions:
  • Accept raw numpy arrays from the Geo Expert.
  • Preprocess internally (normalise, device placement).
  • Load model weights lazily (once per process, cached).
  • Return strictly typed output dicts for the Agentic Orchestrator.
  • Fall back gracefully if checkpoints are missing (demo/mock mode).

Device priority: CUDA → Apple MPS → CPU
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from satquery.config import (
    CHANGE_THRESHOLD, NOISE_THRESHOLD, DEVICE,
    CKPT_DIR, FUSION_D_MODEL, NUM_BIGEARTHNET_CLASSES,
)
from satquery.preprocessing.geo_interface import (
    preprocess_optical,
    preprocess_sar,
    preprocess_bitemporal,
)


# ─── Lazy model loaders (singleton pattern via lru_cache) ────────────────────

@lru_cache(maxsize=1)
def _load_change_model():
    """Load (or mock) the Siamese U-Net for change detection."""
    from satquery.models.siamese_unet import SiameseUNet
    model = SiameseUNet(in_channels=3, pretrained=False, freeze_stages=0).to(DEVICE)
    model.eval()

    ckpt_path = CKPT_DIR / "siamese_change.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[SiameseUNet] Loaded checkpoint: {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')})")
    else:
        warnings.warn(
            f"No checkpoint found at {ckpt_path}. "
            "Running in MOCK mode — outputs are random. "
            "Run satquery/training/train_change.py to generate a checkpoint.",
            UserWarning,
            stacklevel=2,
        )

    return model, ckpt_path.exists()


@lru_cache(maxsize=1)
def _load_fusion_model():
    """Load (or mock) the Optical-SAR Fusion Model."""
    from satquery.models.optical_sar_fusion import OpticalSARFusionModel
    model = OpticalSARFusionModel(
        opt_pretrained=False,
        sar_pretrained=False,
        freeze_opt_stages=0,
        freeze_sar_stages=0,
    ).to(DEVICE)
    model.eval()

    ckpt_path = CKPT_DIR / "optical_sar_fused.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[FusionModel] Loaded checkpoint: {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')})")
    else:
        warnings.warn(
            f"No checkpoint found at {ckpt_path}. "
            "Running in MOCK mode — outputs are random. "
            "Run satquery/training/train_fusion.py to generate a checkpoint.",
            UserWarning,
            stacklevel=2,
        )

    return model, ckpt_path.exists()


# ─── API Contract 1: Bi-Temporal Change Analysis ─────────────────────────────

def get_change_analysis(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    threshold: float = CHANGE_THRESHOLD,
    noise_floor: float = NOISE_THRESHOLD,
) -> dict:
    """
    Identifies structural/semantic changes between two co-registered
    multi-temporal remote-sensing images.

    Args:
        img_t1       : np.ndarray  float32  (C, H, W)  pre-change image  [0, 1]
        img_t2       : np.ndarray  float32  (C, H, W)  post-change image [0, 1]
                       Channels: 3 (RGB), 4 (RGB+NIR), or 13 (Sentinel-2 full)
        threshold    : Sigmoid threshold for binary change classification [0, 1].
        noise_floor  : Minimum changed-pixel fraction to report as a real change.
                       Changes below this are zeroed out (noise suppression).

    Returns:
        dict with the following guaranteed keys:
        ┌─────────────────────┬───────────────────────────────────────────────┐
        │ Key                 │ Type / Shape / Description                    │
        ├─────────────────────┼───────────────────────────────────────────────┤
        │ change_mask         │ np.ndarray uint8 (H, W)  values {0, 255}      │
        │                     │ 255 = changed, 0 = unchanged                  │
        │ change_detected     │ bool                                          │
        │ confidence          │ float  0.0–1.0  mean prob of changed pixels   │
        │ changed_area_pct    │ float  percentage of total pixels flagged     │
        │ summary             │ str    human-readable result for LLM router   │
        │ model_used          │ str    model identifier for audit log         │
        │ checkpoint_loaded   │ bool   True if real weights were used         │
        └─────────────────────┴───────────────────────────────────────────────┘

    Raises:
        ValueError: If input shapes are incompatible (propagated from geo_interface).
    """
    # ── 1. Preprocess ─────────────────────────────────────────────────────
    t1_tensor, t2_tensor = preprocess_bitemporal(img_t1, img_t2)
    # Resize tensors to 3-channel if needed (model trained on RGB)
    t1_tensor = _ensure_3ch(t1_tensor)
    t2_tensor = _ensure_3ch(t2_tensor)

    h, w = img_t1.shape[1], img_t1.shape[2]

    # ── 2. Inference ──────────────────────────────────────────────────────
    model, ckpt_loaded = _load_change_model()
    with torch.no_grad():
        prob_map: torch.Tensor = model(t1_tensor, t2_tensor)   # (1, 1, H, W)

    prob_np: np.ndarray = prob_map.squeeze().cpu().numpy()    # (H, W) float32

    # ── 3. Threshold → binary mask ────────────────────────────────────────
    binary_mask: np.ndarray = (prob_np >= threshold).astype(np.uint8) * 255

    # ── 4. Empty-mask defense ─────────────────────────────────────────────
    total_pixels    = h * w
    changed_pixels  = int(np.sum(binary_mask == 255))
    changed_area_pct = changed_pixels / total_pixels

    if changed_area_pct < noise_floor:
        # Below noise floor → zero mask, report no change
        binary_mask      = np.zeros((h, w), dtype=np.uint8)
        change_detected  = False
        confidence       = 0.0
        changed_area_pct = 0.0
    else:
        change_detected = True
        # Confidence = mean probability of ONLY the changed pixels
        confidence = float(np.mean(prob_np[binary_mask == 255]))

    # ── 5. Build response ─────────────────────────────────────────────────
    pct_rounded = round(changed_area_pct * 100, 2)
    summary = (
        f"Detected {pct_rounded}% spatial change between T1 and T2 "
        f"(confidence: {round(confidence, 3)})."
        if change_detected
        else "No significant change detected between T1 and T2 (below noise threshold)."
    )

    return {
        "change_mask":       binary_mask,            # uint8 (H, W) {0, 255}
        "change_detected":   change_detected,         # bool
        "confidence":        round(confidence, 3),    # float [0, 1]
        "changed_area_pct":  pct_rounded,             # float (percentage)
        "summary":           summary,                 # str
        "model_used":        "SiameseUNet-ResNet34",  # audit log
        "checkpoint_loaded": ckpt_loaded,             # bool
    }


# ─── API Contract 2: Optical-SAR Feature Fusion ──────────────────────────────

def extract_fused_features(
    optical: np.ndarray,
    sar: np.ndarray,
) -> dict:
    """
    Fuses optical (spectral/contextual) and SAR (structural/backscatter) imagery
    via cross-attention to produce a joint embedding for downstream tasks
    (VQA, captioning, land-cover classification).

    Args:
        optical : np.ndarray  float32  (3, H, W)   RGB or B2-B3-B4 in [0, 1].
        sar     : np.ndarray  float32  (2, H, W)   VV/VH Sentinel-1 in [0, 1].

    Returns:
        dict with the following guaranteed keys:
        ┌─────────────────────────┬─────────────────────────────────────────────┐
        │ Key                     │ Type / Shape / Description                  │
        ├─────────────────────────┼─────────────────────────────────────────────┤
        │ fused_embedding         │ np.ndarray float32 (D,)  global feature vec │
        │                         │ D = FUSION_D_MODEL (768)                    │
        │ land_cover_logits       │ np.ndarray float32 (19,) BigEarthNet logits │
        │ land_cover_probs        │ np.ndarray float32 (19,) after sigmoid      │
        │ attention_map           │ np.ndarray float32 (H_attn, W_attn)         │
        │                         │ Cross-attention weights (optical→SAR)        │
        │ status                  │ str  "ok" | "mock"                          │
        │ model_used              │ str  model identifier for audit log         │
        │ checkpoint_loaded       │ bool                                        │
        └─────────────────────────┴─────────────────────────────────────────────┘

    Raises:
        ValueError: If optical or SAR array shapes are invalid.
    """
    # ── 1. Preprocess ─────────────────────────────────────────────────────
    opt_tensor = preprocess_optical(optical)    # (1, 3, H, W)
    sar_tensor = preprocess_sar(sar)            # (1, 2, H, W)

    # Resize to (224, 224) for ViT patch embedding compatibility
    if opt_tensor.shape[-2:] != (224, 224):
        import torch.nn.functional as F
        opt_tensor = F.interpolate(opt_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        sar_tensor = F.interpolate(sar_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    # ── 2. Inference ──────────────────────────────────────────────────────
    model, ckpt_loaded = _load_fusion_model()
    with torch.no_grad():
        output = model(opt_tensor, sar_tensor, mode="classify")

    embedding    = output["embedding"].squeeze(0).cpu().numpy()    # (D,)
    logits       = output["logits"].squeeze(0).cpu().numpy()       # (19,)
    attn_weights = output["attn_weights"].squeeze(0).cpu().numpy() # (N_opt, N_sar)

    # ── 3. Reshape attention map to 2D spatial grid ───────────────────────
    # ViT-B/16 at 224px: N_opt = 196 + 1 (CLS) = 197
    # Drop CLS token and reshape to (H_p, W_p)
    attn_no_cls = attn_weights[1:] if attn_weights.shape[0] > 196 else attn_weights
    h_p = int(attn_no_cls.shape[0] ** 0.5)
    w_p = h_p
    if h_p * w_p == attn_no_cls.shape[0]:
        attn_map_2d = attn_no_cls.mean(axis=-1).reshape(h_p, w_p)  # avg over SAR tokens
    else:
        attn_map_2d = attn_weights.mean(axis=-1)   # 1D fallback

    # ── 4. Build response ─────────────────────────────────────────────────
    return {
        "fused_embedding":   embedding.astype(np.float32),            # (D,)
        "land_cover_logits": logits.astype(np.float32),               # (19,)
        "land_cover_probs":  _sigmoid_np(logits).astype(np.float32),  # (19,)
        "attention_map":     attn_map_2d.astype(np.float32),          # (H_p, W_p)
        "status":            "ok" if ckpt_loaded else "mock",
        "model_used":        "OpticalSARFusion-ViTB-ConvNeXtT",
        "checkpoint_loaded": ckpt_loaded,
    }


# ─── Utility helpers ─────────────────────────────────────────────────────────

def _ensure_3ch(tensor: torch.Tensor) -> torch.Tensor:
    """If tensor has >3 channels, select first 3. If <3, repeat to fill."""
    b, c, h, w = tensor.shape
    if c == 3:
        return tensor
    if c > 3:
        return tensor[:, :3, :, :]
    # c < 3: repeat last channel
    return tensor.expand(b, 3, h, w)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
