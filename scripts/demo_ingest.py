"""
scripts/demo_ingest.py
───────────────────────
Phase 1 sanity demonstration.

Creates synthetic multi-channel numpy arrays simulating Geo Expert output,
preprocesses them, and runs a full forward pass through both model pipelines.
Reports tensor shapes, device placement, and peak memory usage.

Usage:
    python scripts/demo_ingest.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

# ── Setup ─────────────────────────────────────────────────────────────────────
from satquery.config import DEVICE, IMG_SIZE

print(f"\n{'='*60}")
print(f"SatQuery AI — Demo Ingest (Phase 1 Sanity Check)")
print(f"Device: {DEVICE}")
print(f"{'='*60}\n")


def _peak_memory_mb() -> float:
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if DEVICE.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


# ── 1. Simulate Geo Expert output ────────────────────────────────────────────
print("[1] Generating synthetic arrays (C, H, W) from Geo Expert mock...")

rng = np.random.default_rng(0)

# Sentinel-2 full 13-band image (full multispectral)
ms_image   = rng.random((13, IMG_SIZE, IMG_SIZE)).astype(np.float32)

# Sentinel-1 SAR VV/VH pair
sar_image  = rng.random((2,  IMG_SIZE, IMG_SIZE)).astype(np.float32)

# Bi-temporal RGB pair (LEVIR-CD format)
img_t1     = rng.random((3,  IMG_SIZE, IMG_SIZE)).astype(np.float32)
img_t2     = rng.random((3,  IMG_SIZE, IMG_SIZE)).astype(np.float32)
# Force obvious change in upper-left quadrant for visual demo
img_t2[:, :IMG_SIZE//4, :IMG_SIZE//4] = 1.0 - img_t2[:, :IMG_SIZE//4, :IMG_SIZE//4]

print(f"  Sentinel-2 13-band: {ms_image.shape}  dtype={ms_image.dtype}")
print(f"  SAR VV/VH:          {sar_image.shape}  dtype={sar_image.dtype}")
print(f"  T1 RGB:             {img_t1.shape}    dtype={img_t1.dtype}")
print(f"  T2 RGB:             {img_t2.shape}    dtype={img_t2.dtype}")


# ── 2. Preprocess via Geo Interface ──────────────────────────────────────────
print("\n[2] Preprocessing via geo_interface...")
from satquery.preprocessing.geo_interface import (
    preprocess_optical, preprocess_sar, preprocess_bitemporal
)

# Use first 3 bands of MS as optical input
optical_3ch = ms_image[:3]
opt_tensor  = preprocess_optical(optical_3ch)
sar_tensor  = preprocess_sar(sar_image)
t1_tensor, t2_tensor = preprocess_bitemporal(img_t1, img_t2)

print(f"  opt_tensor:  {opt_tensor.shape}   device={opt_tensor.device}")
print(f"  sar_tensor:  {sar_tensor.shape}   device={sar_tensor.device}")
print(f"  t1_tensor:   {t1_tensor.shape}    device={t1_tensor.device}")
print(f"  t2_tensor:   {t2_tensor.shape}    device={t2_tensor.device}")


# ── 3. Siamese UNet forward pass ─────────────────────────────────────────────
print("\n[3] Siamese UNet (change detection) forward pass...")
from satquery.models.siamese_unet import SiameseUNet

model_cd = SiameseUNet(in_channels=3, pretrained=False, freeze_stages=0).to(DEVICE)
model_cd.eval()

t0 = time.perf_counter()
with torch.no_grad():
    prob_map = model_cd(t1_tensor, t2_tensor)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"  prob_map shape: {prob_map.shape}")
print(f"  prob range:     [{prob_map.min().item():.3f}, {prob_map.max().item():.3f}]")
print(f"  Inference time: {elapsed_ms:.1f} ms")
print(f"  Peak VRAM:      {_peak_memory_mb():.1f} MB")

del model_cd
gc.collect()


# ── 4. Fusion Model forward pass ─────────────────────────────────────────────
print("\n[4] Optical-SAR Fusion Model forward pass (224×224)...")
from satquery.models.optical_sar_fusion import OpticalSARFusionModel
import torch.nn.functional as F

model_fuse = OpticalSARFusionModel(
    opt_pretrained=False, sar_pretrained=False,
    freeze_opt_stages=0, freeze_sar_stages=0,
).to(DEVICE)
model_fuse.eval()

# Resize to 224 (ViT requirement)
opt_224 = F.interpolate(opt_tensor, size=(224, 224), mode="bilinear", align_corners=False)
sar_224 = F.interpolate(sar_tensor, size=(224, 224), mode="bilinear", align_corners=False)

t0 = time.perf_counter()
with torch.no_grad():
    fuse_out = model_fuse(opt_224, sar_224, mode="classify")
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"  embedding shape: {fuse_out['embedding'].shape}")
print(f"  logits shape:    {fuse_out['logits'].shape}")
print(f"  attn_weights:    {fuse_out['attn_weights'].shape}")
print(f"  Inference time:  {elapsed_ms:.1f} ms")
print(f"  Peak VRAM:       {_peak_memory_mb():.1f} MB")

del model_fuse
gc.collect()


# ── 5. API-level demo ─────────────────────────────────────────────────────────
print("\n[5] API function demo (mock mode — no checkpoint needed)...")
from satquery.inference.api import get_change_analysis, extract_fused_features

change_result = get_change_analysis(img_t1, img_t2)
fused_result  = extract_fused_features(optical_3ch, sar_image)

print("\n  get_change_analysis() output:")
for k, v in change_result.items():
    if isinstance(v, np.ndarray):
        print(f"    {k}: np.ndarray {v.shape} dtype={v.dtype}")
    else:
        print(f"    {k}: {v!r}")

print("\n  extract_fused_features() output:")
for k, v in fused_result.items():
    if isinstance(v, np.ndarray):
        print(f"    {k}: np.ndarray {v.shape} dtype={v.dtype}")
    else:
        print(f"    {k}: {v!r}")


# ── 6. Batch allocation stress test (batch=16, 512×512) ──────────────────────
print("\n[6] Batch=16, 512×512 multi-spectral allocation stress test...")
try:
    batch = torch.rand(16, 13, 512, 512).to(DEVICE)
    print(f"  ✓ Allocated {batch.numel() * 4 / 1e6:.1f} MB tensor on {DEVICE}")
    print(f"  Peak VRAM: {_peak_memory_mb():.1f} MB")
    del batch
    gc.collect()
except RuntimeError as e:
    print(f"  [Warning] Allocation failed (insufficient memory): {e}")

print(f"\n{'='*60}")
print("Demo complete — all systems operational.")
print(f"{'='*60}\n")
