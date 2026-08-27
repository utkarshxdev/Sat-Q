"""
satquery/config.py
──────────────────
Central configuration for SatQuery AI.
All paths, hyperparameters, and device selection live here.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# ─── Root paths ───────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
CKPT_DIR   = ROOT_DIR / "checkpoints"
DATA_DIR   = ROOT_DIR / "data"
ONNX_DIR   = ROOT_DIR / "onnx_models"

CKPT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
ONNX_DIR.mkdir(exist_ok=True)

# ─── Device selection (Mac MPS → Colab CUDA → CPU) ───────────────────────────
def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE: torch.device = _select_device()

# ─── Image geometry ───────────────────────────────────────────────────────────
IMG_SIZE        = 512          # spatial resolution used throughout
OPTICAL_CHANNELS = 3           # RGB / Sentinel-2 B2-B3-B4
SAR_CHANNELS     = 2           # VV + VH (Sentinel-1)
CHANGE_IN_CHANNELS = 3         # change detection input channels (match backbone)

# ─── Fusion model ─────────────────────────────────────────────────────────────
FUSION_D_MODEL  = 768          # ViT-B/16 hidden dim
FUSION_HEADS    = 8
NUM_BIGEARTHNET_CLASSES = 19   # BigEarthNet-19 label schema

# ─── Change detection ─────────────────────────────────────────────────────────
CHANGE_THRESHOLD  = 0.5        # sigmoid threshold for binary mask
NOISE_THRESHOLD   = 0.005      # < 0.5% pixels changed → treat as noise

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8
LR_BACKBONE     = 1e-5         # frozen backbone fine-tuning LR (very small)
LR_HEAD         = 3e-4         # new head LR
WEIGHT_DECAY    = 1e-2
MAX_EPOCHS      = 30
WARMUP_EPOCHS   = 3

# ─── Focal loss ───────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0
FOCAL_ALPHA  = 0.75            # upweight rare change class

# ─── ONNX ─────────────────────────────────────────────────────────────────────
ONNX_OPSET  = 18
ONNX_CHANGE_PATH = ONNX_DIR / "siamese_change.onnx"
ONNX_FUSION_PATH = ONNX_DIR / "optical_sar_fused.onnx"

# ─── Pre-trained hub IDs ─────────────────────────────────────────────────────
# torchgeo SeCo ResNet18 trained on Sentinel-2 — used as change detection backbone
SECO_MODEL_ID   = "torchgeo/resnet18_sentinel2_rgb_seco"
# IBM/NASA Prithvi-100M — optional, heavier, for fusion encoder
PRITHVI_MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
