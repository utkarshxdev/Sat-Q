"""
satquery/inference/onnx_runner.py
───────────────────────────────────
ONNX Runtime inference engine — drop-in replacement for the PyTorch path.

Swaps heavy PyTorch forward passes with ultra-fast onnxruntime sessions.
Automatically selects CUDAExecutionProvider on Colab and
CoreMLExecutionProvider on Mac (fallback: CPUExecutionProvider).

Usage:
    from satquery.inference.onnx_runner import OnnxChangeDetector, OnnxFusionModel

    detector = OnnxChangeDetector()
    prob_map = detector.run(t1_np, t2_np)   # (H, W) float32

    fusion   = OnnxFusionModel()
    emb, logits = fusion.run(opt_np, sar_np)
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Tuple

import numpy as np

from satquery.config import ONNX_DIR


def _build_session(onnx_path: str):
    """Build an onnxruntime InferenceSession with best available provider."""
    import onnxruntime as ort

    available = ort.get_available_providers()

    # Provider priority: CUDA > CoreML (Mac) > CPU
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"[ONNX] Session providers: {providers[0]}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


class OnnxChangeDetector:
    """
    ONNX-accelerated Siamese Change Detector.

    Args:
        onnx_path : Path to siamese_change.onnx. Defaults to ONNX_DIR.
    """

    def __init__(self, onnx_path: str | None = None) -> None:
        path = str(onnx_path or ONNX_DIR / "siamese_change.onnx")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                "Run: python satquery/inference/onnx_export.py --model change"
            )
        self.session = _build_session(path)
        self.input_names  = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            img_t1, img_t2 : float32 (1, C, H, W) or (C, H, W).
        Returns:
            prob_map : float32 (H, W) change probability in [0, 1].
        """
        t1 = _ensure_4d(img_t1)
        t2 = _ensure_4d(img_t2)

        feeds = {self.input_names[0]: t1, self.input_names[1]: t2}
        outputs = self.session.run(self.output_names, feeds)
        return outputs[0].squeeze()   # (H, W)


class OnnxFusionModel:
    """
    ONNX-accelerated Optical-SAR Fusion Model.

    Args:
        onnx_path : Path to optical_sar_fused.onnx. Defaults to ONNX_DIR.
    """

    def __init__(self, onnx_path: str | None = None) -> None:
        path = str(onnx_path or ONNX_DIR / "optical_sar_fused.onnx")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                "Run: python satquery/inference/onnx_export.py --model fusion"
            )
        self.session = _build_session(path)
        self.input_names  = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            optical : float32 (1, 3, 224, 224) or (3, 224, 224).
            sar     : float32 (1, 2, 224, 224) or (2, 224, 224).
        Returns:
            embedding : float32 (768,)
            logits    : float32 (19,)
        """
        opt = _ensure_4d(optical)
        s   = _ensure_4d(sar)

        feeds   = {self.input_names[0]: opt, self.input_names[1]: s}
        outputs = self.session.run(self.output_names, feeds)
        embedding = outputs[0].squeeze()  # (768,)
        logits    = outputs[1].squeeze()  # (19,)
        return embedding, logits


def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Add batch dim if missing."""
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    return arr.astype(np.float32)
