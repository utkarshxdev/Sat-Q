"""
satquery/inference/onnx_export.py
───────────────────────────────────
Exports trained PyTorch checkpoints to static ONNX graphs.

Usage (after training):
    python satquery/inference/onnx_export.py --model change
    python satquery/inference/onnx_export.py --model fusion
    python satquery/inference/onnx_export.py --model all

Exports:
    onnx_models/siamese_change.onnx   — SiameseUNet (dynamic H/W axes)
    onnx_models/optical_sar_fused.onnx — FusionModel (fixed 224x224)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from satquery.config import CKPT_DIR, ONNX_DIR, ONNX_OPSET, DEVICE


def export_change_model() -> None:
    """Export SiameseUNet to ONNX with dynamic spatial dimensions."""
    from satquery.models.siamese_unet import SiameseUNet

    ckpt_path = CKPT_DIR / "siamese_change.pth"
    onnx_path = ONNX_DIR / "siamese_change.onnx"

    model = SiameseUNet(in_channels=3, pretrained=False, freeze_stages=0)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [Warning] No checkpoint found. Exporting untrained weights.")

    model.eval()

    # Dummy inputs: (batch=1, channels=3, H=256, W=256)
    dummy_t1 = torch.randn(1, 3, 256, 256)
    dummy_t2 = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        args=(dummy_t1, dummy_t2),
        f=str(onnx_path),
        opset_version=ONNX_OPSET,
        input_names=["img_t1", "img_t2"],
        output_names=["prob_map"],
        dynamic_axes={
            "img_t1":   {0: "batch", 2: "height", 3: "width"},
            "img_t2":   {0: "batch", 2: "height", 3: "width"},
            "prob_map": {0: "batch", 2: "height", 3: "width"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ✓ SiameseUNet exported → {onnx_path}")
    _verify_onnx(str(onnx_path))


def export_fusion_model() -> None:
    """Export OpticalSARFusionModel to ONNX (fixed 224×224 input)."""
    from satquery.models.optical_sar_fusion import OpticalSARFusionModel

    ckpt_path = CKPT_DIR / "optical_sar_fused.pth"
    onnx_path = ONNX_DIR / "optical_sar_fused.onnx"

    model = OpticalSARFusionModel(
        opt_pretrained=False, sar_pretrained=False,
        freeze_opt_stages=0, freeze_sar_stages=0,
    )
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [Warning] No checkpoint found. Exporting untrained weights.")

    model.eval()

    # ViT expects 224x224; ConvNeXt handles other sizes but ONNX export is fixed
    dummy_opt = torch.randn(1, 3, 224, 224)
    dummy_sar = torch.randn(1, 2, 224, 224)

    # Export only embedding + logits (skip attn_weights which are dict-unpacked)
    class _FusionWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.m.eval()

        def forward(self, opt, sar):
            out = self.m(opt, sar, mode="classify")
            return out["embedding"], out["logits"]

    wrapper = _FusionWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        args=(dummy_opt, dummy_sar),
        f=str(onnx_path),
        opset_version=ONNX_OPSET,
        input_names=["optical", "sar"],
        output_names=["embedding", "logits"],
        dynamic_axes={
            "optical":   {0: "batch"},
            "sar":       {0: "batch"},
            "embedding": {0: "batch"},
            "logits":    {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ✓ FusionModel exported → {onnx_path}")
    _verify_onnx(str(onnx_path))


def _verify_onnx(path: str) -> None:
    """Run onnx.checker to validate the exported graph."""
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"  ✓ ONNX graph verified: {path}")
    except Exception as e:
        print(f"  [Warning] ONNX verification failed: {e}")


def main(args: argparse.Namespace) -> None:
    print(f"[ONNX Export] opset={ONNX_OPSET}")
    if args.model in ("change", "all"):
        print("\nExporting SiameseUNet (change detection)...")
        export_change_model()
    if args.model in ("fusion", "all"):
        print("\nExporting OpticalSARFusionModel...")
        export_fusion_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "--model",
        choices=["change", "fusion", "all"],
        default="all",
        help="Which model to export.",
    )
    main(parser.parse_args())
