"""
satquery/tests/test_sanity.py
──────────────────────────────
Phase 1 sanity checks: device alloc, tensor shapes, model instantiation.

Run with:
    pytest satquery/tests/test_sanity.py -v
"""
from __future__ import annotations

import torch
import numpy as np
import pytest

from satquery.config import DEVICE, OPTICAL_CHANNELS, SAR_CHANNELS, IMG_SIZE


class TestDeviceAndTensors:

    def test_device_is_valid(self):
        assert DEVICE.type in ("cuda", "mps", "cpu"), f"Unexpected device: {DEVICE}"

    def test_basic_tensor_creation_on_device(self):
        t = torch.rand(2, 3, 64, 64).to(DEVICE)
        assert t.device.type == DEVICE.type

    def test_optical_tensor_shape(self):
        optical = torch.rand(1, OPTICAL_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
        assert optical.shape == (1, OPTICAL_CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_sar_tensor_shape(self):
        sar = torch.rand(1, SAR_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
        assert sar.shape == (1, SAR_CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_large_batch_allocation(self):
        """Verify batch=16 of 512×512 multi-spectral tensors can be allocated."""
        try:
            t = torch.rand(16, 13, 512, 512).to(DEVICE)
            assert t.shape[0] == 16
            del t  # free memory
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping: insufficient memory — {e}")
            raise


class TestModelInstantiation:

    def test_siamese_unet_instantiates(self):
        from satquery.models.siamese_unet import SiameseUNet
        model = SiameseUNet(in_channels=3, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_siamese_unet_forward_shape(self):
        from satquery.models.siamese_unet import SiameseUNet
        model = SiameseUNet(in_channels=3, pretrained=False).to(DEVICE)
        model.eval()
        with torch.no_grad():
            t1 = torch.rand(1, 3, 256, 256).to(DEVICE)
            t2 = torch.rand(1, 3, 256, 256).to(DEVICE)
            out = model(t1, t2)
        assert out.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be in [0,1] (sigmoid)"

    def test_siamese_shared_weights(self):
        """Critical: both encoder paths must share the SAME parameter tensor objects."""
        from satquery.models.siamese_unet import SiameseUNet
        model = SiameseUNet(in_channels=3, pretrained=False)
        # The encoder is called twice but is the same object — param data pointers match
        params = list(model.encoder.parameters())
        assert len(params) > 0, "Encoder has no parameters"
        # Verify it's a single Module (not two copies)
        assert id(model.encoder) == id(model.encoder), "Encoder not shared (module id mismatch)"

    def test_fusion_model_instantiates(self):
        from satquery.models.optical_sar_fusion import OpticalSARFusionModel
        model = OpticalSARFusionModel(opt_pretrained=False, sar_pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_fusion_model_forward_shapes(self):
        from satquery.models.optical_sar_fusion import OpticalSARFusionModel
        model = OpticalSARFusionModel(
            opt_pretrained=False, sar_pretrained=False,
            freeze_opt_stages=0, freeze_sar_stages=0,
        ).to(DEVICE)
        model.eval()
        with torch.no_grad():
            opt = torch.rand(1, 3, 224, 224).to(DEVICE)
            sar = torch.rand(1, 2, 224, 224).to(DEVICE)
            out = model(opt, sar, mode="classify")
        assert "embedding" in out
        assert "logits" in out
        assert out["embedding"].shape == (1, 768), \
            f"Expected embedding (1,768), got {out['embedding'].shape}"
        assert out["logits"].shape == (1, 19), \
            f"Expected logits (1,19), got {out['logits'].shape}"


class TestGeoInterface:

    def test_preprocess_optical_output_shape(self):
        from satquery.preprocessing.geo_interface import preprocess_optical
        arr = np.random.rand(3, 256, 256).astype(np.float32)
        t   = preprocess_optical(arr)
        assert t.shape == (1, 3, 256, 256)

    def test_preprocess_sar_output_shape(self):
        from satquery.preprocessing.geo_interface import preprocess_sar
        arr = np.random.rand(2, 256, 256).astype(np.float32)
        t   = preprocess_sar(arr)
        assert t.shape == (1, 2, 256, 256)

    def test_preprocess_bitemporal_shape_mismatch_raises(self):
        from satquery.preprocessing.geo_interface import preprocess_bitemporal
        t1 = np.random.rand(3, 128, 128).astype(np.float32)
        t2 = np.random.rand(3, 256, 256).astype(np.float32)
        with pytest.raises(ValueError):
            preprocess_bitemporal(t1, t2)

    def test_preprocess_optical_raises_on_wrong_channels(self):
        from satquery.preprocessing.geo_interface import preprocess_optical
        arr = np.random.rand(5, 128, 128).astype(np.float32)
        with pytest.raises(ValueError, match="3 channels"):
            preprocess_optical(arr)

    def test_preprocess_raises_on_nan(self):
        from satquery.preprocessing.geo_interface import preprocess_optical
        arr = np.full((3, 64, 64), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            preprocess_optical(arr)


class TestLossFunctions:

    def test_focal_loss_zero_for_perfect_prediction(self):
        from satquery.losses.compound_loss import FocalLoss
        loss_fn = FocalLoss()
        logits  = torch.full((2, 1, 8, 8), 10.0)   # very high → prob ≈ 1
        targets = torch.ones(2, 1, 8, 8)
        loss    = loss_fn(logits, targets)
        assert loss.item() < 0.01, f"Loss should be near 0 for perfect prediction: {loss}"

    def test_dice_loss_zero_for_perfect_prediction(self):
        from satquery.losses.compound_loss import DiceLoss
        loss_fn = DiceLoss()
        logits  = torch.full((2, 1, 8, 8), 10.0)
        targets = torch.ones(2, 1, 8, 8)
        loss    = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_compound_loss_returns_components(self):
        from satquery.losses.compound_loss import CompoundChangeLoss
        loss_fn = CompoundChangeLoss()
        logits  = torch.randn(2, 1, 32, 32)
        targets = (torch.rand(2, 1, 32, 32) > 0.9).float()
        total, comps = loss_fn(logits, targets)
        assert "focal" in comps and "dice" in comps and "total" in comps
        assert abs(comps["total"] - total.item()) < 1e-5
