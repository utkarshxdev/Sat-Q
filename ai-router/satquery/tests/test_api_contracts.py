"""
satquery/tests/test_api_contracts.py
──────────────────────────────────────
Integration tests for the two primary API functions.

Tests verify:
  1. Output key presence and completeness.
  2. Output dtypes and value ranges.
  3. Shape invariants under various input sizes.
  4. Empty-mask defense (noise_floor threshold).
  5. agent_registry dispatch_tool correctness.

Run with:
    pytest satquery/tests/test_api_contracts.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from satquery.inference.api import get_change_analysis, extract_fused_features
from satquery.inference.agent_registry import dispatch_tool, get_tool_schemas


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def optical_image() -> np.ndarray:
    """Random float32 (3, 224, 224) optical image in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((3, 224, 224)).astype(np.float32)


@pytest.fixture
def sar_image() -> np.ndarray:
    """Random float32 (2, 224, 224) SAR image in [0, 1]."""
    rng = np.random.default_rng(43)
    return rng.random((2, 224, 224)).astype(np.float32)


@pytest.fixture
def temporal_pair_identical() -> tuple[np.ndarray, np.ndarray]:
    """Two identical images (should produce ~0% change)."""
    img = np.random.default_rng(0).random((3, 128, 128)).astype(np.float32)
    return img, img.copy()


@pytest.fixture
def temporal_pair_different() -> tuple[np.ndarray, np.ndarray]:
    """Two very different images (should produce high change %)."""
    rng = np.random.default_rng(1)
    t1 = rng.random((3, 128, 128)).astype(np.float32)
    t2 = rng.random((3, 128, 128)).astype(np.float32)
    # Force large difference by flipping pixel distribution
    t2 = 1.0 - t2
    return t1, t2


# ─── get_change_analysis tests ───────────────────────────────────────────────

class TestChangeAnalysis:

    def test_returns_all_required_keys(self, temporal_pair_identical):
        t1, t2 = temporal_pair_identical
        result = get_change_analysis(t1, t2)
        required_keys = {
            "change_mask", "change_detected", "confidence",
            "changed_area_pct", "summary", "model_used", "checkpoint_loaded",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_change_mask_dtype_and_values(self, temporal_pair_identical):
        t1, t2 = temporal_pair_identical
        result = get_change_analysis(t1, t2)
        mask   = result["change_mask"]
        assert isinstance(mask, np.ndarray), "change_mask must be np.ndarray"
        assert mask.dtype == np.uint8, f"change_mask dtype must be uint8, got {mask.dtype}"
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), (
            f"change_mask must only contain {{0, 255}}, got {unique_vals}"
        )

    def test_change_mask_spatial_dims_match_input(self, temporal_pair_different):
        t1, t2 = temporal_pair_different
        result = get_change_analysis(t1, t2)
        h, w   = t1.shape[1], t1.shape[2]
        assert result["change_mask"].shape == (h, w), (
            f"Expected mask shape ({h},{w}), got {result['change_mask'].shape}"
        )

    def test_confidence_in_range(self, temporal_pair_different):
        t1, t2 = temporal_pair_different
        result = get_change_analysis(t1, t2)
        conf   = result["confidence"]
        assert 0.0 <= conf <= 1.0, f"confidence must be in [0,1], got {conf}"

    def test_changed_area_pct_is_float(self, temporal_pair_different):
        t1, t2 = temporal_pair_different
        result  = get_change_analysis(t1, t2)
        pct     = result["changed_area_pct"]
        assert isinstance(pct, (int, float)), "changed_area_pct must be numeric"
        assert 0.0 <= pct <= 100.0, f"changed_area_pct out of range: {pct}"

    def test_change_detected_is_bool(self, temporal_pair_different):
        t1, t2 = temporal_pair_different
        result  = get_change_analysis(t1, t2)
        assert isinstance(result["change_detected"], bool)

    def test_empty_mask_defense_identical_images(self, temporal_pair_identical):
        """Identical images with threshold=0.5 should have very low change area.
        With an untrained model the output is random, so we test the noise floor
        defense itself by setting a high noise_floor."""
        t1, t2 = temporal_pair_identical
        # Force noise floor to 100% to guarantee defense triggers
        result = get_change_analysis(t1, t2, noise_floor=1.01)
        assert result["change_detected"] is False
        assert result["confidence"] == 0.0
        assert np.all(result["change_mask"] == 0), "Mask must be all zeros after defense"

    def test_summary_is_string(self, temporal_pair_identical):
        t1, t2 = temporal_pair_identical
        result  = get_change_analysis(t1, t2)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_various_input_sizes(self):
        """Verify pipeline handles non-standard spatial dims."""
        for size in [(3, 64, 64), (3, 512, 512)]:
            t1 = np.random.rand(*size).astype(np.float32)
            t2 = np.random.rand(*size).astype(np.float32)
            result = get_change_analysis(t1, t2)
            h, w = size[1], size[2]
            assert result["change_mask"].shape == (h, w)

    def test_raises_on_mismatched_shapes(self):
        t1 = np.random.rand(3, 128, 128).astype(np.float32)
        t2 = np.random.rand(3, 256, 256).astype(np.float32)
        with pytest.raises(ValueError, match="must match"):
            get_change_analysis(t1, t2)

    def test_raises_on_nan_input(self):
        t1 = np.full((3, 64, 64), np.nan, dtype=np.float32)
        t2 = np.random.rand(3, 64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="NaN"):
            get_change_analysis(t1, t2)


# ─── extract_fused_features tests ───────────────────────────────────────────

class TestFusedFeatures:

    def test_returns_all_required_keys(self, optical_image, sar_image):
        result = extract_fused_features(optical_image, sar_image)
        required_keys = {
            "fused_embedding", "land_cover_logits", "land_cover_probs",
            "attention_map", "status", "model_used", "checkpoint_loaded",
        }
        assert required_keys.issubset(result.keys())

    def test_fused_embedding_shape(self, optical_image, sar_image):
        result    = extract_fused_features(optical_image, sar_image)
        embedding = result["fused_embedding"]
        assert isinstance(embedding, np.ndarray), "fused_embedding must be np.ndarray"
        assert embedding.ndim == 1, f"fused_embedding must be 1D, got {embedding.ndim}D"
        assert embedding.dtype == np.float32

    def test_land_cover_logits_shape(self, optical_image, sar_image):
        result = extract_fused_features(optical_image, sar_image)
        logits  = result["land_cover_logits"]
        assert logits.shape == (19,), f"Expected (19,), got {logits.shape}"
        assert logits.dtype == np.float32

    def test_land_cover_probs_range(self, optical_image, sar_image):
        result = extract_fused_features(optical_image, sar_image)
        probs  = result["land_cover_probs"]
        assert probs.shape == (19,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), (
            "land_cover_probs must be in [0,1]"
        )

    def test_attention_map_is_2d(self, optical_image, sar_image):
        result   = extract_fused_features(optical_image, sar_image)
        attn_map = result["attention_map"]
        assert isinstance(attn_map, np.ndarray)
        assert attn_map.ndim == 2, f"attention_map must be 2D, got {attn_map.ndim}D"

    def test_status_is_valid_string(self, optical_image, sar_image):
        result = extract_fused_features(optical_image, sar_image)
        assert result["status"] in ("ok", "mock")

    def test_raises_on_wrong_optical_channels(self, sar_image):
        bad_optical = np.random.rand(5, 224, 224).astype(np.float32)  # wrong channels
        with pytest.raises(ValueError, match="3 channels"):
            extract_fused_features(bad_optical, sar_image)

    def test_raises_on_wrong_sar_channels(self, optical_image):
        bad_sar = np.random.rand(3, 224, 224).astype(np.float32)  # should be 2-ch
        with pytest.raises(ValueError, match="2 channels"):
            extract_fused_features(optical_image, bad_sar)


# ─── Agent Registry tests ─────────────────────────────────────────────────────

class TestAgentRegistry:

    def test_get_tool_schemas_returns_list(self):
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 2

    def test_schema_has_required_openai_fields(self):
        schemas = get_tool_schemas()
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert "required" in schema["parameters"]

    def test_dispatch_unknown_tool_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown tool"):
            dispatch_tool("nonexistent_tool", {})

    def test_dispatch_missing_arg_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required"):
            dispatch_tool("get_change_analysis", {"img_t1": [[0]]})  # missing img_t2

    def test_dispatch_change_analysis_returns_serialisable(self):
        """dispatch_tool must return a JSON-serialisable dict (no np.ndarray)."""
        import json
        t1 = np.random.rand(3, 64, 64).astype(np.float32).tolist()
        t2 = np.random.rand(3, 64, 64).astype(np.float32).tolist()
        result = dispatch_tool("get_change_analysis", {"img_t1": t1, "img_t2": t2})
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"dispatch_tool result is not JSON-serialisable: {e}")

    def test_dispatch_fused_features_returns_serialisable(self):
        import json
        opt = np.random.rand(3, 224, 224).astype(np.float32).tolist()
        sar = np.random.rand(2, 224, 224).astype(np.float32).tolist()
        result = dispatch_tool("extract_fused_features", {"optical": opt, "sar": sar})
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"dispatch_tool result is not JSON-serialisable: {e}")
