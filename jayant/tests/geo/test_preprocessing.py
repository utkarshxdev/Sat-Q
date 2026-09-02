"""Tests for mask-aware, model-agnostic preprocessing."""

import numpy as np

from geo import PreprocessingConfig, preprocess_raster, read_raster


def test_preprocessing_fills_invalid_pixels_and_preserves_metadata(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    values[0, 0, 0] = -9999.0
    raster = read_raster(write_raster(tmp_path / "image.tif", values))

    prepared = preprocess_raster(raster)

    assert prepared.data.dtype == np.float32
    assert prepared.data[0, 0, 0] == 0.0
    assert not prepared.valid_mask[0, 0]
    assert prepared.metadata == raster.metadata


def test_minmax_normalization_uses_only_valid_values(tmp_path, write_raster) -> None:
    values = np.array([[[1.0, 3.0], [5.0, -9999.0]]], dtype=np.float32)
    raster = read_raster(write_raster(tmp_path / "image.tif", values))

    prepared = preprocess_raster(raster, PreprocessingConfig(normalization="minmax"))

    np.testing.assert_allclose(prepared.data[0, :2, :], [[0.0, 0.5], [1.0, 0.0]])


def test_fixed_zscore_statistics_are_supported(tmp_path, write_raster) -> None:
    values = np.array([[[2.0, 4.0]]], dtype=np.float32)
    raster = read_raster(write_raster(tmp_path / "image.tif", values))

    prepared = preprocess_raster(
        raster,
        PreprocessingConfig(normalization="zscore", means=(2.0,), stds=(2.0,)),
    )

    np.testing.assert_allclose(prepared.data, [[[0.0, 1.0]]])
