"""Tests for public input-validation helpers."""

import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from geo import (
    GeoValidationError,
    RasterValidationConfig,
    normalize_band_indexes,
    read_raster,
    validate_array,
    validate_bitemporal_pair,
    validate_optical_sar_pair,
    validate_raster,
)


def test_validate_array_accepts_channel_first_numeric_data() -> None:
    array = validate_array(np.zeros((3, 8, 8), dtype=np.float32))
    assert array.shape == (3, 8, 8)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((8, 8), dtype=np.float32), "shape"),
        (np.zeros((3, 8, 8), dtype=np.float32) * np.nan, "finite"),
        (np.zeros((3, 8, 8), dtype=object), "numeric"),
    ],
)
def test_validate_array_rejects_invalid_inputs(value: np.ndarray, message: str) -> None:
    with pytest.raises(GeoValidationError, match=message):
        validate_array(value)


def test_validate_array_can_defer_finite_check_for_raw_raster_data() -> None:
    result = validate_array(np.full((1, 2, 2), np.nan), require_finite=False)
    assert result.shape == (1, 2, 2)


def test_band_indexes_are_one_based_and_in_range() -> None:
    assert normalize_band_indexes(None, 2) == (1, 2)
    assert normalize_band_indexes((2, 1), 2) == (2, 1)
    with pytest.raises(GeoValidationError, match="between 1 and 2"):
        normalize_band_indexes((3,), 2)


def test_validate_raster_returns_valid_structured_result(tmp_path, write_raster) -> None:
    path = write_raster(tmp_path / "valid.tif", np.ones((3, 4, 4), dtype=np.float32))

    result = validate_raster(path)

    assert result.valid
    assert result.errors == ()
    assert result.metadata is not None
    assert result.metadata.count == 3


def test_validate_raster_reports_missing_crs(tmp_path, write_raster) -> None:
    path = write_raster(tmp_path / "missing-crs.tif", np.ones((1, 3, 3), dtype=np.float32), crs=None)

    result = validate_raster(path)

    assert not result.valid
    assert any(issue.code == "missing_crs" for issue in result.errors)


def test_validate_raster_applies_explicit_band_count_constraint(tmp_path, write_raster) -> None:
    path = write_raster(tmp_path / "bands.tif", np.ones((2, 3, 3), dtype=np.float32))

    result = validate_raster(path, RasterValidationConfig(exact_band_count=3))

    assert not result.valid
    assert any(issue.code == "invalid_band_count" for issue in result.errors)


def test_validate_raster_reports_invalid_numeric_values(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    values[0, 0, 0] = np.nan
    path = write_raster(tmp_path / "nan.tif", values, nodata=None)

    result = validate_raster(path)

    assert not result.valid
    assert any(issue.code == "invalid_numeric_values" for issue in result.errors)


def test_optical_sar_validation_reports_mismatched_crs(tmp_path, write_raster) -> None:
    values = np.ones((1, 3, 3), dtype=np.float32)
    transform = from_origin(0, 30, 10, 10)
    optical = write_raster(tmp_path / "optical.tif", values, transform=transform, crs="EPSG:32643")
    sar = write_raster(tmp_path / "sar.tif", values, transform=transform, crs="EPSG:4326")

    result = validate_optical_sar_pair(optical, sar)

    assert not result.valid
    assert any(issue.code == "mismatched_crs" for issue in result.errors)


def test_optical_sar_validation_reports_different_resolutions(tmp_path, write_raster) -> None:
    values = np.ones((1, 3, 3), dtype=np.float32)
    optical = write_raster(tmp_path / "optical.tif", values, transform=from_origin(0, 30, 10, 10))
    sar = write_raster(tmp_path / "sar.tif", values, transform=from_origin(0, 60, 20, 20))

    result = validate_optical_sar_pair(optical, sar)

    assert not result.valid
    assert any(issue.code == "different_resolution" for issue in result.errors)


def test_bitemporal_validation_reports_different_dimensions(tmp_path, write_raster) -> None:
    first = write_raster(tmp_path / "t1.tif", np.ones((2, 3, 3), dtype=np.float32))
    second = write_raster(tmp_path / "t2.tif", np.ones((2, 2, 3), dtype=np.float32))

    result = validate_bitemporal_pair(first, second)

    assert not result.valid
    assert any(issue.code == "different_dimensions" for issue in result.errors)


def test_pair_validation_reports_non_overlapping_rasters(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    first = write_raster(tmp_path / "first.tif", values, transform=from_origin(0, 20, 10, 10))
    second = write_raster(tmp_path / "second.tif", values, transform=from_origin(100, 20, 10, 10))

    result = validate_bitemporal_pair(first, second)

    assert not result.valid
    assert result.spatial_overlap is False
    assert any(issue.code == "non_overlapping_rasters" for issue in result.errors)


def test_pair_validation_reports_transformability_without_correcting(tmp_path, write_raster) -> None:
    first = write_raster(
        tmp_path / "first.tif",
        np.ones((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )
    second = write_raster(
        tmp_path / "second.tif",
        np.ones((1, 2, 2), dtype=np.float32),
        transform=from_origin(0, 40, 20, 20),
    )

    result = validate_bitemporal_pair(first, second)

    assert not result.valid
    assert result.can_transform_to_common_grid is True
    assert result.second.metadata is not None
    assert result.second.metadata.shape == (2, 2)


def test_optical_sar_validator_rejects_equal_size_resolution_shifted_grid(tmp_path, write_raster) -> None:
    values = np.ones((1, 4, 4), dtype=np.float32)
    optical = write_raster(tmp_path / "optical.tif", values, transform=from_origin(0, 40, 10, 10))
    sar = write_raster(tmp_path / "sar.tif", values, transform=from_origin(5, 40, 10, 10))

    result = validate_optical_sar_pair(optical, sar)

    assert not result.valid
    assert result.spatial_overlap is True
    assert result.exact_common_grid is False
    assert "affine transform" in result.grid_mismatches
    assert "bounds" in result.grid_mismatches
    assert any(issue.code == "different_transform" for issue in result.errors)


def test_bitemporal_validator_rejects_equal_size_resolution_shifted_grid(tmp_path, write_raster) -> None:
    values = np.ones((2, 4, 4), dtype=np.float32)
    first = write_raster(tmp_path / "t1.tif", values, transform=from_origin(0, 40, 10, 10))
    second = write_raster(tmp_path / "t2.tif", values, transform=from_origin(5, 40, 10, 10))

    result = validate_bitemporal_pair(first, second)

    assert not result.valid
    assert result.spatial_overlap is True
    assert result.exact_common_grid is False
    assert any(issue.code == "different_transform" for issue in result.errors)


@pytest.mark.parametrize(
    "transform",
    [
        Affine.rotation(5) * from_origin(0, 40, 10, 10),
        Affine(10, 0.5, 0, 0, -10, 40),
    ],
    ids=["rotation", "shear"],
)
def test_pair_validator_reports_rotation_or_shear_misalignment(tmp_path, write_raster, transform: Affine) -> None:
    values = np.ones((1, 4, 4), dtype=np.float32)
    first = write_raster(tmp_path / "first.tif", values, transform=from_origin(0, 40, 10, 10))
    second = write_raster(tmp_path / "second.tif", values, transform=transform)

    result = validate_optical_sar_pair(first, second)

    assert not result.valid
    assert result.exact_common_grid is False
    assert any(issue.code == "different_transform" for issue in result.errors)


def test_pair_validator_distinguishes_partial_overlap_from_common_grid(tmp_path, write_raster) -> None:
    values = np.ones((1, 4, 4), dtype=np.float32)
    first = write_raster(tmp_path / "first.tif", values, transform=from_origin(0, 40, 10, 10))
    second = write_raster(tmp_path / "second.tif", values, transform=from_origin(35, 40, 10, 10))

    result = validate_bitemporal_pair(first, second)

    assert result.spatial_overlap is True
    assert result.exact_common_grid is False
    assert any(issue.code == "overlap_not_common_grid" for issue in result.warnings)


@pytest.mark.parametrize(
    ("first_crs", "second_crs"),
    [(None, "EPSG:32643"), (None, None)],
    ids=["one-missing", "both-missing"],
)
def test_optical_sar_validator_handles_missing_crs_safely(
    tmp_path,
    write_raster,
    first_crs: str | None,
    second_crs: str | None,
) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    transform = from_origin(0, 20, 10, 10)
    first = write_raster(tmp_path / "first.tif", values, transform=transform, crs=first_crs)
    second = write_raster(tmp_path / "second.tif", values, transform=transform, crs=second_crs)

    result = validate_optical_sar_pair(first, second)

    assert not result.valid
    assert result.spatial_overlap is None
    assert result.can_transform_to_common_grid is False
    assert any(issue.code == "missing_crs" for issue in result.errors)
    assert any(issue.code == "cannot_transform_to_common_grid" for issue in result.errors)


def test_bitemporal_validator_handles_missing_crs_safely(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    transform = from_origin(0, 20, 10, 10)
    first = write_raster(tmp_path / "first.tif", values, transform=transform, crs="EPSG:32643")
    second = write_raster(tmp_path / "second.tif", values, transform=transform, crs=None)

    result = validate_bitemporal_pair(first, second)

    assert not result.valid
    assert result.spatial_overlap is None
    assert result.can_transform_to_common_grid is False
    assert any(issue.code == "missing_crs" for issue in result.errors)


def test_pair_tolerance_is_configurable_and_not_hardcoded(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    first = write_raster(tmp_path / "first.tif", values, transform=from_origin(0, 20, 10, 10))
    second = write_raster(tmp_path / "second.tif", values, transform=from_origin(1e-6, 20, 10, 10))
    strict = RasterValidationConfig(grid_absolute_tolerance=0.0, grid_relative_tolerance=0.0)
    permissive = RasterValidationConfig(grid_absolute_tolerance=1e-5, grid_relative_tolerance=0.0)

    strict_result = validate_optical_sar_pair(first, second, strict, strict)
    permissive_result = validate_optical_sar_pair(first, second, permissive, permissive)

    assert strict_result.exact_common_grid is False
    assert permissive_result.exact_common_grid is True
    assert permissive_result.valid


def test_nan_nodata_is_allowed_by_default_and_policy_is_explicit(tmp_path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    path = write_raster(tmp_path / "nan-nodata.tif", values, nodata=float("nan"))

    default_result = validate_raster(path)
    numeric_result = validate_raster(path, RasterValidationConfig(nodata_policy="numeric"))

    assert default_result.valid
    assert default_result.nodata_kind == "nan"
    assert not numeric_result.valid
    assert any(issue.code == "nodata_policy_mismatch" for issue in numeric_result.errors)


def test_bitemporal_dtype_and_nodata_differences_are_diagnostics(tmp_path, write_raster) -> None:
    transform = from_origin(0, 20, 10, 10)
    first = write_raster(
        tmp_path / "first.tif",
        np.ones((1, 2, 2), dtype=np.float32),
        transform=transform,
        nodata=-9999.0,
    )
    second = write_raster(
        tmp_path / "second.tif",
        np.ones((1, 2, 2), dtype=np.uint16),
        transform=transform,
        nodata=0.0,
    )

    result = validate_bitemporal_pair(first, second)

    assert result.valid
    assert result.dtype_match is False
    assert result.nodata_match is False
    assert any(issue.code == "different_dtype" for issue in result.warnings)
    assert any(issue.code == "different_nodata" for issue in result.warnings)


def test_pair_validation_does_not_mutate_loaded_rasters(tmp_path, write_raster) -> None:
    values = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    first = read_raster(write_raster(tmp_path / "first.tif", values))
    second = read_raster(write_raster(tmp_path / "second.tif", values, transform=from_origin(1, 20, 10, 10)))
    first_data = first.data.copy()
    second_data = second.data.copy()
    first_mask = first.valid_mask.copy()
    second_mask = second.valid_mask.copy()

    validate_bitemporal_pair(first, second)

    np.testing.assert_array_equal(first.data, first_data)
    np.testing.assert_array_equal(second.data, second_data)
    np.testing.assert_array_equal(first.valid_mask, first_mask)
    np.testing.assert_array_equal(second.valid_mask, second_mask)
