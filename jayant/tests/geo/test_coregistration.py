"""Tests for strict grid checks and explicit reprojection."""

from pathlib import Path
from dataclasses import replace
import importlib

import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform

from geo import (
    AlignmentConfig,
    GeoAlignmentError,
    GeoRaster,
    align_raster_to_reference,
    read_raster,
    validate_coregistration,
)

coregistration_module = importlib.import_module("geo.coregistration")


def test_coregistration_accepts_equal_grids(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 4, 4), dtype=np.float32)
    transform = from_origin(500000, 4100000, 10, 10)
    first = read_raster(write_raster(tmp_path / "first.tif", values, transform=transform))
    second = read_raster(write_raster(tmp_path / "second.tif", values, transform=transform))

    validate_coregistration(first.metadata, second.metadata)


def test_coregistration_rejects_shifted_grid(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 4, 4), dtype=np.float32)
    first = read_raster(write_raster(tmp_path / "first.tif", values, transform=from_origin(0, 40, 10, 10)))
    second = read_raster(write_raster(tmp_path / "second.tif", values, transform=from_origin(10, 40, 10, 10)))

    with pytest.raises(GeoAlignmentError, match="transform|bounds"):
        validate_coregistration(first.metadata, second.metadata)


def test_alignment_requires_explicit_config_and_targets_reference_grid(tmp_path: Path, write_raster) -> None:
    first_values = np.zeros((1, 4, 4), dtype=np.float32)
    second_values = np.ones((1, 2, 2), dtype=np.float32)
    first = read_raster(write_raster(tmp_path / "first.tif", first_values, transform=from_origin(0, 40, 10, 10)))
    second = read_raster(write_raster(tmp_path / "second.tif", second_values, transform=from_origin(0, 40, 20, 20)))

    with pytest.raises(GeoAlignmentError):
        validate_coregistration(first.metadata, second.metadata)

    aligned = align_raster_to_reference(second, first, AlignmentConfig())
    assert aligned.data.shape == first.data.shape
    assert aligned.metadata.transform == first.metadata.transform
    assert aligned.metadata.crs == first.metadata.crs
    assert aligned.valid_mask.shape == (4, 4)


def test_same_grid_uses_noop_fast_path_and_structured_output(tmp_path: Path, write_raster) -> None:
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    transform = from_origin(0, 40, 10, 10)
    source_path = write_raster(tmp_path / "source.tif", values, transform=transform)
    reference_path = write_raster(tmp_path / "reference.tif", np.zeros_like(values), transform=transform)

    result = align_raster_to_reference(source_path, reference_path)

    np.testing.assert_array_equal(result.aligned_array, values)
    assert result.aligned_array is not values
    assert result.reference_metadata.transform == read_raster(reference_path).metadata.transform
    assert result.source_metadata.path == source_path.resolve()
    assert result.alignment_diagnostics.reprojected is False
    assert result.alignment_diagnostics.resampled is False
    assert result.alignment_diagnostics.grid_comparison["exact_common_grid"] is True


def test_output_metadata_matches_aligned_array_and_reference_grid(tmp_path: Path, write_raster) -> None:
    source_values = np.stack(
        [np.arange(4, dtype=np.uint16).reshape(2, 2), np.full((2, 2), 8, dtype=np.uint16)],
    )
    source_path = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=from_origin(0, 20, 10, 10),
        nodata=None,
    )
    reference_path = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 20, 5, 5),
    )

    result = align_raster_to_reference(
        source_path,
        reference_path,
        AlignmentConfig(destination_nodata=99.0),
    )
    reference = read_raster(reference_path)

    assert result.output_metadata.crs == reference.metadata.crs
    assert result.output_metadata.transform == reference.metadata.transform
    assert result.output_metadata.width == reference.metadata.width
    assert result.output_metadata.height == reference.metadata.height
    assert result.output_metadata.bounds == reference.metadata.bounds
    assert result.output_metadata.count == result.aligned_array.shape[0] == 2
    assert result.output_metadata.dtype == np.dtype(result.aligned_array.dtype).name == "uint16"
    assert result.output_metadata.nodata == 99
    assert result.reference_metadata.count == 1
    assert result.source_metadata.count == 2


def test_post_alignment_verification_rejects_bad_generated_metadata(
    tmp_path: Path,
    write_raster,
    monkeypatch,
) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    source = write_raster(tmp_path / "source.tif", values)
    reference = write_raster(tmp_path / "reference.tif", values)
    original_builder = coregistration_module._build_output_metadata

    def bad_builder(source_raster, reference_raster, aligned_array, destination_nodata):
        metadata = original_builder(source_raster, reference_raster, aligned_array, destination_nodata)
        return replace(metadata, transform=metadata.transform * Affine.translation(1, 0))

    monkeypatch.setattr(coregistration_module, "_build_output_metadata", bad_builder)

    with pytest.raises(GeoAlignmentError, match="output metadata does not match the reference grid"):
        align_raster_to_reference(source, reference)


def test_invalid_transformed_bounds_are_rejected(tmp_path: Path, write_raster, monkeypatch) -> None:
    source = write_raster(
        tmp_path / "source.tif",
        np.ones((1, 2, 2), dtype=np.float32),
        crs="EPSG:4326",
        transform=from_origin(-1, 1, 1, 1),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.ones((1, 2, 2), dtype=np.float32),
        crs="EPSG:3857",
        transform=from_origin(-100000, 100000, 100000, 100000),
    )
    monkeypatch.setattr(
        coregistration_module,
        "transform_bounds",
        lambda *args, **kwargs: (0.0, 0.0, float("nan"), 1.0),
    )

    with pytest.raises(GeoAlignmentError, match="Transformed source bounds are invalid"):
        align_raster_to_reference(source, reference)


def test_shifted_same_size_grid_performs_spatial_transformation(tmp_path: Path, write_raster) -> None:
    reference_values = np.zeros((1, 4, 4), dtype=np.float32)
    source_values = np.tile(np.array([10, 20, 30, 40], dtype=np.float32), (4, 1))[None, :, :]
    reference = write_raster(
        tmp_path / "reference.tif",
        reference_values,
        transform=from_origin(0, 40, 10, 10),
    )
    source = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=from_origin(10, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference, AlignmentConfig(resampling=Resampling.nearest))

    expected = np.tile(np.array([0, 10, 20, 30], dtype=np.float32), (4, 1))
    np.testing.assert_array_equal(result.aligned_array[0], expected)
    assert result.alignment_diagnostics.reprojected is True
    assert result.alignment_diagnostics.resampled is True
    assert result.alignment_diagnostics.grid_comparison["crs"] is True
    assert result.alignment_diagnostics.grid_comparison["dimensions"] is True
    assert result.alignment_diagnostics.grid_comparison["resolution"] is True
    assert result.alignment_diagnostics.grid_comparison["affine_transform"] is False


def test_different_resolution_is_resampled_to_reference_grid(tmp_path: Path, write_raster) -> None:
    source_values = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    source = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=from_origin(0, 40, 20, 20),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference)

    expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.float32)
    np.testing.assert_array_equal(result.aligned_array[0], expected)
    assert result.reference_metadata.resolution == (10.0, 10.0)


def test_different_crs_is_reprojected_to_reference_crs(tmp_path: Path, write_raster) -> None:
    source_values = np.full((1, 4, 4), 7.0, dtype=np.float32)
    source_transform = from_origin(-1, 1, 0.5, 0.5)
    source = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=source_transform,
        crs="EPSG:4326",
    )
    reference_transform, _, _ = calculate_default_transform(
        "EPSG:4326",
        "EPSG:3857",
        4,
        4,
        -1,
        -1,
        1,
        1,
        dst_width=4,
        dst_height=4,
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=reference_transform,
        crs="EPSG:3857",
    )

    result = align_raster_to_reference(source, reference)

    assert result.aligned_array.shape == (1, 4, 4)
    assert result.reference_metadata.crs.to_string() == "EPSG:3857"
    assert result.source_metadata.crs.to_string() == "EPSG:4326"
    assert result.alignment_diagnostics.reprojected is True
    assert np.all(result.aligned_valid_mask)
    np.testing.assert_allclose(result.aligned_array, 7.0)


def test_different_dimensions_produce_reference_dimensions(tmp_path: Path, write_raster) -> None:
    source = write_raster(
        tmp_path / "source.tif",
        np.ones((1, 2, 3), dtype=np.float32),
        transform=from_origin(0, 30, 10, 10),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference)

    assert result.aligned_array.shape == (1, 4, 4)
    assert result.alignment_diagnostics.source_dimensions == (2, 3)
    assert result.alignment_diagnostics.reference_dimensions == (4, 4)


@pytest.mark.parametrize(
    "transform",
    [
        Affine.rotation(2) * from_origin(0, 40, 10, 10),
        Affine(10, 0.5, 0, 0, -10, 40),
    ],
    ids=["rotation", "shear"],
)
def test_rotation_and_shear_are_geospatially_transformed(tmp_path: Path, write_raster, transform: Affine) -> None:
    source = write_raster(
        tmp_path / "source.tif",
        np.ones((1, 4, 4), dtype=np.float32),
        transform=transform,
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference)

    assert result.aligned_array.shape == (1, 4, 4)
    assert result.alignment_diagnostics.reprojected is True
    assert result.alignment_diagnostics.grid_comparison["affine_transform"] is False


def test_partial_overlap_fills_reference_pixels_outside_source_coverage(tmp_path: Path, write_raster) -> None:
    source = write_raster(
        tmp_path / "source.tif",
        np.full((1, 4, 4), 5.0, dtype=np.float32),
        transform=from_origin(20, 40, 10, 10),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference, AlignmentConfig(destination_nodata=99.0))

    np.testing.assert_array_equal(result.aligned_array[0], np.array([[99, 99, 5, 5]] * 4, dtype=np.float32))
    np.testing.assert_array_equal(result.aligned_valid_mask, np.array([[False, False, True, True]] * 4))
    assert result.alignment_diagnostics.spatial_overlap is True


def test_non_overlapping_rasters_are_rejected_before_reprojection(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    source = write_raster(tmp_path / "source.tif", values, transform=from_origin(100, 20, 10, 10))
    reference = write_raster(tmp_path / "reference.tif", values, transform=from_origin(0, 20, 10, 10))

    with pytest.raises(GeoAlignmentError, match="do not overlap spatially"):
        align_raster_to_reference(source, reference)


def test_multiband_source_preserves_band_count(tmp_path: Path, write_raster) -> None:
    source_values = np.stack(
        [np.full((2, 2), fill_value=band, dtype=np.float32) for band in (1, 2, 3)],
    )
    source = write_raster(tmp_path / "source.tif", source_values, transform=from_origin(0, 20, 10, 10))
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 20, 5, 5),
    )

    result = align_raster_to_reference(source, reference)

    assert result.aligned_array.shape == (3, 4, 4)
    assert result.source_metadata.count == 3


def test_missing_crs_fails_without_guessing(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    source = write_raster(tmp_path / "source.tif", values, crs=None)
    reference = write_raster(tmp_path / "reference.tif", values)

    with pytest.raises(GeoAlignmentError, match="no CRS"):
        align_raster_to_reference(source, reference)


def test_identity_transform_is_rejected_as_missing_georeferencing(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 2, 2), dtype=np.float32)
    source_raster = read_raster(write_raster(tmp_path / "source.tif", values))
    reference = write_raster(tmp_path / "reference.tif", values)
    invalid_source = GeoRaster(
        data=source_raster.data,
        metadata=replace(source_raster.metadata, transform=Affine.identity()),
        valid_mask=source_raster.valid_mask,
    )

    with pytest.raises(GeoAlignmentError, match="transform"):
        align_raster_to_reference(invalid_source, reference)


def test_source_nodata_and_destination_nodata_are_deterministic(tmp_path: Path, write_raster) -> None:
    source_values = np.full((1, 4, 4), 5.0, dtype=np.float32)
    source_values[0, :, 0] = -9999.0
    source = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=from_origin(10, 40, 10, 10),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 40, 10, 10),
    )

    result = align_raster_to_reference(source, reference, AlignmentConfig(destination_nodata=77.0))

    assert result.aligned_array[0, 0, 0] == 77.0
    assert result.aligned_array[0, 0, 1] == 77.0
    assert result.aligned_valid_mask[0, 0] is np.False_
    assert result.aligned_valid_mask[0, 1] is np.False_
    assert result.aligned_valid_mask[0, 2]
    assert result.alignment_diagnostics.nodata_policy == "respect_source"


def test_direct_georaster_nodata_excludes_pixels_even_when_mask_says_valid(tmp_path: Path, write_raster) -> None:
    values = np.array([[[5.0, -9999.0], [7.0, 8.0]]], dtype=np.float32)
    loaded = read_raster(
        write_raster(
            tmp_path / "source.tif",
            values,
            transform=from_origin(0, 20, 10, 10),
        ),
    )
    direct = GeoRaster(
        data=values,
        metadata=loaded.metadata,
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros_like(values),
        transform=loaded.metadata.transform,
    )

    result = align_raster_to_reference(direct, reference, AlignmentConfig(destination_nodata=77.0))

    assert not result.aligned_valid_mask[0, 1]
    assert result.aligned_array[0, 0, 1] == 77.0


def test_direct_georaster_false_mask_remains_invalid(tmp_path: Path, write_raster) -> None:
    values = np.full((1, 2, 2), 5.0, dtype=np.float32)
    loaded = read_raster(write_raster(tmp_path / "source.tif", values))
    direct = GeoRaster(
        data=values,
        metadata=loaded.metadata,
        valid_mask=np.array([[True, True], [True, False]]),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros_like(values),
        transform=loaded.metadata.transform,
    )

    result = align_raster_to_reference(direct, reference, AlignmentConfig(destination_nodata=77.0))

    assert not result.aligned_valid_mask[1, 1]
    assert result.aligned_array[0, 1, 1] == 77.0


def test_direct_georaster_without_nodata_keeps_zero_as_valid(tmp_path: Path, write_raster) -> None:
    values = np.array([[[0.0, 5.0], [7.0, 8.0]]], dtype=np.float32)
    loaded = read_raster(
        write_raster(
            tmp_path / "source.tif",
            values,
            transform=from_origin(0, 20, 10, 10),
            nodata=None,
        ),
    )
    direct = GeoRaster(data=values, metadata=loaded.metadata, valid_mask=np.ones((2, 2), dtype=bool))
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros_like(values),
        transform=loaded.metadata.transform,
    )

    result = align_raster_to_reference(direct, reference, AlignmentConfig(destination_nodata=77.0))

    assert result.aligned_valid_mask[0, 0]
    assert result.aligned_array[0, 0, 0] == 0.0


def test_direct_georaster_nan_nodata_is_excluded_from_effective_mask(tmp_path: Path, write_raster) -> None:
    values = np.array([[[np.nan, 5.0], [7.0, 8.0]]], dtype=np.float32)
    loaded = read_raster(
        write_raster(
            tmp_path / "source.tif",
            values,
            transform=from_origin(0, 20, 10, 10),
            nodata=float("nan"),
        ),
    )
    direct = GeoRaster(data=values, metadata=loaded.metadata, valid_mask=np.ones((2, 2), dtype=bool))
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros_like(values),
        transform=loaded.metadata.transform,
    )

    result = align_raster_to_reference(direct, reference, AlignmentConfig(destination_nodata=77.0))

    assert not result.aligned_valid_mask[0, 0]
    assert result.aligned_array[0, 0, 0] == 77.0


def test_declared_nan_nodata_is_respected_during_alignment(tmp_path: Path, write_raster) -> None:
    source_values = np.full((1, 2, 2), 5.0, dtype=np.float32)
    source_values[0, 0, 0] = np.nan
    source = write_raster(
        tmp_path / "source.tif",
        source_values,
        transform=from_origin(10, 20, 10, 10),
        nodata=float("nan"),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 2, 3), dtype=np.float32),
        transform=from_origin(0, 20, 10, 10),
    )

    result = align_raster_to_reference(source, reference, AlignmentConfig(destination_nodata=77.0))

    assert result.aligned_array[0, 0, 0] == 77.0
    assert result.aligned_array[0, 0, 1] == 77.0
    assert not result.aligned_valid_mask[0, 0]
    assert not result.aligned_valid_mask[0, 1]
    assert result.aligned_valid_mask[0, 2]


def test_alignment_preserves_non_float_source_dtype(tmp_path: Path, write_raster) -> None:
    values = np.arange(4, dtype=np.uint16).reshape(1, 2, 2)
    path = write_raster(
        tmp_path / "source.tif",
        values,
        transform=from_origin(0, 20, 10, 10),
        nodata=None,
    )
    loaded = read_raster(path)
    integer_source = GeoRaster(
        data=values,
        metadata=loaded.metadata,
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    reference = write_raster(
        tmp_path / "reference.tif",
        np.zeros((1, 4, 4), dtype=np.float32),
        transform=from_origin(0, 20, 5, 5),
    )

    result = align_raster_to_reference(integer_source, reference)

    assert result.aligned_array.dtype == np.dtype("uint16")
    assert result.alignment_diagnostics.aligned_dtype == "uint16"
