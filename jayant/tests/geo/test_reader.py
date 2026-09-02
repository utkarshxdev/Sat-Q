"""Tests for GeoTIFF reading and metadata extraction."""

from pathlib import Path

import numpy as np
import pytest

from geo import GeoFormatError, GeoReadError, GeoValidationError, read_metadata, read_raster


def test_reader_returns_channel_first_native_dtype_and_metadata(tmp_path: Path, write_raster) -> None:
    values = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    path = write_raster(tmp_path / "image.tif", values)

    raster = read_raster(path, bands=(1, 3))

    assert raster.data.shape == (2, 4, 4)
    assert raster.data.dtype == np.float32
    assert raster.metadata.band_indexes == (1, 3)
    assert raster.metadata.crs.to_string() == "EPSG:32643"
    assert raster.metadata.shape == (4, 4)
    assert raster.valid_mask.all()
    assert read_metadata(path, bands=(2,)).count == 1


def test_reader_preserves_integer_dtype_unless_conversion_is_requested(tmp_path: Path, write_raster) -> None:
    values = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
    path = write_raster(tmp_path / "integer.tif", values, nodata=None)

    native = read_raster(path)
    converted = read_raster(path, dtype="float32")

    assert native.data.dtype == np.dtype("uint16")
    assert native.metadata.dtype == "uint16"
    assert native.metadata.source_dtype == "uint16"
    assert native.metadata.native_dtype == "uint16"
    assert converted.data.dtype == np.dtype("float32")
    assert converted.metadata.dtype == "float32"
    assert converted.metadata.native_dtype == "uint16"


def test_reader_intersects_nodata_and_nonfinite_values(tmp_path: Path, write_raster) -> None:
    values = np.ones((2, 3, 3), dtype=np.float32)
    values[:, 0, 0] = -9999.0
    values[0, 1, 1] = np.nan
    path = write_raster(tmp_path / "masked.tiff", values)

    raster = read_raster(path)

    assert not raster.valid_mask[0, 0]
    assert not raster.valid_mask[1, 1]
    assert raster.valid_mask[2, 2]


def test_reader_keeps_single_band_dimension_for_tiff(tmp_path: Path, write_raster) -> None:
    values = np.ones((1, 2, 3), dtype=np.float32)

    raster = read_raster(write_raster(tmp_path / "single.tiff", values))

    assert raster.data.shape == (1, 2, 3)


def test_metadata_extraction_preserves_source_fields(tmp_path: Path, write_raster) -> None:
    values = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
    path = write_raster(
        tmp_path / "metadata.tif",
        values,
        nodata=None,
    )

    metadata = read_metadata(path)

    assert metadata.width == 3
    assert metadata.height == 2
    assert metadata.count == 1
    assert metadata.source_count == 1
    assert metadata.source_dtype == "uint16"
    assert metadata.dtype == "uint16"
    assert metadata.resolution == (10.0, 10.0)
    assert metadata.bounds == (500000.0, 4099980.0, 500030.0, 4100000.0)
    assert metadata.nodata is None
    assert metadata.driver == "GTiff"
    assert metadata.as_dict()["dtype"] == "uint16"


def test_reader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GeoValidationError, match="does not exist"):
        read_raster(tmp_path / "missing.tif")


def test_reader_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not a GeoTIFF")

    with pytest.raises(GeoFormatError, match="Unsupported raster extension"):
        read_raster(path)


def test_reader_wraps_invalid_raster_file(tmp_path: Path) -> None:
    path = tmp_path / "invalid.tif"
    path.write_text("not a raster", encoding="utf-8")

    with pytest.raises(GeoReadError):
        read_raster(path)
