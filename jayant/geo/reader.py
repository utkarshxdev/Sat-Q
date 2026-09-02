"""GeoTIFF/TIFF reading into model-friendly channel-first NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio

from .exceptions import GeoReadError, GeoValidationError
from .metadata import _metadata_from_dataset
from .types import GeoRaster
from .validation import _nodata_value_mask, normalize_band_indexes, validate_raster_path


def read_raster(
    path: str | Path,
    bands: Sequence[int] | None = None,
    *,
    dtype: str | np.dtype | None = None,
) -> GeoRaster:
    """Read a GeoTIFF/TIFF as native or explicitly converted ``(C, H, W)`` data.

    The returned mask is false for masked, nodata, or non-finite pixels. Raw
    invalid values remain in ``data`` until ``preprocess_raster`` is called.
    When ``dtype`` is omitted, Rasterio's native file dtype is preserved.
    Supplying ``dtype`` explicitly requests conversion during the read.
    """

    source_path = validate_raster_path(path)
    requested_dtype: str | None = None
    if dtype is not None:
        try:
            normalized_dtype = np.dtype(dtype)
        except TypeError as error:
            raise GeoValidationError(f"Requested raster dtype {dtype!r} is invalid.") from error
        if not np.issubdtype(normalized_dtype, np.number):
            raise GeoValidationError(f"Requested raster dtype {dtype!r} must be numeric.")
        requested_dtype = normalized_dtype.name
    try:
        with rasterio.open(source_path) as dataset:
            band_indexes = normalize_band_indexes(bands, dataset.count)
            if requested_dtype is None:
                data = dataset.read(band_indexes)
            else:
                data = dataset.read(band_indexes, out_dtype=requested_dtype)
            valid_mask = np.all(dataset.read_masks(band_indexes) > 0, axis=0)
            nodata = None if dataset.nodata is None else float(dataset.nodata)
            valid_mask &= _nodata_value_mask(data, nodata)
            valid_mask &= np.all(np.isfinite(data), axis=0)
            metadata = _metadata_from_dataset(
                dataset,
                source_path,
                band_indexes,
                loaded_dtype=np.dtype(data.dtype).name,
            )
    except (rasterio.errors.RasterioIOError, ValueError, TypeError) as error:
        raise GeoReadError(f"Unable to read raster {source_path}: {error}") from error

    return GeoRaster(data=data, metadata=metadata, valid_mask=valid_mask)


def read_geotiff(
    path: str | Path,
    bands: Sequence[int] | None = None,
    *,
    dtype: str | np.dtype | None = None,
) -> GeoRaster:
    """Compatibility-named wrapper around :func:`read_raster`."""

    return read_raster(path, bands, dtype=dtype)
