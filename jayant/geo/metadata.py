"""Raster metadata extraction without exposing open Rasterio datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import rasterio
from rasterio.io import DatasetReader

from .exceptions import GeoReadError
from .types import RasterMetadata
from .validation import normalize_band_indexes, validate_raster_path


def _metadata_from_dataset(
    dataset: DatasetReader,
    path: Path,
    band_indexes: tuple[int, ...],
    *,
    loaded_dtype: str | None = None,
) -> RasterMetadata:
    """Build metadata whose dtype matches the associated loaded array."""

    descriptions = tuple(dataset.descriptions[index - 1] for index in band_indexes)
    scales = tuple(float(dataset.scales[index - 1]) for index in band_indexes)
    offsets = tuple(float(dataset.offsets[index - 1]) for index in band_indexes)
    units = tuple(dataset.units[index - 1] for index in band_indexes)
    nodata = None if dataset.nodata is None else float(dataset.nodata)
    native_dtype = str(dataset.dtypes[band_indexes[0] - 1])
    return RasterMetadata(
        path=path,
        width=int(dataset.width),
        height=int(dataset.height),
        count=len(band_indexes),
        source_count=int(dataset.count),
        source_dtype=loaded_dtype or native_dtype,
        crs=dataset.crs,
        transform=dataset.transform,
        resolution=(float(dataset.res[0]), float(dataset.res[1])),
        bounds=(
            float(dataset.bounds.left),
            float(dataset.bounds.bottom),
            float(dataset.bounds.right),
            float(dataset.bounds.top),
        ),
        nodata=nodata,
        band_indexes=band_indexes,
        band_descriptions=descriptions,
        scales=scales,
        offsets=offsets,
        units=units,
        driver=str(dataset.driver),
        native_dtype=native_dtype,
    )


def read_metadata(path: str | Path, bands: Sequence[int] | None = None) -> RasterMetadata:
    """Read raster metadata while keeping the Rasterio dataset handle private."""

    source_path = validate_raster_path(path)
    try:
        with rasterio.open(source_path) as dataset:
            band_indexes = normalize_band_indexes(bands, dataset.count)
            return _metadata_from_dataset(dataset, source_path, band_indexes)
    except (rasterio.errors.RasterioIOError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise GeoReadError(f"Unable to read raster metadata from {source_path}: {error}") from error
