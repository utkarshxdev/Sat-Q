"""Readers for co-registered optical/SAR and bi-temporal raster pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .coregistration import align_raster_to_reference, validate_coregistration
from .exceptions import GeoAlignmentError
from .reader import read_raster
from .types import AlignmentConfig, GeoRaster, RasterPair
from .validation import validate_geo_raster


def _pair(first: GeoRaster, second: GeoRaster) -> RasterPair:
    """Validate pair dimensions and construct the public pair object."""

    validate_geo_raster(first)
    validate_geo_raster(second)
    if first.data.shape[1:] != second.data.shape[1:]:
        raise GeoAlignmentError("Paired rasters must have equal height and width.")
    return RasterPair(first=first, second=second)


def read_coregistered_pair(
    first_path: str | Path,
    second_path: str | Path,
    first_bands: Sequence[int] | None = None,
    second_bands: Sequence[int] | None = None,
) -> RasterPair:
    """Read two rasters and require identical CRS, transform, bounds, and shape."""

    first = read_raster(first_path, first_bands)
    second = read_raster(second_path, second_bands)
    validate_coregistration(first.metadata, second.metadata)
    return _pair(first, second)


def read_optical_sar_pair(
    optical_path: str | Path,
    sar_path: str | Path,
    optical_bands: Sequence[int] | None = None,
    sar_bands: Sequence[int] | None = None,
) -> RasterPair:
    """Read a strictly co-registered optical/SAR pair."""

    return read_coregistered_pair(optical_path, sar_path, optical_bands, sar_bands)


def read_bitemporal_pair(
    first_path: str | Path,
    second_path: str | Path,
    bands: Sequence[int] | None = None,
    *,
    alignment: AlignmentConfig | None = None,
) -> RasterPair:
    """Read a bi-temporal pair, optionally aligning the second raster.

    Without ``alignment`` the two source grids must already match. Supplying
    an ``AlignmentConfig`` explicitly opts into reprojection of the second
    image onto the first image's grid.
    """

    first = read_raster(first_path, bands)
    second = read_raster(second_path, bands)
    if first.channels != second.channels:
        raise GeoAlignmentError("Bi-temporal rasters must have the same channel count.")
    if alignment is not None:
        second = align_raster_to_reference(second, first, alignment).as_geo_raster()
    else:
        validate_coregistration(first.metadata, second.metadata)
    return _pair(first, second)
