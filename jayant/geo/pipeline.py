"""High-level orchestration for Geo Expert inputs.

This module only composes reading, alignment, validation, and preprocessing.
It deliberately contains no model or inference code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .pairs import read_bitemporal_pair, read_optical_sar_pair
from .preprocessing import preprocess_raster
from .reader import read_raster
from .types import AlignmentConfig, GeoRaster, PreprocessingConfig, RasterPair


def prepare_single_raster(
    path: str | Path,
    bands: Sequence[int] | None = None,
    config: PreprocessingConfig | None = None,
) -> GeoRaster:
    """Read and prepare one optical, multispectral, or SAR raster."""

    return preprocess_raster(read_raster(path, bands), config)


def prepare_optical_sar_pair(
    optical_path: str | Path,
    sar_path: str | Path,
    optical_bands: Sequence[int] | None = None,
    sar_bands: Sequence[int] | None = None,
    *,
    optical_config: PreprocessingConfig | None = None,
    sar_config: PreprocessingConfig | None = None,
) -> RasterPair:
    """Read, validate, and prepare a co-registered optical/SAR pair."""

    pair = read_optical_sar_pair(optical_path, sar_path, optical_bands, sar_bands)
    return RasterPair(
        first=preprocess_raster(pair.first, optical_config),
        second=preprocess_raster(pair.second, sar_config),
    )


def prepare_bitemporal_pair(
    first_path: str | Path,
    second_path: str | Path,
    bands: Sequence[int] | None = None,
    *,
    alignment: AlignmentConfig | None = None,
    config: PreprocessingConfig | None = None,
) -> RasterPair:
    """Read, optionally align, and prepare a bi-temporal raster pair."""

    pair = read_bitemporal_pair(first_path, second_path, bands, alignment=alignment)
    return RasterPair(
        first=preprocess_raster(pair.first, config),
        second=preprocess_raster(pair.second, config),
    )
