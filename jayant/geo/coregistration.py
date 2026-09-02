"""Rasterio-based transformation of source rasters onto reference grids."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from ._grid import bounds_are_valid, bounds_overlap, compare_grids
from .exceptions import GeoAlignmentError
from .reader import read_raster
from .types import (
    AlignmentConfig,
    AlignmentDiagnostics,
    AlignmentResult,
    GeoRaster,
    RasterMetadata,
)
from .validation import _nodata_value_mask, validate_raster

AlignmentInput: TypeAlias = str | Path | GeoRaster


def grids_match(
    first: RasterMetadata,
    second: RasterMetadata,
    tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> bool:
    """Return whether two metadata objects describe the same complete grid."""

    return compare_grids(
        first,
        second,
        absolute_tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    ).exact_common_grid


def validate_coregistration(
    first: RasterMetadata,
    second: RasterMetadata,
    *,
    tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> None:
    """Raise if two rasters do not already share one complete spatial grid."""

    comparison = compare_grids(
        first,
        second,
        absolute_tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not comparison.exact_common_grid:
        raise GeoAlignmentError(
            "Rasters are not co-registered; mismatched "
            + ", ".join(comparison.mismatches)
            + ".",
        )


def _load_alignment_input(source: AlignmentInput) -> GeoRaster:
    """Load a path or pass through an already loaded raster."""

    return source if isinstance(source, GeoRaster) else read_raster(source)


def _validated_alignment_input(source: AlignmentInput, role: str) -> GeoRaster:
    """Load and enforce the spatial/numeric prerequisites for alignment."""

    raster = _load_alignment_input(source)
    result = validate_raster(raster)
    if not result.valid:
        details = "; ".join(issue.message for issue in result.errors)
        raise GeoAlignmentError(f"{role} raster failed alignment validation: {details}")
    return raster


def _source_bounds_in_reference(
    source: RasterMetadata,
    reference: RasterMetadata,
) -> tuple[float, float, float, float]:
    """Return source bounds expressed in the reference CRS."""

    if source.crs is None or reference.crs is None:
        raise GeoAlignmentError("Both source and reference rasters must have a CRS; no CRS will be guessed.")
    if not bounds_are_valid(source.bounds) or not bounds_are_valid(reference.bounds):
        raise GeoAlignmentError("Source and reference rasters must have valid bounds before alignment.")
    if source.crs == reference.crs:
        transformed_bounds = source.bounds
    else:
        try:
            transformed_bounds = tuple(
                transform_bounds(source.crs, reference.crs, *source.bounds, densify_pts=21),
            )
        except (rasterio.errors.RasterioError, ValueError, TypeError) as error:
            raise GeoAlignmentError(f"Could not transform source bounds into the reference CRS: {error}") from error
    if not bounds_are_valid(transformed_bounds):
        raise GeoAlignmentError(
            "Transformed source bounds are invalid; expected finite bounds with min < max on both axes.",
        )
    return transformed_bounds


def _require_spatial_overlap(source: RasterMetadata, reference: RasterMetadata) -> tuple[float, float, float, float]:
    """Raise for non-overlap and return source bounds in the reference CRS."""

    source_bounds = _source_bounds_in_reference(source, reference)
    if not bounds_overlap(reference.bounds, source_bounds):
        raise GeoAlignmentError(
            "Source and reference rasters do not overlap spatially; alignment would produce no source coverage.",
        )
    return source_bounds


def _destination_value_for_dtype(value: float, dtype: np.dtype) -> int | float:
    """Return a destination nodata value safely representable by ``dtype``."""

    if dtype.kind in "iu":
        if not np.isfinite(value) or float(value) != float(int(value)):
            raise GeoAlignmentError(f"Destination nodata {value!r} is not an integer value for dtype {dtype}.")
        limits = np.iinfo(dtype)
        integer_value = int(value)
        if integer_value < limits.min or integer_value > limits.max:
            raise GeoAlignmentError(f"Destination nodata {value!r} is outside the range of dtype {dtype}.")
        return integer_value
    if dtype.kind == "f":
        limits = np.finfo(dtype)
        if not np.isfinite(value) or value < limits.min or value > limits.max:
            raise GeoAlignmentError(f"Destination nodata {value!r} is not representable by dtype {dtype}.")
        return dtype.type(value).item()
    if dtype.kind == "c":
        return dtype.type(value).item()
    raise GeoAlignmentError(f"Raster dtype {dtype} cannot be used for geospatial alignment.")


def _effective_source_valid_mask(source: GeoRaster, config: AlignmentConfig) -> np.ndarray:
    """Return the authoritative mask with optional metadata nodata excluded."""

    effective_mask = np.array(source.valid_mask, dtype=bool, copy=True)
    if config.nodata_policy == "respect_source":
        effective_mask &= _nodata_value_mask(source.data, source.metadata.nodata)
    return effective_mask


def _reproject_data(
    source: GeoRaster,
    reference: GeoRaster,
    config: AlignmentConfig,
    destination_nodata: int | float,
    source_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject all source bands and the valid mask onto the reference grid."""

    source_dtype = source.data.dtype
    destination = np.full(
        (source.channels, reference.metadata.height, reference.metadata.width),
        destination_nodata,
        dtype=source_dtype,
    )
    source_nodata = source.metadata.nodata if config.nodata_policy == "respect_source" else None
    try:
        for channel in range(source.channels):
            masked_band = np.ma.array(source.data[channel], mask=~source_valid_mask, copy=False)
            reproject(
                source=masked_band,
                destination=destination[channel],
                src_transform=source.metadata.transform,
                src_crs=source.metadata.crs,
                src_nodata=source_nodata,
                dst_transform=reference.metadata.transform,
                dst_crs=reference.metadata.crs,
                dst_nodata=destination_nodata,
                init_dest_nodata=True,
                resampling=config.resampling,
            )

        valid_destination = np.zeros(
            (reference.metadata.height, reference.metadata.width),
            dtype=np.uint8,
        )
        reproject(
            source=source_valid_mask.astype(np.uint8),
            destination=valid_destination,
            src_transform=source.metadata.transform,
            src_crs=source.metadata.crs,
            src_nodata=0,
            dst_transform=reference.metadata.transform,
            dst_crs=reference.metadata.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    except (rasterio.errors.RasterioError, ValueError, TypeError) as error:
        raise GeoAlignmentError(f"Rasterio failed to transform the source onto the reference grid: {error}") from error

    valid_mask = valid_destination > 0
    destination[:, ~valid_mask] = destination_nodata
    return destination, valid_mask


def _build_output_metadata(
    source: GeoRaster,
    reference: GeoRaster,
    aligned_array: np.ndarray,
    destination_nodata: int | float,
) -> RasterMetadata:
    """Build metadata consistent with aligned source data on the reference grid."""

    return replace(
        source.metadata,
        width=reference.metadata.width,
        height=reference.metadata.height,
        count=int(aligned_array.shape[0]),
        source_dtype=np.dtype(aligned_array.dtype).name,
        crs=reference.metadata.crs,
        transform=reference.metadata.transform,
        resolution=reference.metadata.resolution,
        bounds=reference.metadata.bounds,
        nodata=destination_nodata,
    )


def _verify_output(
    aligned_array: np.ndarray,
    aligned_valid_mask: np.ndarray,
    output_metadata: RasterMetadata,
    reference: GeoRaster,
    config: AlignmentConfig,
    destination_nodata: int | float,
) -> None:
    """Verify the generated output metadata and array contract independently."""

    if output_metadata.crs != reference.metadata.crs:
        raise GeoAlignmentError("Aligned output CRS does not match the reference CRS.")
    if output_metadata.width != reference.metadata.width:
        raise GeoAlignmentError("Aligned output width does not match the reference width.")
    if output_metadata.height != reference.metadata.height:
        raise GeoAlignmentError("Aligned output height does not match the reference height.")

    output_grid = compare_grids(
        output_metadata,
        reference.metadata,
        absolute_tolerance=config.tolerance,
        relative_tolerance=config.relative_tolerance,
    )
    if not output_grid.exact_common_grid:
        raise GeoAlignmentError(
            "Aligned output metadata does not match the reference grid: "
            + ", ".join(output_grid.mismatches)
            + ".",
        )

    if aligned_array.ndim != 3:
        raise GeoAlignmentError("Aligned output array must have shape (C, H, W).")
    expected_shape = (output_metadata.count, output_metadata.height, output_metadata.width)
    if aligned_array.shape != expected_shape:
        raise GeoAlignmentError(
            f"Aligned output shape {aligned_array.shape} does not match output metadata {expected_shape}.",
        )
    if output_metadata.count != aligned_array.shape[0]:
        raise GeoAlignmentError("Aligned output band count does not match output metadata.")
    try:
        output_dtype = np.dtype(output_metadata.dtype)
    except TypeError as error:
        raise GeoAlignmentError("Aligned output metadata contains an invalid dtype.") from error
    if aligned_array.dtype != output_dtype:
        raise GeoAlignmentError(
            f"Aligned output dtype {aligned_array.dtype} does not match output metadata dtype {output_dtype}.",
        )
    if aligned_valid_mask.shape != output_metadata.shape or aligned_valid_mask.dtype != np.bool_:
        raise GeoAlignmentError("Aligned valid mask does not match output metadata (H, W) boolean contract.")
    if output_metadata.nodata != destination_nodata:
        raise GeoAlignmentError("Aligned output nodata does not match the configured destination nodata.")


def align_raster_to_reference(
    source: AlignmentInput,
    reference: AlignmentInput,
    config: AlignmentConfig | None = None,
) -> AlignmentResult:
    """Transform ``source`` onto the exact spatial grid of ``reference``.

    Rasterio performs the geospatial transformation. Same-grid inputs use a
    copy-based fast path. Partial-overlap inputs receive
    ``destination_nodata`` outside source coverage, and completely
    non-overlapping inputs are rejected before transformation.
    """

    policy = config or AlignmentConfig()
    source_raster = _validated_alignment_input(source, "Source")
    reference_raster = _validated_alignment_input(reference, "Reference")
    source_bounds = _require_spatial_overlap(source_raster.metadata, reference_raster.metadata)
    comparison = compare_grids(
        source_raster.metadata,
        reference_raster.metadata,
        absolute_tolerance=policy.tolerance,
        relative_tolerance=policy.relative_tolerance,
    )
    same_grid = comparison.exact_common_grid
    destination_nodata = _destination_value_for_dtype(policy.destination_nodata, source_raster.data.dtype)
    source_valid_mask = _effective_source_valid_mask(source_raster, policy)

    if same_grid:
        aligned_array = np.array(source_raster.data, copy=True)
        aligned_valid_mask = np.array(source_valid_mask, dtype=bool, copy=True)
        aligned_array[:, ~aligned_valid_mask] = destination_nodata
    else:
        aligned_array, aligned_valid_mask = _reproject_data(
            source_raster,
            reference_raster,
            policy,
            destination_nodata,
            source_valid_mask,
        )

    output_metadata = _build_output_metadata(
        source_raster,
        reference_raster,
        aligned_array,
        destination_nodata,
    )
    _verify_output(
        aligned_array,
        aligned_valid_mask,
        output_metadata,
        reference_raster,
        policy,
        destination_nodata,
    )
    diagnostics = AlignmentDiagnostics(
        source_crs=source_raster.metadata.crs.to_string() if source_raster.metadata.crs is not None else None,
        reference_crs=reference_raster.metadata.crs.to_string() if reference_raster.metadata.crs is not None else None,
        source_resolution=source_raster.metadata.resolution,
        reference_resolution=reference_raster.metadata.resolution,
        source_dimensions=source_raster.metadata.shape,
        reference_dimensions=reference_raster.metadata.shape,
        reprojected=not same_grid,
        resampled=not same_grid,
        resampling_method=policy.resampling.name,
        grid_comparison=comparison.as_dict(),
        spatial_overlap=True,
        overlap_bounds_in_reference=source_bounds,
        source_nodata=source_raster.metadata.nodata,
        destination_nodata=float(destination_nodata),
        nodata_policy=policy.nodata_policy,
        source_dtype=np.dtype(source_raster.data.dtype).name,
        aligned_dtype=np.dtype(aligned_array.dtype).name,
    )
    return AlignmentResult(
        aligned_array=aligned_array,
        aligned_valid_mask=aligned_valid_mask,
        output_metadata=output_metadata,
        reference_metadata=reference_raster.metadata,
        source_metadata=source_raster.metadata,
        alignment_diagnostics=diagnostics,
    )
