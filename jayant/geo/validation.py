"""Validation helpers for paths, arrays, raster containers, and band indexes."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, transform_bounds

from ._grid import GridComparison, bounds_are_valid, bounds_overlap, compare_grids
from .exceptions import GeoFormatError, GeoReadError, GeoValidationError
from .types import (
    GeoRaster,
    NodataKind,
    PairValidationResult,
    RasterMetadata,
    RasterValidationConfig,
    RasterValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

SUPPORTED_RASTER_EXTENSIONS = frozenset({".tif", ".tiff"})
RasterInput: TypeAlias = str | Path | GeoRaster


def validate_raster_path(path: str | Path) -> Path:
    """Validate and normalize a readable GeoTIFF/TIFF path."""

    try:
        source_path = Path(path).expanduser()
    except TypeError as error:
        raise GeoValidationError("Raster path must be a string or pathlib.Path.") from error
    if not source_path.exists():
        raise GeoValidationError(f"Raster path does not exist: {source_path}")
    if not source_path.is_file():
        raise GeoValidationError(f"Raster path is not a file: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_RASTER_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_RASTER_EXTENSIONS))
        raise GeoFormatError(f"Unsupported raster extension '{source_path.suffix or '<none>'}'; expected {supported}.")
    return source_path.resolve()


def normalize_band_indexes(
    bands: Sequence[int] | None,
    source_count: int,
) -> tuple[int, ...]:
    """Return validated, 1-based band indexes for a raster read."""

    if source_count < 1:
        raise GeoValidationError("Raster must contain at least one band.")
    if bands is None:
        return tuple(range(1, source_count + 1))
    try:
        indexes = tuple(operator.index(band) for band in bands)
    except TypeError as error:
        raise GeoValidationError("Band indexes must be integers.") from error
    if not indexes:
        raise GeoValidationError("At least one band must be selected.")
    if any(index < 1 or index > source_count for index in indexes):
        raise GeoValidationError(f"Band indexes must be between 1 and {source_count}.")
    return indexes


def validate_array(
    array: np.ndarray,
    *,
    name: str = "array",
    expected_channels: int | None = None,
    require_finite: bool = True,
) -> np.ndarray:
    """Validate an input array and return it as a NumPy view when possible.

    Model-facing arrays must be channel-first: ``(C, H, W)``. Raster data
    containing invalid pixels may set ``require_finite=False`` until the
    preprocessing step fills those pixels.
    """

    result = np.asarray(array)
    if result.ndim != 3:
        raise GeoValidationError(f"{name} must have shape (C, H, W).")
    if any(size < 1 for size in result.shape):
        raise GeoValidationError(f"{name} must have non-empty dimensions.")
    if not np.issubdtype(result.dtype, np.number):
        raise GeoValidationError(f"{name} must contain numeric values.")
    if expected_channels is not None and result.shape[0] != expected_channels:
        raise GeoValidationError(
            f"{name} must have {expected_channels} channels; got {result.shape[0]}.",
        )
    if require_finite and not np.isfinite(result).all():
        raise GeoValidationError(f"{name} must contain only finite values.")
    return result


def _nodata_value_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return pixels that do not equal the supplied metadata nodata value."""

    if nodata is None:
        return np.ones(data.shape[1:], dtype=bool)
    if np.isnan(nodata):
        return ~np.any(np.isnan(data), axis=0)
    return ~np.any(data == nodata, axis=0)


def validate_metadata(metadata: RasterMetadata) -> None:
    """Validate the dimensions and selected-band fields in raster metadata."""

    if metadata.width < 1 or metadata.height < 1:
        raise GeoValidationError("Raster dimensions must be positive.")
    if metadata.count < 1 or len(metadata.band_indexes) != metadata.count:
        raise GeoValidationError("Metadata band count does not match band indexes.")
    if metadata.source_count < metadata.count:
        raise GeoValidationError("Selected band count cannot exceed source band count.")


def validate_geo_raster(raster: GeoRaster) -> None:
    """Validate a ``GeoRaster`` container and its channel-first contract."""

    validate_metadata(raster.metadata)
    if raster.data.ndim != 3:
        raise GeoValidationError("GeoRaster.data must have shape (C, H, W).")
    if raster.data.shape != (raster.metadata.count, raster.metadata.height, raster.metadata.width):
        raise GeoValidationError("GeoRaster.data shape does not match its metadata.")
    if raster.valid_mask.shape != (raster.metadata.height, raster.metadata.width):
        raise GeoValidationError("GeoRaster.valid_mask must have shape (H, W).")
    if raster.valid_mask.dtype != np.bool_:
        raise GeoValidationError("GeoRaster.valid_mask must have boolean dtype.")


def _issue(
    code: str,
    message: str,
    *,
    path: Path | None = None,
    field: str | None = None,
    severity: ValidationSeverity = "error",
) -> ValidationIssue:
    """Create a consistently shaped validation issue."""

    return ValidationIssue(code=code, message=message, path=path, field=field, severity=severity)


def _issues_are_valid(issues: Sequence[ValidationIssue]) -> bool:
    """Return whether a collection contains no fatal findings."""

    return not any(issue.severity == "error" for issue in issues)


def _nodata_kind(value: float | None) -> NodataKind:
    """Classify a nodata marker without treating NaN as automatically invalid."""

    if value is None:
        return "none"
    if np.isnan(value):
        return "nan"
    return "numeric"


def _load_for_validation(source: RasterInput) -> tuple[GeoRaster | None, tuple[ValidationIssue, ...]]:
    """Load a path or validate an already loaded raster without raising."""

    if isinstance(source, GeoRaster):
        try:
            validate_geo_raster(source)
        except GeoValidationError as error:
            return None, (_issue("invalid_raster_container", str(error)),)
        return source, ()

    # Import lazily because reader -> metadata -> validation is a valid module
    # dependency chain during package import.
    from .reader import read_raster

    try:
        return read_raster(source), ()
    except GeoFormatError as error:
        return None, (_issue("unsupported_format", str(error), field="path"),)
    except GeoReadError as error:
        return None, (_issue("unreadable_raster", str(error), field="path"),)
    except GeoValidationError as error:
        message = str(error)
        if "does not exist" in message:
            code = "missing_file"
        elif "not a file" in message:
            code = "not_a_file"
        else:
            code = "invalid_path"
        return None, (_issue(code, message, field="path"),)
    except (TypeError, ValueError, OSError) as error:
        return None, (_issue("unreadable_raster", f"Unable to read raster: {error}", field="path"),)


def _validate_loaded_raster(
    raster: GeoRaster,
    config: RasterValidationConfig,
) -> tuple[ValidationIssue, ...]:
    """Check metadata and numeric content of an already loaded raster."""

    metadata = raster.metadata
    path = metadata.path
    issues: list[ValidationIssue] = []
    if config.exact_band_count is not None and metadata.count != config.exact_band_count:
        issues.append(
            _issue(
                "invalid_band_count",
                f"Expected exactly {config.exact_band_count} bands; found {metadata.count}.",
                path=path,
                field="band_count",
            ),
        )
    if config.min_band_count is not None and metadata.count < config.min_band_count:
        issues.append(
            _issue(
                "band_count_below_minimum",
                f"Expected at least {config.min_band_count} bands; found {metadata.count}.",
                path=path,
                field="band_count",
            ),
        )
    if config.max_band_count is not None and metadata.count > config.max_band_count:
        issues.append(
            _issue(
                "band_count_above_maximum",
                f"Expected at most {config.max_band_count} bands; found {metadata.count}.",
                path=path,
                field="band_count",
            ),
        )

    try:
        np.dtype(metadata.source_dtype)
    except TypeError:
        issues.append(
            _issue(
                "invalid_dtype",
                f"Raster dtype '{metadata.source_dtype}' is not a valid NumPy dtype.",
                path=path,
                field="dtype",
            ),
        )
    if config.allowed_dtypes is not None and metadata.source_dtype not in config.allowed_dtypes:
        allowed = ", ".join(config.allowed_dtypes)
        issues.append(
            _issue(
                "unsupported_dtype",
                f"Raster dtype '{metadata.source_dtype}' is not allowed; expected one of: {allowed}.",
                path=path,
                field="dtype",
            ),
        )

    if config.require_crs and metadata.crs is None:
        issues.append(_issue("missing_crs", "Raster has no CRS; geographic mapping is undefined.", path=path, field="crs"))

    transform_values = tuple(metadata.transform) if metadata.transform is not None else ()
    transform_valid = len(transform_values) == 9 and bool(np.isfinite(transform_values).all())
    if config.require_transform and (not transform_valid or metadata.transform.is_identity):
        issues.append(
            _issue(
                "missing_transform",
                "Raster has no usable affine transform for mapping pixels to coordinates.",
                path=path,
                field="transform",
            ),
        )

    resolution = metadata.resolution
    resolution_valid = (
        len(resolution) == 2
        and bool(np.isfinite(resolution).all())
        and resolution[0] > 0.0
        and resolution[1] > 0.0
    )
    if config.require_resolution and not resolution_valid:
        issues.append(
            _issue(
                "invalid_resolution",
                f"Raster resolution must contain two positive finite values; found {resolution!r}.",
                path=path,
                field="resolution",
            ),
        )

    bounds = metadata.bounds
    bounds_valid = (
        len(bounds) == 4
        and bool(np.isfinite(bounds).all())
        and bounds[0] < bounds[2]
        and bounds[1] < bounds[3]
    )
    if not bounds_valid:
        issues.append(
            _issue(
                "invalid_bounds",
                f"Raster bounds must be finite and ordered as (left, bottom, right, top); found {bounds!r}.",
                path=path,
                field="bounds",
            ),
        )

    nodata_kind = _nodata_kind(metadata.nodata)
    if config.nodata_policy == "optional":
        if nodata_kind == "none":
            issues.append(
                _issue(
                    "missing_nodata",
                    "Raster does not declare a nodata value; downstream masking must provide one.",
                    path=path,
                    field="nodata",
                    severity="error" if config.require_nodata else "warning",
                ),
            )
    elif config.nodata_policy == "numeric":
        if nodata_kind == "none":
            issues.append(
                _issue(
                    "missing_nodata",
                    "Raster validation requires a finite numeric nodata value, but none is declared.",
                    path=path,
                    field="nodata",
                ),
            )
        elif nodata_kind == "nan" or not np.isfinite(metadata.nodata):
            issues.append(
                _issue(
                    "nodata_policy_mismatch",
                    f"Raster nodata value {metadata.nodata!r} is not a finite numeric marker.",
                    path=path,
                    field="nodata",
                ),
            )
    elif config.nodata_policy == "nan":
        if nodata_kind == "none":
            issues.append(
                _issue(
                    "missing_nodata",
                    "Raster validation requires NaN nodata, but no nodata value is declared.",
                    path=path,
                    field="nodata",
                ),
            )
        elif nodata_kind != "nan":
            issues.append(
                _issue(
                    "nodata_policy_mismatch",
                    f"Raster nodata value {metadata.nodata!r} is not NaN as required by the validation policy.",
                    path=path,
                    field="nodata",
                ),
            )
    elif nodata_kind != "none":
        issues.append(
            _issue(
                "nodata_policy_mismatch",
                f"Raster declares nodata={metadata.nodata!r}, but the validation policy requires no nodata metadata.",
                path=path,
                field="nodata",
            ),
        )

    if not np.issubdtype(raster.data.dtype, np.number):
        issues.append(
            _issue(
                "invalid_numeric_dtype",
                f"Raster array dtype '{raster.data.dtype}' is not numeric.",
                path=path,
                field="data",
            ),
        )
        invalid_values = 0
    else:
        invalid_numeric_mask = ~np.isfinite(raster.data)
        if _nodata_kind(metadata.nodata) == "nan":
            # A declared NaN nodata marker is expected to be non-finite. Any
            # remaining infinities are still invalid numeric source values.
            invalid_numeric_mask &= ~np.isnan(raster.data)
        invalid_values = int(invalid_numeric_mask.sum())
    if invalid_values:
        issues.append(
            _issue(
                "invalid_numeric_values",
                f"Raster contains {invalid_values} NaN or infinite numeric values.",
                path=path,
                field="data",
                severity="error" if config.require_finite else "warning",
            ),
        )
    return tuple(issues)


def validate_raster(
    source: RasterInput,
    config: RasterValidationConfig | None = None,
) -> RasterValidationResult:
    """Validate one raster path or loaded ``GeoRaster`` without modifying it."""

    raster, load_issues = _load_for_validation(source)
    issues = list(load_issues)
    if raster is not None:
        issues.extend(_validate_loaded_raster(raster, config or RasterValidationConfig()))
        return RasterValidationResult(
            valid=_issues_are_valid(issues),
            issues=tuple(issues),
            metadata=raster.metadata,
            nodata_kind=_nodata_kind(raster.metadata.nodata),
        )
    return RasterValidationResult(valid=False, issues=tuple(issues))


def _can_transform_to_reference(source: RasterMetadata, reference: RasterMetadata) -> bool:
    """Check whether Rasterio can calculate a CRS/grid transformation.

    This is deliberately weaker than co-registration: it does not reproject
    pixels, compare footprints, or prove that the source is on the reference
    grid.
    """

    if source.crs is None or reference.crs is None:
        return False
    if source.transform is None or reference.transform is None:
        return False
    source_transform = tuple(source.transform)
    reference_transform = tuple(reference.transform)
    if (
        len(source_transform) != 9
        or len(reference_transform) != 9
        or not np.isfinite(source_transform).all()
        or not np.isfinite(reference_transform).all()
        or source.transform.is_identity
        or reference.transform.is_identity
        or not bounds_are_valid(source.bounds)
        or source.width < 1
        or source.height < 1
    ):
        return False
    try:
        calculate_default_transform(
            source.crs,
            reference.crs,
            source.width,
            source.height,
            *source.bounds,
        )
    except (rasterio.errors.RasterioError, ValueError, TypeError):
        return False
    return True


def _same_nodata(first: float | None, second: float | None) -> bool:
    """Compare numeric, NaN, and absent nodata markers semantically."""

    if first is None or second is None:
        return first is None and second is None
    if np.isnan(first) or np.isnan(second):
        return bool(np.isnan(first) and np.isnan(second))
    return bool(first == second)


def _pair_grid_tolerances(
    first_config: RasterValidationConfig,
    second_config: RasterValidationConfig,
) -> tuple[float, float]:
    """Use the stricter member tolerance for a pair comparison."""

    return (
        min(first_config.grid_absolute_tolerance, second_config.grid_absolute_tolerance),
        min(first_config.grid_relative_tolerance, second_config.grid_relative_tolerance),
    )


def _validate_pair_metadata(
    first: RasterMetadata,
    second: RasterMetadata,
    *,
    pair_name: str,
    require_equal_channels: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[
    tuple[ValidationIssue, ...],
    bool | None,
    bool,
    bool,
    bool | None,
    bool | None,
    GridComparison,
]:
    """Validate spatial compatibility without resampling either raster."""

    issues: list[ValidationIssue] = []
    second_path = second.path
    if require_equal_channels and first.count != second.count:
        issues.append(
            _issue(
                "different_band_counts",
                f"{pair_name} rasters must have the same selected band count; found {first.count} and {second.count}.",
                path=second_path,
                field="band_count",
            ),
        )
    grid = compare_grids(
        first,
        second,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not grid.crs_match:
        issues.append(
            _issue(
                "mismatched_crs",
                f"{pair_name} rasters use different CRSs: {first.crs!s} and {second.crs!s}.",
                path=second_path,
                field="crs",
            ),
        )
    if not grid.dimensions_match:
        issues.append(
            _issue(
                "different_dimensions",
                f"{pair_name} rasters have different spatial dimensions: {first.shape} and {second.shape}.",
                path=second_path,
                field="dimensions",
            ),
        )
    if not grid.resolution_match:
        issues.append(
            _issue(
                "different_resolution",
                f"{pair_name} rasters have different resolutions: {first.resolution} and {second.resolution}.",
                path=second_path,
                field="resolution",
            ),
        )
    if not grid.transform_match:
        issues.append(
            _issue(
                "different_transform",
                f"{pair_name} rasters have different affine transforms.",
                path=second_path,
                field="transform",
            ),
        )
    if not grid.bounds_match:
        issues.append(
            _issue(
                "different_bounds",
                f"{pair_name} rasters have different bounds: {first.bounds} and {second.bounds}.",
                path=second_path,
                field="bounds",
            ),
        )

    spatial_overlap: bool | None = None
    if first.crs is not None and second.crs is not None and bounds_are_valid(first.bounds) and bounds_are_valid(second.bounds):
        try:
            second_bounds = (
                second.bounds
                if first.crs == second.crs
                else transform_bounds(second.crs, first.crs, *second.bounds, densify_pts=21)
            )
            spatial_overlap = bounds_overlap(first.bounds, second_bounds)
        except (rasterio.errors.RasterioError, ValueError, TypeError):
            spatial_overlap = None
            issues.append(
                _issue(
                    "overlap_unknown",
                    f"Could not determine spatial overlap for {pair_name} rasters.",
                    path=second_path,
                    field="bounds",
                ),
            )
    elif first.crs is None or second.crs is None:
        issues.append(
            _issue(
                "overlap_unknown",
                f"Spatial overlap for {pair_name} rasters cannot be established without a CRS on both rasters.",
                path=second_path,
                field="crs",
            ),
        )
    else:
        issues.append(
            _issue(
                "overlap_unknown",
                f"Spatial overlap for {pair_name} rasters cannot be established from invalid bounds.",
                path=second_path,
                field="bounds",
            ),
        )
    if spatial_overlap is False:
        issues.append(
            _issue(
                "non_overlapping_rasters",
                f"{pair_name} rasters have no positive-area spatial overlap.",
                path=second_path,
                field="bounds",
            ),
        )

    can_transform = _can_transform_to_reference(second, first)
    if not can_transform:
        issues.append(
            _issue(
                "cannot_transform_to_common_grid",
                f"Rasterio cannot calculate a valid transformation from the second {pair_name} raster to the first raster's grid; this is separate from co-registration.",
                path=second_path,
                field="grid",
            ),
        )
    if spatial_overlap is True and not grid.exact_common_grid:
        issues.append(
            _issue(
                "overlap_not_common_grid",
                f"{pair_name} rasters overlap spatially but are not on the same common grid.",
                path=second_path,
                field="grid",
                severity="warning",
            ),
        )

    dtype_match = first.source_dtype == second.source_dtype
    nodata_match = _same_nodata(first.nodata, second.nodata)
    if pair_name == "bi-temporal" and not dtype_match:
        issues.append(
            _issue(
                "different_dtype",
                f"Bi-temporal rasters use different source dtypes: {first.source_dtype} and {second.source_dtype}; this is diagnostic only.",
                path=second_path,
                field="dtype",
                severity="warning",
            ),
        )
    if pair_name == "bi-temporal" and not nodata_match:
        issues.append(
            _issue(
                "different_nodata",
                f"Bi-temporal rasters use different nodata markers: {first.nodata!r} and {second.nodata!r}; this is diagnostic only.",
                path=second_path,
                field="nodata",
                severity="warning",
            ),
        )
    return tuple(issues), spatial_overlap, grid.exact_common_grid, can_transform, dtype_match, nodata_match, grid


def _validate_pair(
    first_source: RasterInput,
    second_source: RasterInput,
    *,
    first_config: RasterValidationConfig | None,
    second_config: RasterValidationConfig | None,
    pair_name: str,
    require_equal_channels: bool,
) -> PairValidationResult:
    """Shared implementation for the public pair validators."""

    first_policy = first_config or RasterValidationConfig()
    second_policy = second_config or RasterValidationConfig()
    absolute_tolerance, relative_tolerance = _pair_grid_tolerances(first_policy, second_policy)
    first = validate_raster(first_source, first_policy)
    second = validate_raster(second_source, second_policy)
    issues = list(first.issues) + list(second.issues)
    spatial_overlap: bool | None = None
    can_transform: bool | None = None
    exact_common_grid: bool | None = None
    grid_mismatches: tuple[str, ...] = ()
    dtype_match: bool | None = None
    nodata_match: bool | None = None
    if first.metadata is not None and second.metadata is not None:
        (
            pair_issues,
            spatial_overlap,
            exact_common_grid,
            can_transform,
            dtype_match,
            nodata_match,
            grid,
        ) = _validate_pair_metadata(
            first.metadata,
            second.metadata,
            pair_name=pair_name,
            require_equal_channels=require_equal_channels,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        grid_mismatches = grid.mismatches
        issues.extend(pair_issues)
    return PairValidationResult(
        valid=_issues_are_valid(issues),
        issues=tuple(issues),
        first=first,
        second=second,
        spatial_overlap=spatial_overlap,
        can_transform_to_common_grid=can_transform,
        exact_common_grid=exact_common_grid,
        grid_mismatches=grid_mismatches,
        dtype_match=dtype_match,
        nodata_match=nodata_match,
    )


def validate_optical_sar_pair(
    optical: RasterInput,
    sar: RasterInput,
    optical_config: RasterValidationConfig | None = None,
    sar_config: RasterValidationConfig | None = None,
) -> PairValidationResult:
    """Validate optical/SAR members and their spatial-grid compatibility."""

    return _validate_pair(
        optical,
        sar,
        first_config=optical_config,
        second_config=sar_config,
        pair_name="optical/SAR",
        require_equal_channels=False,
    )


def validate_bitemporal_pair(
    first: RasterInput,
    second: RasterInput,
    config: RasterValidationConfig | None = None,
) -> PairValidationResult:
    """Validate bi-temporal members and require compatible channel counts and grids."""

    return _validate_pair(
        first,
        second,
        first_config=config,
        second_config=config,
        pair_name="bi-temporal",
        require_equal_channels=True,
    )
