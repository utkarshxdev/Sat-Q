"""Public data contracts for geospatial input processing.

The objects in this module intentionally hide Rasterio dataset handles. A
downstream consumer receives NumPy data, a valid-pixel mask, and the small set
of geospatial fields needed to map predictions back to the source image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

Bounds = tuple[float, float, float, float]
NormalizationMode = Literal["none", "minmax", "zscore"]
ValidationSeverity = Literal["error", "warning"]
NodataPolicy = Literal["optional", "numeric", "nan", "none"]
NodataKind = Literal["numeric", "nan", "none"]
AlignmentNodataPolicy = Literal["respect_source", "mask_only"]


@dataclass(frozen=True, slots=True)
class RasterMetadata:
    """Metadata describing a raster array and its selected source bands.

    ``transform`` maps pixel coordinates to the coordinate system identified
    by ``crs``. ``band_indexes`` are the original 1-based source indexes used
    to create the returned channel-first array. ``source_dtype`` always
    matches the dtype of the associated loaded array. ``native_dtype`` keeps
    the file's storage dtype when a caller explicitly requests conversion.
    """

    path: Path
    width: int
    height: int
    count: int
    source_count: int
    source_dtype: str
    crs: CRS | None
    transform: Affine
    resolution: tuple[float, float]
    bounds: Bounds
    nodata: float | None
    band_indexes: tuple[int, ...]
    band_descriptions: tuple[str | None, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    units: tuple[str | None, ...]
    driver: str
    native_dtype: str | None = None

    @property
    def dtype(self) -> str:
        """Return the dtype of the associated loaded array."""

        return self.source_dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Return the spatial shape as ``(height, width)``."""

        return self.height, self.width

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly metadata for logs and geographic consumers."""

        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "source_count": self.source_count,
            "source_dtype": self.source_dtype,
            "dtype": self.source_dtype,
            "native_dtype": self.native_dtype,
            "crs": self.crs.to_string() if self.crs is not None else None,
            "transform": list(self.transform),
            "resolution": list(self.resolution),
            "bounds": list(self.bounds),
            "nodata": self.nodata,
            "band_indexes": list(self.band_indexes),
            "band_descriptions": list(self.band_descriptions),
            "scales": list(self.scales),
            "offsets": list(self.offsets),
            "units": list(self.units),
            "driver": self.driver,
        }


@dataclass(frozen=True, slots=True)
class GeoRaster:
    """A channel-first raster, its mapping metadata, and valid-pixel mask.

    For direct ``GeoRaster`` inputs, ``valid_mask`` is the authoritative
    baseline validity mask. Alignment may further exclude pixels matching
    metadata nodata when ``AlignmentConfig.nodata_policy`` is
    ``"respect_source"``.
    """

    data: np.ndarray
    metadata: RasterMetadata
    valid_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the data shape as ``(channels, height, width)``."""

        return (
            int(self.data.shape[0]),
            int(self.data.shape[1]),
            int(self.data.shape[2]),
        )

    @property
    def channels(self) -> int:
        """Return the number of channels in ``data``."""

        return int(self.data.shape[0])


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable validation finding."""

    code: str
    message: str
    severity: ValidationSeverity = "error"
    path: Path | None = None
    field: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the issue in a serializable form."""

        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": str(self.path) if self.path is not None else None,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class RasterValidationConfig:
    """Optional constraints for validating one raster input.

    Grid comparisons use an absolute and relative tolerance of ``1e-9`` by
    default. The absolute component handles values near zero, while the
    relative component handles large projected coordinates without treating
    sub-millimetre floating-point noise as a grid shift.
    """

    exact_band_count: int | None = None
    min_band_count: int | None = None
    max_band_count: int | None = None
    allowed_dtypes: tuple[str, ...] | None = None
    require_crs: bool = True
    require_transform: bool = True
    require_resolution: bool = True
    require_nodata: bool = False
    require_finite: bool = True
    grid_absolute_tolerance: float = 1e-9
    grid_relative_tolerance: float = 1e-9
    nodata_policy: NodataPolicy = "optional"

    def __post_init__(self) -> None:
        counts = (self.exact_band_count, self.min_band_count, self.max_band_count)
        if any(count is not None and count < 1 for count in counts):
            raise ValueError("Band-count constraints must be positive.")
        if self.exact_band_count is not None and (
            self.exact_band_count < (self.min_band_count or 1)
            or self.exact_band_count > (self.max_band_count or self.exact_band_count)
        ):
            raise ValueError("exact_band_count must satisfy the configured minimum and maximum.")
        if self.min_band_count is not None and self.max_band_count is not None:
            if self.min_band_count > self.max_band_count:
                raise ValueError("min_band_count cannot exceed max_band_count.")
        if self.grid_absolute_tolerance < 0.0 or not np.isfinite(self.grid_absolute_tolerance):
            raise ValueError("grid_absolute_tolerance must be finite and non-negative.")
        if self.grid_relative_tolerance < 0.0 or not np.isfinite(self.grid_relative_tolerance):
            raise ValueError("grid_relative_tolerance must be finite and non-negative.")
        if self.nodata_policy not in {"optional", "numeric", "nan", "none"}:
            raise ValueError("nodata_policy must be 'optional', 'numeric', 'nan', or 'none'.")


@dataclass(frozen=True, slots=True)
class RasterValidationResult:
    """Structured validation result for one raster input."""

    valid: bool
    issues: tuple[ValidationIssue, ...]
    metadata: RasterMetadata | None = None
    nodata_kind: NodataKind | None = None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return only findings that make the input invalid."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return non-fatal validation findings."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def as_dict(self) -> dict[str, object]:
        """Return the result in a serializable form."""

        return {
            "valid": self.valid,
            "issues": [issue.as_dict() for issue in self.issues],
            "metadata": self.metadata.as_dict() if self.metadata is not None else None,
            "nodata_kind": self.nodata_kind,
        }


@dataclass(frozen=True, slots=True)
class PairValidationResult:
    """Structured validation result for two raster inputs."""

    valid: bool
    issues: tuple[ValidationIssue, ...]
    first: RasterValidationResult
    second: RasterValidationResult
    spatial_overlap: bool | None
    can_transform_to_common_grid: bool | None
    exact_common_grid: bool | None = None
    grid_mismatches: tuple[str, ...] = ()
    dtype_match: bool | None = None
    nodata_match: bool | None = None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return all pair and member errors."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return all pair and member warnings."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def as_dict(self) -> dict[str, object]:
        """Return the result in a serializable form."""

        return {
            "valid": self.valid,
            "issues": [issue.as_dict() for issue in self.issues],
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "spatial_overlap": self.spatial_overlap,
            "can_transform_to_common_grid": self.can_transform_to_common_grid,
            "exact_common_grid": self.exact_common_grid,
            "grid_mismatches": list(self.grid_mismatches),
            "dtype_match": self.dtype_match,
            "nodata_match": self.nodata_match,
        }


@dataclass(frozen=True, slots=True)
class RasterPair:
    """Two spatially aligned rasters with a computed common valid mask."""

    first: GeoRaster
    second: GeoRaster

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the shared spatial shape as ``(height, width)``."""

        return int(self.first.data.shape[1]), int(self.first.data.shape[2])

    @property
    def common_valid_mask(self) -> np.ndarray:
        """Return pixels valid in both rasters."""

        return self.first.valid_mask & self.second.valid_mask


@dataclass(frozen=True, slots=True)
class AlignmentDiagnostics:
    """Audit information describing one raster-to-grid alignment."""

    source_crs: str | None
    reference_crs: str | None
    source_resolution: tuple[float, float]
    reference_resolution: tuple[float, float]
    source_dimensions: tuple[int, int]
    reference_dimensions: tuple[int, int]
    reprojected: bool
    resampled: bool
    resampling_method: str
    grid_comparison: dict[str, bool]
    spatial_overlap: bool
    overlap_bounds_in_reference: Bounds
    source_nodata: float | None
    destination_nodata: float
    nodata_policy: AlignmentNodataPolicy
    source_dtype: str
    aligned_dtype: str

    def as_dict(self) -> dict[str, object]:
        """Return diagnostics in a serializable form."""

        return {
            "source_crs": self.source_crs,
            "reference_crs": self.reference_crs,
            "source_resolution": list(self.source_resolution),
            "reference_resolution": list(self.reference_resolution),
            "source_dimensions": list(self.source_dimensions),
            "reference_dimensions": list(self.reference_dimensions),
            "reprojected": self.reprojected,
            "resampled": self.resampled,
            "resampling_method": self.resampling_method,
            "grid_comparison": dict(self.grid_comparison),
            "spatial_overlap": self.spatial_overlap,
            "overlap_bounds_in_reference": list(self.overlap_bounds_in_reference),
            "source_nodata": self.source_nodata,
            "destination_nodata": self.destination_nodata,
            "nodata_policy": self.nodata_policy,
            "source_dtype": self.source_dtype,
            "aligned_dtype": self.aligned_dtype,
        }


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Source data aligned to an exact reference grid.

    ``output_metadata`` describes the returned array. ``reference_metadata``
    is retained separately so consumers can inspect the unmodified reference
    raster metadata.
    """

    aligned_array: np.ndarray
    aligned_valid_mask: np.ndarray
    output_metadata: RasterMetadata
    reference_metadata: RasterMetadata
    source_metadata: RasterMetadata
    alignment_diagnostics: AlignmentDiagnostics

    @property
    def data(self) -> np.ndarray:
        """Compatibility view of ``aligned_array``."""

        return self.aligned_array

    @property
    def metadata(self) -> RasterMetadata:
        """Compatibility view of metadata describing the aligned output."""

        return self.output_metadata

    @property
    def valid_mask(self) -> np.ndarray:
        """Compatibility view of the aligned valid-pixel mask."""

        return self.aligned_valid_mask

    def as_geo_raster(self) -> GeoRaster:
        """Return the aligned output using the standard ``GeoRaster`` contract."""

        return GeoRaster(
            data=self.aligned_array,
            metadata=self.output_metadata,
            valid_mask=self.aligned_valid_mask,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the alignment result metadata and diagnostics."""

        return {
            "aligned_array": self.aligned_array,
            "output_metadata": self.output_metadata.as_dict(),
            "reference_metadata": self.reference_metadata.as_dict(),
            "source_metadata": self.source_metadata.as_dict(),
            "alignment_diagnostics": self.alignment_diagnostics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for deterministic, model-agnostic raster preparation.

    The default mode preserves source values as ``float32`` while filling
    invalid pixels with zero. Normalization is opt-in because optical and SAR
    products use different physical units.
    """

    normalization: NormalizationMode = "none"
    percentile_clip: tuple[float, float] | None = None
    means: tuple[float, ...] | None = None
    stds: tuple[float, ...] | None = None
    invalid_fill: float = 0.0

    def __post_init__(self) -> None:
        if self.normalization not in {"none", "minmax", "zscore"}:
            raise ValueError("normalization must be 'none', 'minmax', or 'zscore'.")
        if self.percentile_clip is not None:
            low, high = self.percentile_clip
            if not 0.0 <= low < high <= 100.0:
                raise ValueError("percentile_clip must satisfy 0 <= low < high <= 100.")
        if (self.means is None) != (self.stds is None):
            raise ValueError("means and stds must be supplied together.")
        if self.means is not None and self.normalization != "zscore":
            raise ValueError("means and stds require normalization='zscore'.")
        if self.stds is not None and any(std <= 0.0 for std in self.stds):
            raise ValueError("All standard deviations must be positive.")
        if not np.isfinite(self.invalid_fill):
            raise ValueError("invalid_fill must be finite.")


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Explicit policy for reprojecting a raster onto a reference grid."""

    resampling: Resampling = Resampling.nearest
    destination_nodata: float = 0.0
    tolerance: float = 1e-9
    relative_tolerance: float = 1e-9
    nodata_policy: AlignmentNodataPolicy = "respect_source"

    def __post_init__(self) -> None:
        if not np.isfinite(self.destination_nodata):
            raise ValueError("destination_nodata must be finite.")
        if self.tolerance < 0.0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and non-negative.")
        if self.relative_tolerance < 0.0 or not np.isfinite(self.relative_tolerance):
            raise ValueError("relative_tolerance must be finite and non-negative.")
        if self.nodata_policy not in {"respect_source", "mask_only"}:
            raise ValueError("nodata_policy must be 'respect_source' or 'mask_only'.")
