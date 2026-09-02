"""Small public API for SatQuery AI geospatial input processing."""

from .coregistration import align_raster_to_reference, grids_match, validate_coregistration
from .exceptions import GeoAlignmentError, GeoError, GeoFormatError, GeoReadError, GeoValidationError
from .metadata import read_metadata
from .pairs import read_bitemporal_pair, read_coregistered_pair, read_optical_sar_pair
from .pipeline import prepare_bitemporal_pair, prepare_optical_sar_pair, prepare_single_raster
from .preprocessing import preprocess_raster
from .reader import read_geotiff, read_raster
from .types import (
    AlignmentConfig,
    AlignmentDiagnostics,
    AlignmentResult,
    GeoRaster,
    PairValidationResult,
    PreprocessingConfig,
    RasterMetadata,
    RasterPair,
    RasterValidationConfig,
    RasterValidationResult,
    ValidationIssue,
)
from .validation import (
    normalize_band_indexes,
    validate_array,
    validate_bitemporal_pair,
    validate_geo_raster,
    validate_optical_sar_pair,
    validate_raster,
    validate_raster_path,
)

__all__ = [
    "AlignmentConfig",
    "AlignmentDiagnostics",
    "AlignmentResult",
    "GeoAlignmentError",
    "GeoError",
    "GeoFormatError",
    "GeoRaster",
    "GeoReadError",
    "GeoValidationError",
    "PairValidationResult",
    "PreprocessingConfig",
    "RasterMetadata",
    "RasterPair",
    "RasterValidationConfig",
    "RasterValidationResult",
    "ValidationIssue",
    "align_raster_to_reference",
    "grids_match",
    "normalize_band_indexes",
    "prepare_bitemporal_pair",
    "prepare_optical_sar_pair",
    "prepare_single_raster",
    "preprocess_raster",
    "read_bitemporal_pair",
    "read_coregistered_pair",
    "read_geotiff",
    "read_metadata",
    "read_optical_sar_pair",
    "read_raster",
    "validate_array",
    "validate_bitemporal_pair",
    "validate_coregistration",
    "validate_geo_raster",
    "validate_optical_sar_pair",
    "validate_raster",
    "validate_raster_path",
]
