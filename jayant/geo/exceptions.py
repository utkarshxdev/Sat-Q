"""Exceptions raised by the Geo Expert input-processing package."""


class GeoError(Exception):
    """Base class for all package-specific errors."""


class GeoReadError(GeoError, OSError):
    """Raised when a raster cannot be opened or read."""


class GeoValidationError(GeoError, ValueError):
    """Raised when a path, raster, array, or configuration is invalid."""


class GeoFormatError(GeoValidationError):
    """Raised when a path does not use a supported GeoTIFF extension."""


class GeoAlignmentError(GeoError, ValueError):
    """Raised when rasters do not share a required spatial grid."""
