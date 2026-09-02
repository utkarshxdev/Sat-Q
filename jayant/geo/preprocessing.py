"""Model-agnostic preparation of validated raster values."""

from __future__ import annotations

import numpy as np

from .exceptions import GeoValidationError
from .types import GeoRaster, PreprocessingConfig
from .validation import validate_geo_raster


def _band_statistics(values: np.ndarray, config: PreprocessingConfig) -> tuple[float, float]:
    """Compute normalization statistics for one valid band."""

    if config.means is not None and config.stds is not None:
        raise AssertionError("Fixed statistics are selected by the caller.")
    if config.normalization == "minmax":
        return float(values.min()), float(values.max())
    return float(values.mean()), float(values.std())


def preprocess_raster(
    raster: GeoRaster,
    config: PreprocessingConfig | None = None,
) -> GeoRaster:
    """Return a float32 raster with invalid pixels filled and values prepared.

    Normalization is per-channel and uses only valid finite pixels. Fixed
    ``means`` and ``stds`` can be supplied for inference-time consistency.
    Geospatial metadata and the valid-pixel mask are preserved unchanged.
    """

    validate_geo_raster(raster)
    policy = config or PreprocessingConfig()
    if policy.means is not None and len(policy.means) != raster.channels:
        raise GeoValidationError("means length must equal the raster channel count.")
    if policy.stds is not None and len(policy.stds) != raster.channels:
        raise GeoValidationError("stds length must equal the raster channel count.")

    prepared = np.full(raster.data.shape, policy.invalid_fill, dtype=np.float32)
    for channel in range(raster.channels):
        valid = raster.valid_mask & np.isfinite(raster.data[channel])
        if not valid.any():
            continue
        values = raster.data[channel, valid].astype(np.float32, copy=False)
        if policy.percentile_clip is not None:
            low, high = np.percentile(values, policy.percentile_clip)
            clipped = np.clip(raster.data[channel, valid], low, high)
            prepared[channel, valid] = clipped
            values = prepared[channel, valid]
        else:
            prepared[channel, valid] = raster.data[channel, valid]

        if policy.normalization != "none":
            if policy.means is not None and policy.stds is not None:
                center = policy.means[channel]
                spread = policy.stds[channel]
            else:
                center, spread = _band_statistics(values, policy)
            if policy.normalization == "minmax":
                denominator = spread - center
                prepared[channel, valid] = (
                    0.0 if denominator == 0.0 else (prepared[channel, valid] - center) / denominator
                )
            else:
                prepared[channel, valid] = (
                    0.0 if spread == 0.0 else (prepared[channel, valid] - center) / spread
                )
        prepared[channel, ~valid] = policy.invalid_fill

    return GeoRaster(data=prepared, metadata=raster.metadata, valid_mask=raster.valid_mask.copy())
