"""Shared temporary GeoTIFF fixtures for Geo Expert tests."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin


@pytest.fixture
def write_raster():
    """Return a helper for writing small test GeoTIFFs."""

    def _write(
        path: Path,
        values: np.ndarray,
        *,
        transform: Affine | None = None,
        crs: str | None = "EPSG:32643",
        nodata: float | None = -9999.0,
    ) -> Path:
        profile = {
            "driver": "GTiff",
            "height": values.shape[1],
            "width": values.shape[2],
            "count": values.shape[0],
            "dtype": str(values.dtype),
            "transform": transform or from_origin(500000, 4100000, 10, 10),
        }
        if crs is not None:
            profile["crs"] = crs
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(values)
        return path

    return _write
