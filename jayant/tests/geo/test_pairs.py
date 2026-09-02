"""Tests for optical/SAR and bi-temporal pair contracts."""

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from geo import AlignmentConfig, GeoAlignmentError, read_bitemporal_pair, read_optical_sar_pair


def test_optical_sar_pair_keeps_channel_counts_and_intersects_masks(tmp_path: Path, write_raster) -> None:
    transform = from_origin(500000, 4100000, 10, 10)
    optical = np.ones((3, 4, 4), dtype=np.float32)
    sar = np.full((2, 4, 4), 2.0, dtype=np.float32)
    optical[:, 0, 0] = -9999.0
    optical_path = write_raster(tmp_path / "optical.tif", optical, transform=transform)
    sar_path = write_raster(tmp_path / "sar.tif", sar, transform=transform)

    pair = read_optical_sar_pair(optical_path, sar_path)

    assert pair.first.data.shape == (3, 4, 4)
    assert pair.second.data.shape == (2, 4, 4)
    assert not pair.common_valid_mask[0, 0]
    assert pair.common_valid_mask[1, 1]


def test_bitemporal_pair_rejects_different_grids_without_alignment(tmp_path: Path, write_raster) -> None:
    values = np.ones((2, 4, 4), dtype=np.float32)
    first = write_raster(tmp_path / "t1.tif", values, transform=from_origin(0, 40, 10, 10))
    second = write_raster(tmp_path / "t2.tif", values, transform=from_origin(10, 40, 10, 10))

    with pytest.raises(GeoAlignmentError):
        read_bitemporal_pair(first, second)


def test_bitemporal_pair_can_align_explicitly(tmp_path: Path, write_raster) -> None:
    first_values = np.zeros((2, 4, 4), dtype=np.float32)
    second_values = np.ones((2, 2, 2), dtype=np.float32)
    first = write_raster(tmp_path / "t1.tif", first_values, transform=from_origin(0, 40, 10, 10))
    second = write_raster(tmp_path / "t2.tif", second_values, transform=from_origin(0, 40, 20, 20))

    pair = read_bitemporal_pair(first, second, alignment=AlignmentConfig())

    assert pair.first.data.shape == pair.second.data.shape == (2, 4, 4)
    assert pair.second.metadata.transform == pair.first.metadata.transform
