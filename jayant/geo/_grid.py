"""Private shared helpers for comparing raster grid metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import RasterMetadata


@dataclass(frozen=True, slots=True)
class GridComparison:
    """Field-by-field comparison of two raster grids."""

    crs_match: bool
    dimensions_match: bool
    resolution_match: bool
    transform_match: bool
    bounds_match: bool

    @property
    def exact_common_grid(self) -> bool:
        """Return whether all grid-defining fields match within tolerance."""

        return all(
            (
                self.crs_match,
                self.dimensions_match,
                self.resolution_match,
                self.transform_match,
                self.bounds_match,
            ),
        )

    @property
    def mismatches(self) -> tuple[str, ...]:
        """Return human-readable names of grid fields that differ."""

        fields = (
            ("CRS", self.crs_match),
            ("dimensions", self.dimensions_match),
            ("resolution", self.resolution_match),
            ("affine transform", self.transform_match),
            ("bounds", self.bounds_match),
        )
        return tuple(name for name, matches in fields if not matches)

    def as_dict(self) -> dict[str, bool]:
        """Return field-by-field comparison flags for diagnostics."""

        return {
            "crs": self.crs_match,
            "dimensions": self.dimensions_match,
            "resolution": self.resolution_match,
            "affine_transform": self.transform_match,
            "bounds": self.bounds_match,
            "exact_common_grid": self.exact_common_grid,
        }


def _numeric_values_match(
    first: tuple[float, ...],
    second: tuple[float, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    """Compare numeric metadata safely with absolute and relative tolerances."""

    try:
        return bool(
            np.allclose(
                first,
                second,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                equal_nan=False,
            ),
        )
    except (TypeError, ValueError):
        return False


def compare_grids(
    first: RasterMetadata,
    second: RasterMetadata,
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> GridComparison:
    """Compare CRS, dimensions, resolution, transform, and bounds.

    Dimensions and CRS use exact equality. Numeric fields use
    ``absolute_tolerance + relative_tolerance * abs(reference)`` semantics,
    matching NumPy's ``allclose`` behavior. This helper performs no alignment
    or data mutation and is intentionally private so validation and future
    co-registration code can share one comparison policy.
    """

    if absolute_tolerance < 0.0 or not np.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    if relative_tolerance < 0.0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and non-negative.")
    return GridComparison(
        crs_match=first.crs == second.crs,
        dimensions_match=(first.width, first.height) == (second.width, second.height),
        resolution_match=_numeric_values_match(
            first.resolution,
            second.resolution,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
        transform_match=_numeric_values_match(
            tuple(first.transform),
            tuple(second.transform),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
        bounds_match=_numeric_values_match(
            first.bounds,
            second.bounds,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
    )


def bounds_are_valid(bounds: tuple[float, float, float, float]) -> bool:
    """Return whether bounds are finite and define positive area."""

    return bool(
        len(bounds) == 4
        and np.isfinite(bounds).all()
        and bounds[0] < bounds[2]
        and bounds[1] < bounds[3],
    )


def bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two axis-aligned bounds have positive-area overlap."""

    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )
