# Geo Expert API

The `geo` package is the boundary between geospatial files and downstream
model code. It contains no model, agent, router, or UI logic.

## Core contract

`read_raster(path, bands=None)` returns a `GeoRaster`:

- `data`: NumPy data in the file's native dtype, always shaped `(C, H, W)`.
  Pass the keyword-only `dtype=` argument to explicitly request conversion.
- `valid_mask`: boolean NumPy array shaped `(H, W)`. A pixel is valid only if
  all selected bands are unmasked, non-nodata, and finite.
- `metadata`: `RasterMetadata` containing CRS, affine transform, bounds,
  dimensions, resolution, loaded dtype, nodata, source path, selected 1-based
  bands, descriptions, scales, offsets, units, and driver. `native_dtype`
  retains the file storage dtype when explicit conversion was requested.

Only `.tif` and `.tiff` paths are accepted. Missing paths raise
`GeoValidationError`, unsupported extensions raise `GeoFormatError`, and
Rasterio open/read failures raise `GeoReadError`.

Invalid source values are intentionally not overwritten by the reader.
`preprocess_raster` fills them before model consumption.

## Pair APIs

- `read_optical_sar_pair(...)` requires exact CRS, transform, bounds, and
  spatial dimensions. Optical and SAR channel counts may differ.
- `read_bitemporal_pair(...)` requires the same grid and channel count by
  default. Pass an explicit `AlignmentConfig` to reproject the second image
  onto the first image's grid.
- `RasterPair.common_valid_mask` is the intersection of both masks.

No pair API silently resamples data.

## Validation

`validate_raster(source, config=None)` returns a `RasterValidationResult`.
`validate_optical_sar_pair(...)` and `validate_bitemporal_pair(...)` return a
`PairValidationResult`. Results expose `valid`, structured `issues`, parsed
metadata, `spatial_overlap`, `exact_common_grid`, and whether Rasterio can
calculate a transformation between the second raster and the first raster's
CRS/grid systems.

Validation reports missing CRS, unusable transforms or resolutions, invalid
bounds or dtypes, nodata availability, NaN/infinite values, configured band
count violations, pair grid mismatches, and non-overlap. It never resamples or
otherwise changes its inputs; use `align_raster_to_reference` for explicit
correction.

Pair grid comparisons use one shared internal comparator for CRS, dimensions,
resolution, affine transform, and bounds. `RasterValidationConfig` defaults to
an absolute and relative tolerance of `1e-9`; the absolute component handles
values near zero and the relative component limits floating-point noise for
large projected coordinates. Pair comparisons use the stricter tolerance when
the two member configurations differ.

The default `nodata_policy="optional"` accepts numeric nodata, NaN nodata, or
no nodata metadata. Missing nodata is reported as a warning unless
`require_nodata=True`; use `nodata_policy="numeric"`, `"nan"`, or `"none"` to
enforce a specific policy. Bi-temporal dtype and nodata differences are
reported as diagnostics and do not invalidate an otherwise compatible pair.

## Preprocessing

`PreprocessingConfig` supports:

- `normalization="none"` (default), `"minmax"`, or `"zscore"`;
- optional per-channel percentile clipping;
- fixed per-channel means and standard deviations for reproducible inference;
- a configurable fill value for invalid pixels.

`prepare_single_raster`, `prepare_optical_sar_pair`, and
`prepare_bitemporal_pair` compose reading, validation, alignment, and
preprocessing for downstream consumers.

## Co-registration

`align_raster_to_reference(source, reference, config=None)` transforms the
source raster onto the exact CRS, affine transform, width, and height of the
reference raster. The reference defines the output grid; source band count and
source data dtype are preserved. Rasterio's geospatial `reproject` operation
performs the transformation—array resizing is never used.

The default `AlignmentConfig` uses nearest-neighbour resampling,
`destination_nodata=0.0`, and `nodata_policy="respect_source"`. Set
`resampling` explicitly for continuous data or another application policy.
Source nodata and the source valid mask are respected. For partial overlap,
reference pixels outside source coverage receive the configured destination
nodata value and are marked false in `aligned_valid_mask`. Completely
non-overlapping rasters fail before any reprojection.

The returned `AlignmentResult` contains:

- `aligned_array`: source bands as `(C, H, W)` on the reference grid;
- `aligned_valid_mask`: boolean `(H, W)` mask;
- `output_metadata`: metadata internally consistent with the aligned array,
  including its channel count, dtype, and configured destination nodata;
- `reference_metadata`: the unmodified reference metadata, including the exact
  destination CRS, transform, dimensions, resolution, and bounds;
- `source_metadata`: original source metadata;
- `alignment_diagnostics`: CRS, resolution, dimensions, grid comparison,
  overlap, resampling, nodata, dtype, and whether reprojection occurred.

An already matching grid uses a copy-based no-op fast path. The result still
contains diagnostics showing `reprojected=False` and `resampled=False`.
For direct `GeoRaster` inputs, `valid_mask` is authoritative. With
`nodata_policy="respect_source"`, metadata nodata values are additionally
excluded from the effective source mask. Missing CRS or usable affine
transforms, invalid or untransformable bounds, non-overlap, unrepresentable
destination nodata values, and Rasterio failures raise `GeoAlignmentError`;
no CRS is guessed and input files are never modified.

Example:

```python
from geo import AlignmentConfig, align_raster_to_reference

result = align_raster_to_reference(
    source_path,
    reference_path,
    AlignmentConfig(destination_nodata=-9999.0),
)
array = result.aligned_array
mapping = result.reference_metadata
```
