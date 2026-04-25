"""
data_fetch.py
-------------
Handles all communication with the Copernicus Data Space via the openEO API.
Fetches Sentinel-2 L2A data for a given bounding box and time range.

Notes on raster output
----------------------
openEO's ``download(format="GTiff")`` is supposed to return a multi-band
GeoTIFF when the source datacube still has a "bands" dimension, with one
GeoTIFF band per spectral band (B03, B04, B05, B08).  In practice — depending
on backend version, post-processing graph, or temporal reduction — the result
may collapse to a single-band GeoTIFF, which used to trigger:

    rasterio.errors.RasterioIOError: band index 2 out of range (not in (1,))

This module guards against that:

1. ``_parse_geotiff`` validates the band count, resolves bands by their
   GeoTIFF description (the band name set by openEO) when available, and
   raises a clear error if fewer bands are present than expected.
2. If the multi-band download yields fewer bands than required, we fall back
   to issuing one ``download`` per band (each output is a single-band GeoTIFF
   read at index 1, which is always valid).
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import openeo

logger = logging.getLogger(__name__)

# Copernicus Data Space openEO endpoint
OPENEO_URL = "https://openeo.dataspace.copernicus.eu"

# Sentinel-2 L2A collection ID
COLLECTION = "SENTINEL2_L2A"

# Bands needed for NDWI and NDCI
# B03 = Green, B04 = Red, B05 = Red Edge, B08 = NIR
REQUIRED_BANDS = ["B03", "B04", "B05", "B08"]

# Bounding box offset in degrees (small area around the clicked point)
BBOX_OFFSET = 0.01  # ~1km radius


def get_bounding_box(lat: float, lon: float, offset: float = BBOX_OFFSET) -> Dict[str, float]:
    """
    Build a small bounding box around the given coordinates.

    Returns a dict in the standard openEO ``spatial_extent`` format. Coordinates
    are interpreted in EPSG:4326 (WGS84 lat/lon) by openEO when no ``crs`` key
    is supplied; we make this explicit to make downstream behaviour
    predictable regardless of backend defaults.
    """
    return {
        "west": lon - offset,
        "south": lat - offset,
        "east": lon + offset,
        "north": lat + offset,
        # Explicit CRS prevents any ambiguity if the openEO backend default
        # ever changes — Sentinel-2 is reprojected to UTM internally, but the
        # bbox we send is lat/lon.
        "crs": "EPSG:4326",
    }


def get_time_range(days_back: int = 10) -> Tuple[str, str]:
    """Compute (start_date, end_date) ISO strings covering the last N days."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def connect_to_openeo() -> openeo.Connection:
    """
    Authenticate and connect to the Copernicus openEO backend.

    Uses cached OIDC credentials when available, otherwise triggers an
    interactive device-code login (only on first run).
    """
    try:
        conn = openeo.connect(OPENEO_URL)
        conn.authenticate_oidc()
        logger.info("Connected to Copernicus openEO backend successfully.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to openEO: {e}")
        raise RuntimeError(f"openEO connection failed: {e}")


def _build_normalized_cube(
    conn: openeo.Connection,
    bbox: Dict[str, float],
    start_date: str,
    end_date: str,
):
    """
    Build the cloud-masked, time-composited, normalized Sentinel-2 datacube.
    The resulting cube still has a "bands" dimension covering REQUIRED_BANDS.
    """
    datacube = conn.load_collection(
        COLLECTION,
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=REQUIRED_BANDS + ["SCL"],
        max_cloud_cover=80,
    )

    # --- Cloud masking using SCL band ---
    # SCL valid classes: 4=Vegetation, 5=Not-vegetated, 6=Water, 7=Unclassified
    scl = datacube.band("SCL")
    valid_mask = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)
    datacube_masked = datacube.mask(~valid_mask).filter_bands(REQUIRED_BANDS)

    # Median composite over time → one representative image per pixel
    composite = datacube_masked.reduce_dimension(dimension="t", reducer="median")

    # Sentinel-2 DN (0–10000) → reflectance (0.0–1.0)
    return composite / 10000.0


def fetch_sentinel2_bands(lat: float, lon: float) -> Dict[str, Any]:
    """
    Core data pipeline:
    1. Connect to openEO
    2. Build a cloud-masked median composite over the last few days for the
       bounding box around (lat, lon)
    3. Download as GeoTIFF and parse into per-band numpy arrays
    4. If the multi-band download yields fewer bands than expected, fall back
       to per-band downloads (each is a guaranteed-single-band GeoTIFF).
    """
    conn = connect_to_openeo()

    bbox = get_bounding_box(lat, lon)
    start_date, end_date = get_time_range(days_back=10)

    logger.info(
        f"Fetching Sentinel-2 data for bbox={bbox}, time={start_date}/{end_date}"
    )

    normalized = _build_normalized_cube(conn, bbox, start_date, end_date)

    bands: Dict[str, np.ndarray]
    try:
        result_bytes = normalized.download(format="GTiff")
        bands = _parse_geotiff(result_bytes, REQUIRED_BANDS)
    except _InsufficientBandsError as exc:
        logger.warning(
            f"Multi-band GeoTIFF download did not return all required bands "
            f"({exc}); falling back to per-band downloads."
        )
        bands = _download_bands_individually(normalized, REQUIRED_BANDS)

    return {
        "bands": bands,
        "bbox": bbox,
        "time_range": (start_date, end_date),
        "timestamp": datetime.utcnow().isoformat(),
    }


class _InsufficientBandsError(RuntimeError):
    """Raised when a downloaded GeoTIFF has fewer bands than expected."""


def _parse_geotiff(geotiff_bytes: bytes, band_names: list) -> Dict[str, np.ndarray]:
    """
    Parse the downloaded GeoTIFF bytes into individual numpy arrays per band.

    Resolution strategy:

    1. Inspect ``src.count`` (number of raster bands) and ``src.descriptions``
       (per-band metadata; openEO sets these to the openEO band names like
       "B03", "B04", ...).
    2. If a band's name appears in the descriptions, read it from that
       index — this is robust to band reordering by the backend.
    3. Otherwise fall back to positional order (band 1 → first name, etc.).
    4. If the GeoTIFF contains fewer bands than expected, raise
       ``_InsufficientBandsError`` so the caller can fall back to per-band
       downloads instead of failing with the cryptic
       "band index 2 out of range (not in (1,))" rasterio error.
    """
    import rasterio

    with rasterio.open(io.BytesIO(geotiff_bytes)) as src:
        band_count = src.count
        descriptions = list(src.descriptions or [])
        logger.info(
            f"Parsed GeoTIFF: bands={band_count}, descriptions={descriptions}, "
            f"crs={src.crs}, shape=({src.height}, {src.width})"
        )

        if band_count == 0:
            raise _InsufficientBandsError("downloaded GeoTIFF contains no bands")

        if band_count < len(band_names):
            raise _InsufficientBandsError(
                f"downloaded GeoTIFF has only {band_count} band(s) "
                f"but {len(band_names)} ({band_names}) are required"
            )

        # Map "B03" / "B04" / ... → 1-based band index via descriptions
        name_to_idx: Dict[str, int] = {}
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                name_to_idx[desc] = i

        nodata = src.nodata
        bands: Dict[str, np.ndarray] = {}
        for i, name in enumerate(band_names, start=1):
            idx = name_to_idx.get(name, i)
            if idx > band_count:
                # Should never happen given the band_count check above, but
                # guards against a malformed descriptions array.
                raise _InsufficientBandsError(
                    f"band '{name}' resolved to index {idx} which is out of "
                    f"range (GeoTIFF has {band_count} band(s))"
                )
            arr = src.read(idx).astype(np.float32)
            if nodata is not None:
                arr[arr == nodata] = np.nan
            bands[name] = arr

    return bands


def _read_first_band(geotiff_bytes: bytes) -> np.ndarray:
    """
    Read band index 1 from a (presumed) single-band GeoTIFF.

    Used by the per-band fallback path: if a multi-band download came back
    with fewer bands than expected, we re-issue one ``download`` per band
    and each of those is guaranteed to have band index 1 available.
    """
    import rasterio

    with rasterio.open(io.BytesIO(geotiff_bytes)) as src:
        if src.count < 1:
            raise RuntimeError("GeoTIFF contains no bands")
        if src.count > 1:
            logger.info(
                f"Per-band GeoTIFF unexpectedly has {src.count} bands "
                f"(descriptions={src.descriptions}); using band 1"
            )
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr


def _download_bands_individually(normalized_cube, band_names: list) -> Dict[str, np.ndarray]:
    """
    Fallback path: download each band as its own single-band GeoTIFF.

    Each download is an independent openEO synchronous job; we run them in
    parallel via a small thread pool to keep the total wall time close to a
    single multi-band download.
    """

    def _fetch_one(name: str) -> Tuple[str, np.ndarray]:
        single = normalized_cube.filter_bands([name])
        logger.info(f"Per-band download starting for {name}")
        blob = single.download(format="GTiff")
        return name, _read_first_band(blob)

    bands: Dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=len(band_names)) as pool:
        futures = {pool.submit(_fetch_one, name): name for name in band_names}
        for future in as_completed(futures):
            name, arr = future.result()
            bands[name] = arr
            logger.info(f"Per-band download completed for {name}: shape={arr.shape}")

    missing = [n for n in band_names if n not in bands]
    if missing:
        raise RuntimeError(
            f"Per-band fallback failed to retrieve: {missing}. "
            "Sentinel-2 may have no usable scenes for this location/time window."
        )
    return bands
