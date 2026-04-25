"""
data_fetch.py
-------------
Handles all communication with the Copernicus Data Space via the openEO API.
Fetches Sentinel-2 L2A data for a given bounding box and time range.

ROOT CAUSE FIX (422 error):
  The SCL mask kept only classes 4,5,6,7. In spring/cloudy conditions over
  Macedonia this wipes ALL pixels → all NaN → ValueError in processing → 422.

  Fix strategy (3 layers):
  1. PRIMARY:    Loosen SCL mask to also keep class 3 (cloud shadow) and 8
                 (medium cloud probability). Covers most real-world cases.
  2. FALLBACK:   If >80% pixels are still NaN, retry with NO SCL mask at all
                 (raw median composite). Slightly noisier but always has data.
  3. WIDER TIME: If 15-day window is empty, automatically widen to 45 days.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import numpy as np
import rasterio
import openeo

logger = logging.getLogger(__name__)

OPENEO_URL     = "https://openeo.dataspace.copernicus.eu"
COLLECTION     = "SENTINEL2_L2A"
REQUIRED_BANDS = ["B02", "B03", "B04", "B05", "B08"]
BBOX_OFFSET    = 0.01   # ~1 km radius in degrees

# If more than this fraction of pixels are NaN, treat as "no usable data"
NAN_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_bounding_box(lat: float, lon: float, offset: float = BBOX_OFFSET) -> Dict:
    return {
        "west":  lon - offset,
        "south": lat - offset,
        "east":  lon + offset,
        "north": lat + offset,
        "crs":   "EPSG:4326",
    }


def get_time_range(days_back: int) -> Tuple[str, str]:
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def connect_to_openeo() -> openeo.Connection:
    try:
        conn = openeo.connect(OPENEO_URL)
        conn.authenticate_oidc()
        logger.info("Connected to Copernicus openEO backend successfully.")
        return conn
    except Exception as e:
        logger.error(f"openEO connection failed: {e}")
        raise RuntimeError(f"openEO connection failed: {e}")


# ---------------------------------------------------------------------------
# Core: build one process graph per band
# ---------------------------------------------------------------------------

def _build_composite(
    conn: openeo.Connection,
    bbox: Dict,
    start_date: str,
    end_date: str,
    band: str,
    use_scl_mask: bool = True,
) -> openeo.DataCube:
    """
    Build an openEO process graph for ONE band.

    use_scl_mask=True  -> SCL cloud filtering (preferred, cleaner data)
    use_scl_mask=False -> No cloud filtering (fallback when mask kills all pixels)

    Downloading one band at a time avoids the multi-band GeoTIFF ordering bug
    where openEO returns only the first band regardless of how many were requested.
    """
    bands_to_load = [band, "SCL"] if use_scl_mask else [band]

    cube = conn.load_collection(
        COLLECTION,
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=bands_to_load,
        max_cloud_cover=90,  # loose pre-filter; SCL mask does the real work
    )

    if use_scl_mask:
        scl = cube.band("SCL")

        # FIX vs original: include MORE SCL classes.
        # Original kept only 4,5,6,7 which is too aggressive for cloudy spring days.
        # Now also keep:
        #   3 = cloud shadow  (darker but valid spectral signal over water)
        #   8 = medium cloud probability  (border pixels, often perfectly usable)
        # Still rejecting: 0=no-data, 1=saturated, 9=high-cloud, 10=cirrus, 11=snow
        valid_mask = (
            (scl == 3) |   # cloud shadow
            (scl == 4) |   # vegetation
            (scl == 5) |   # not vegetated
            (scl == 6) |   # water  <- most important class for this app
            (scl == 7) |   # unclassified
            (scl == 8)     # medium cloud probability
        )
        cube = cube.filter_bands([band]).mask(~valid_mask)

    # Temporal median composite → cloud-reduced single image per pixel
    composite = cube.reduce_dimension(dimension="t", reducer="median")

    # Normalise DN (0-10000) → reflectance (0.0-1.0)
    return composite / 10000.0


def _download_to_array(datacube: openeo.DataCube, band_name: str) -> np.ndarray:
    """Download a single-band DataCube and return a 2D float32 numpy array."""
    logger.info(f"Per-band download starting for {band_name}...")
    raw = datacube.download(format="GTiff")

    with rasterio.open(io.BytesIO(raw)) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    # Sentinel-2 L2A uses integer 0 as the nodata sentinel.
    # After dividing by 10000, nodata pixels appear as exactly 0.0.
    # Replace with NaN (genuine reflectance of 0.0 is physically impossible).
    arr[arr == 0.0] = np.nan

    nan_frac = float(np.mean(np.isnan(arr)))
    logger.info(
        f"Per-band download completed for {band_name}: "
        f"shape={arr.shape}, NaN={nan_frac:.1%}"
    )
    return arr


def _nan_fraction(bands: Dict[str, np.ndarray]) -> float:
    """Return the average NaN fraction across all downloaded band arrays."""
    fracs = [np.mean(np.isnan(arr)) for arr in bands.values()]
    return float(np.mean(fracs)) if fracs else 1.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_sentinel2_bands(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch B03, B04, B05, B08 from Sentinel-2 L2A for the given coordinate.

    Three-pass retry strategy to handle cloud cover and missing data:
      Pass 1: 15-day window + loosened SCL mask (best quality)
      Pass 2: 15-day window, NO SCL mask  (fallback: noisier but never all-NaN)
      Pass 3: 45-day window, NO SCL mask  (wide fallback for persistent cloud cover)

    Returns
    -------
    dict with keys: bands, bbox, time_range, timestamp, masked
    """
    conn = connect_to_openeo()
    bbox = get_bounding_box(lat, lon)

    attempts = [
        (15, True,  "15-day window + SCL mask"),
        (15, False, "15-day window, NO mask (fallback)"),
        (45, False, "45-day window, NO mask (wide fallback)"),
    ]

    for days_back, use_mask, label in attempts:
        start_date, end_date = get_time_range(days_back)
        logger.info(
            f"Fetching Sentinel-2 | strategy='{label}' | "
            f"bbox={bbox} | time={start_date}/{end_date}"
        )

        bands: Dict[str, np.ndarray] = {}
        download_ok = True

        for band in REQUIRED_BANDS:
            try:
                cube = _build_composite(
                    conn, bbox, start_date, end_date, band, use_mask
                )
                arr = _download_to_array(cube, band)
                bands[band] = arr
            except Exception as e:
                logger.warning(f"Band {band} download failed ({label}): {e}")
                download_ok = False
                break   # try next strategy entirely

        if not download_ok:
            continue

        nan_frac = _nan_fraction(bands)
        logger.info(f"Strategy '{label}': average NaN fraction = {nan_frac:.1%}")

        if nan_frac < NAN_THRESHOLD:
            logger.info(f"Valid data obtained with strategy: '{label}'")
            return {
                "bands":      bands,
                "bbox":       bbox,
                "time_range": (start_date, end_date),
                "timestamp":  datetime.utcnow().isoformat(),
                "masked":     use_mask,
            }

        logger.warning(
            f"Strategy '{label}': NaN={nan_frac:.1%} exceeds threshold "
            f"{NAN_THRESHOLD:.0%}. Trying next strategy."
        )

    raise ValueError(
        f"No valid Sentinel-2 data for lat={lat}, lon={lon} after all fallback "
        f"strategies. The area is likely persistently cloud-covered or outside "
        f"Sentinel-2 coverage. Try again in a few days."
    )