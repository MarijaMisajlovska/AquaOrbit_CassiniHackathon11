"""
data_fetch.py
-------------
Handles all communication with the Copernicus Data Space via the openEO API.
Fetches Sentinel-2 L2A data for a given bounding box and time range.
"""

import openeo
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging

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
    
    Args:
        lat: Latitude of the point of interest
        lon: Longitude of the point of interest
        offset: Half-width of the bounding box in degrees
    
    Returns:
        Dict with west, south, east, north keys (standard openEO bbox format)
    """
    return {
        "west": lon - offset,
        "south": lat - offset,
        "east": lon + offset,
        "north": lat + offset,
    }


def get_time_range(days_back: int = 10) -> Tuple[str, str]:
    """
    Compute a date range from today back N days to get recent imagery.
    
    Args:
        days_back: How many days back to look for satellite data
    
    Returns:
        Tuple of (start_date, end_date) as ISO strings
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def connect_to_openeo() -> openeo.DataCube:
    """
    Authenticate and connect to the Copernicus openEO backend.
    Uses OIDC device code flow — on first run this will open a browser
    for login. Credentials are cached locally after first auth.
    
    Returns:
        Authenticated openEO connection object
    """
    try:
        conn = openeo.connect(OPENEO_URL)
        # authenticate_oidc uses cached credentials if available,
        # otherwise triggers interactive OIDC device-code login
        conn.authenticate_oidc()
        logger.info("Connected to Copernicus openEO backend successfully.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to openEO: {e}")
        raise RuntimeError(f"openEO connection failed: {e}")


def fetch_sentinel2_bands(lat: float, lon: float) -> Dict[str, Any]:
    """
    Core data pipeline:
    1. Connect to openEO
    2. Load Sentinel-2 L2A collection for bounding box & recent time range
    3. Apply cloud masking via SCL band (Scene Classification Layer)
    4. Compute median composite across time (reduces cloud contamination)
    5. Normalize reflectance values (divide by 10000 — Sentinel-2 stores as DN)
    6. Download and return band arrays as numpy arrays
    
    Args:
        lat: Latitude of the point of interest
        lon: Longitude of the point of interest
    
    Returns:
        Dict with band arrays (B03, B04, B05, B08) and metadata
    """
    conn = connect_to_openeo()

    bbox = get_bounding_box(lat, lon)
    start_date, end_date = get_time_range(days_back=10)

    logger.info(f"Fetching Sentinel-2 data for bbox={bbox}, time={start_date}/{end_date}")

    # Load the Sentinel-2 L2A collection
    # We include SCL (Scene Classification Layer) for cloud masking
    datacube = conn.load_collection(
        COLLECTION,
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=REQUIRED_BANDS + ["SCL"],
        max_cloud_cover=80,  # pre-filter scenes with extreme cloud cover
    )

    # --- Cloud masking using SCL band ---
    # SCL values: 4=Vegetation, 5=Not-vegetated, 6=Water, 7=Unclassified
    # We mask out: 0=No data, 1=Saturated, 2=Dark, 3=Cloud shadow,
    #              8=Med cloud, 9=High cloud, 10=Thin cirrus, 11=Snow
    scl = datacube.band("SCL")
    # Keep only valid pixels (SCL classes 4, 5, 6, 7)
    valid_mask = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)

    # Mask the datacube — invalid pixels become NaN
    datacube_masked = datacube.mask(~valid_mask)

    # Drop SCL from the band stack before computing median
    datacube_masked = datacube_masked.filter_bands(REQUIRED_BANDS)

    # Median composite across time dimension — reduces remaining cloud artifacts
    # This gives us one representative image per pixel
    composite = datacube_masked.reduce_dimension(dimension="t", reducer="median")

    # Normalize: Sentinel-2 DN values range 0–10000 → reflectance 0.0–1.0
    normalized = composite / 10000.0

    # Execute the process graph and download as a GeoTIFF
    # synchronous=True waits for result (suitable for small bboxes)
    result = normalized.download(format="GTiff")

    # Parse the downloaded GeoTIFF into numpy arrays per band
    bands = _parse_geotiff(result, REQUIRED_BANDS)

    return {
        "bands": bands,
        "bbox": bbox,
        "time_range": (start_date, end_date),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _parse_geotiff(geotiff_bytes: bytes, band_names: list) -> Dict[str, np.ndarray]:
    """
    Parse the downloaded GeoTIFF bytes into individual numpy arrays per band.
    
    Args:
        geotiff_bytes: Raw bytes of the downloaded GeoTIFF file
        band_names: Ordered list of band names matching the GTiff band order
    
    Returns:
        Dict mapping band name → 2D numpy array of reflectance values
    """
    import io
    import rasterio

    with rasterio.open(io.BytesIO(geotiff_bytes)) as src:
        bands = {}
        for i, name in enumerate(band_names, start=1):
            arr = src.read(i).astype(np.float32)
            # Replace no-data sentinel values with NaN
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            bands[name] = arr

    return bands
