"""
processing.py
-------------
Computes water quality indices from Sentinel-2 reflectance bands.

Indices:
    NDWI  = (B03 - B08) / (B03 + B08)   → Water detection (>0 = water)
    NDCI  = (B05 - B04) / (B05 + B04)   → Chlorophyll/algae proxy (pollution indicator)

Pollution classification:
    NDCI < 0       → LOW    (clean water or non-water surface)
    NDCI 0–0.2     → MEDIUM (moderate algae presence)
    NDCI > 0.2     → HIGH   (likely algae bloom / pollution)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def safe_normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """
    Compute normalized difference index: (A - B) / (A + B)
    Handles division-by-zero gracefully by returning NaN where denominator is 0.
    
    Args:
        band_a: First band (numerator addend)
        band_b: Second band (numerator subtrahend)
    
    Returns:
        2D numpy array of index values in range [-1, 1], NaN for invalid pixels
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (band_a + band_b) == 0,
            np.nan,
            (band_a - band_b) / (band_a + band_b),
        )
    return result.astype(np.float32)


def compute_ndwi(B03: np.ndarray, B08: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)
    
    Positive values indicate open water bodies.
    Negative values indicate land, vegetation, or built-up areas.
    
    Args:
        B03: Green band reflectance array
        B08: NIR band reflectance array
    
    Returns:
        2D array of NDWI values
    """
    return safe_normalized_difference(B03, B08)


def compute_turbidity(B04: np.ndarray, B03: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)

    Positive values indicate open water bodies.
    Negative values indicate land, vegetation, or built-up areas.

    Args:
        B03: Green band reflectance array
        B08: NIR band reflectance array

    Returns:
        2D array of NDWI values
    """
    return safe_normalized_difference(B04, B03)


def compute_suspendent_sediment(B04: np.ndarray, B02: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)

    Positive values indicate open water bodies.
    Negative values indicate land, vegetation, or built-up areas.

    Args:
        B03: Green band reflectance array
        B08: NIR band reflectance array

    Returns:
        2D array of NDWI values
    """
    return safe_normalized_difference(B04, B02)


def compute_ndci(B05: np.ndarray, B04: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Chlorophyll Index.
    NDCI = (RedEdge - Red) / (RedEdge + Red)
    
    Higher values indicate more chlorophyll (algae/phytoplankton),
    which is used as a proxy for water pollution / eutrophication.
    
    Args:
        B05: Red Edge band reflectance array
        B04: Red band reflectance array
    
    Returns:
        2D array of NDCI values
    """
    return safe_normalized_difference(B05, B04)


def classify_pollution(ndci_value: float) -> str:
    """
    Classify pollution level based on NDCI value.
    
    Args:
        ndci_value: Scalar NDCI value (median over AOI)
    
    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    if np.isnan(ndci_value):
        return "UNKNOWN"
    if ndci_value < 0:
        return "LOW"
    elif ndci_value <= 0.2:
        return "MEDIUM"
    else:
        return "HIGH"


def extract_point_value(array: np.ndarray) -> Optional[float]:
    """
    Extract a representative scalar from a 2D array.
    Uses spatial median to be robust against edge artifacts and remaining noise.
    Returns None if all pixels are NaN (no valid data — likely cloud cover).
    
    Args:
        array: 2D numpy array
    
    Returns:
        Median float value, or None if no valid data
    """
    valid = array[~np.isnan(array)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def process_water_quality(bands: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Full processing pipeline:
    1. Compute NDWI and NDCI arrays from band reflectances
    2. Extract median values for the area of interest
    3. Determine water presence and pollution status
    
    Args:
        bands: Dict of band arrays {"B03": arr, "B04": arr, "B05": arr, "B08": arr}
    
    Returns:
        Dict with ndwi, ndci, water_detected, pollution_status, or raises ValueError
        if no valid satellite data is available.
    """
    B02 = bands.get("B02")
    B03 = bands.get("B03")
    B04 = bands.get("B04")
    B05 = bands.get("B05")
    B08 = bands.get("B08")

    if any(b is None for b in [B02, B03, B04, B05, B08]):
        raise ValueError("Missing required bands. Cannot compute water quality indices.")

    # Compute indices across the spatial extent
    ndwi_array = compute_ndwi(B03, B08)
    ndci_array = compute_ndci(B05, B04)
    turb_array = compute_turbidity(B04, B03)
    sediment_array = compute_suspendent_sediment(B04, B02)

    # Extract representative scalar values (spatial median)
    ndwi_val = extract_point_value(ndwi_array)
    ndci_val = extract_point_value(ndci_array)
    turb_val = extract_point_value(turb_array)
    sediment_val = extract_point_value(sediment_array)

    # If both are None → no valid data at all (cloud cover, out-of-bounds, etc.)
    if ndwi_val is None and ndci_val is None and turb_val is None and sediment_val is None:
        raise ValueError(
            "No valid satellite data available for this location. "
            "The area may be fully cloud-covered. Try again in a few days."
        )

    # Water detection: NDWI > 0.3 is a reliable threshold for open water
    # Lower threshold (0.0) may catch shallow/turbid water too
    water_detected = bool(ndwi_val is not None and ndwi_val > 0.1)

    # Pollution classification based on NDCI
    pollution_status = classify_pollution(ndci_val if ndci_val is not None else float("nan"))

    logger.info(
        f"Processing complete: NDWI={ndwi_val:.4f}, NDCI={ndci_val:.4f}, "
        f"water={water_detected}, pollution={pollution_status}"
    )

    return {
        "ndwi": round(ndwi_val, 4) if ndwi_val is not None else None,
        "ndci": round(ndci_val, 4) if ndci_val is not None else None,
        "turbidity": round(turb_val, 4) if turb_val is not None else None,
        "suspendent_sediment": round(sediment_val, 4) if sediment_val is not None else None,
        "water_detected": water_detected,
        "pollution_status": pollution_status,
    }
