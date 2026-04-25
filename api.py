"""
api.py
------
FastAPI backend for real-time water quality monitoring.

Endpoints:
    POST /analyze-water   → accepts {lat, lon}, returns water quality analysis
    GET  /health          → health check

Uses ThreadPoolExecutor to run blocking openEO calls off the async event loop,
so multiple users can query different coordinates simultaneously without blocking.
"""

import asyncio
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from data_fetch import fetch_sentinel2_bands
from processing import process_water_quality

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Water Quality Monitor API",
    description="Real-time water quality analysis using Copernicus Sentinel-2 satellite data.",
    version="1.0.0",
)

# Allow the frontend (running on any origin during hackathon) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thread pool for blocking openEO calls ───────────────────────────────────────
# Max 10 concurrent satellite data fetches (tune based on openEO rate limits)
executor = ThreadPoolExecutor(max_workers=10)

# ── Simple in-memory cache ──────────────────────────────────────────────────────
# Cache results for up to 30 minutes to avoid redundant satellite requests
# Key: rounded (lat, lon) string, Value: (result_dict, cached_at datetime)
_cache: dict = {}
CACHE_TTL_MINUTES = 30
# Round coordinates to 2 decimal places (~1km grid) for cache key
CACHE_PRECISION = 2


def _cache_key(lat: float, lon: float) -> str:
    rounded_lat = round(lat, CACHE_PRECISION)
    rounded_lon = round(lon, CACHE_PRECISION)
    return f"{rounded_lat},{rounded_lon}"


def _get_cached(lat: float, lon: float) -> Optional[dict]:
    key = _cache_key(lat, lon)
    if key in _cache:
        result, cached_at = _cache[key]
        if datetime.utcnow() - cached_at < timedelta(minutes=CACHE_TTL_MINUTES):
            logger.info(f"Cache HIT for key={key}")
            return result
        else:
            del _cache[key]  # expired
    return None


def _set_cached(lat: float, lon: float, result: dict):
    key = _cache_key(lat, lon)
    _cache[key] = (result, datetime.utcnow())


# ── Pydantic models ─────────────────────────────────────────────────────────────
class WaterAnalysisRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v):
        if v < -90 or v > 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v):
        if v < -180 or v > 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v


class LocationInfo(BaseModel):
    lat: float
    lon: float


class WaterAnalysisResponse(BaseModel):
    location: LocationInfo
    ndwi: Optional[float]
    ndci: Optional[float]
    water_detected: bool
    pollution_status: str
    timestamp: str
    cached: bool = False


# ── Core analysis function (runs in thread pool) ────────────────────────────────
def _run_analysis(lat: float, lon: float) -> dict:
    """
    Synchronous function that fetches satellite data and computes water quality.
    This is called from the thread pool so it doesn't block the event loop.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Full response dict ready to be returned by the API

    Raises:
        ValueError: If no valid satellite data is available
        RuntimeError: If the openEO connection or processing fails
    """
    logger.info(f"Starting satellite analysis for lat={lat}, lon={lon}")

    # Step 1: Fetch Sentinel-2 band data from Copernicus openEO
    fetch_result = fetch_sentinel2_bands(lat, lon)

    # Step 2: Compute NDWI, NDCI, and classify pollution
    quality = process_water_quality(fetch_result["bands"])

    return {
        "location": {"lat": lat, "lon": lon},
        "ndwi": quality["ndwi"],
        "ndci": quality["ndci"],
        "water_detected": quality["water_detected"],
        "pollution_status": quality["pollution_status"],
        "timestamp": fetch_result["timestamp"],
        "cached": False,
    }


# ── API Endpoints ───────────────────────────────────────────────────────────────
@app.post("/analyze-water", response_model=WaterAnalysisResponse)
async def analyze_water(request: WaterAnalysisRequest):
    """
    Analyze water quality at the given coordinates using Sentinel-2 satellite data.

    - Fetches latest available imagery (last 10 days)
    - Applies cloud masking and median compositing
    - Returns NDWI (water detection) and NDCI (pollution proxy)
    - Results are cached for 30 minutes to optimize performance
    """
    lat, lon = request.lat, request.lon

    # Check cache first
    cached = _get_cached(lat, lon)
    if cached:
        cached["cached"] = True
        return WaterAnalysisResponse(**cached)

    # Run the blocking satellite pipeline in a thread (non-blocking for the API)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _run_analysis, lat, lon)
    except ValueError as e:
        # No data available (cloud cover, missing pixels, etc.)
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # Connection or processing error
        raise HTTPException(status_code=503, detail=f"Satellite data service error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error for lat={lat}, lon={lon}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Cache the result
    _set_cached(lat, lon, result)

    return WaterAnalysisResponse(**result)


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "service": "Water Quality Monitor API",
        "cache_size": len(_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")