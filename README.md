# 🌊 Water Quality Monitor — Backend

Real-time water quality analysis using Copernicus Sentinel-2 satellite data.

## Project Structure

```
water_quality/
├── api.py              # FastAPI app — endpoints + ThreadPoolExecutor
├── data_fetch.py       # openEO data pipeline — fetches Sentinel-2 bands
├── processing.py       # NDWI/NDCI computation + pollution classification
├── requirements.txt    # Python dependencies
└── water_quality_notebook.ipynb  # Copernicus JupyterHub exploration notebook
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. First-time Copernicus authentication
```bash
python -c "import openeo; c = openeo.connect('https://openeo.dataspace.copernicus.eu'); c.authenticate_oidc()"
```
```bash
Invoke-RestMethod -Uri "http://localhost:8000/analyze-water" -Method Post -ContentType "application/json" -Body '{"lat": 41.0297, "lon": 20.7169}'
```

### 3. Run the API server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/analyze-water \
  -H "Content-Type: application/json" \
  -d '{"lat": 41.0297, "lon": 20.7169}'
```

Expected response:
```json
{
    "location": {
        "lat": 44.7224,
        "lon": 21.1599
    },
    "ndwi": -0.4227,
    "ndci": 0.1948,
    "turbidity": -0.068,
    "suspendent_sediment": 0.082,
    "water_detected": false,
    "pollution_status": "MEDIUM",
    "timestamp": "2026-04-25T14:28:23.527454",
    "cached": false
}
```

## API Endpoints

| Method | Path             | Description                            |
|--------|------------------|----------------------------------------|
| POST   | `/analyze-water` | Analyze water quality at {lat, lon}    |
| GET    | `/health`        | Health check + cache stats             |
| GET    | `/docs`          | Auto-generated Swagger UI              |

## Connecting the Frontend

In your frontend (the Macedonia map), when a user clicks a water area:
```javascript
const response = await fetch('http://localhost:8000/analyze-water', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ lat: clickedLat, lon: clickedLon })
});
const data = await response.json();
// data.pollution_status → "LOW" | "MEDIUM" | "HIGH"
// data.water_detected   → true/false
// data.ndwi             → float
// data.ndci             → float
```

## Pollution Classification

| NDCI Value | Status | Meaning |
|------------|--------|---------|
| < 0        | 🟢 LOW    | Clean water or land surface |
| 0 – 0.2    | 🟡 MEDIUM | Moderate algae presence |
| > 0.2      | 🔴 HIGH   | Algae bloom / likely pollution |

## JupyterHub Notebook

Upload `water_quality_notebook.ipynb` to:
https://jupyterhub.dataspace.copernicus.eu

The notebook lets you:
- Explore any coordinate interactively
- Visualize NDWI and NDCI maps
- Run batch analysis on multiple Macedonian lakes
- Call the FastAPI backend from the notebook

## Performance Notes

- Satellite data fetch takes **60–120 seconds** per unique location (openEO job execution)
- Results are **cached for 30 minutes** to avoid redundant requests
- Cache key rounds to ±0.01° (~1km grid)
- Up to **10 concurrent** satellite requests via ThreadPoolExecutor

For hackathon demos, pre-warm the cache by calling key locations on startup.
