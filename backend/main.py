"""
FastAPI backend for the Sydney housing price prediction tool.

Loads the Ridge pipeline exported by the notebook and serves predictions over HTTP.
The Streamlit front end talks to this, and you can also hit it directly.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Interactive docs once it is running:
    http://localhost:8000/docs
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.features import engineer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("housing-api")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

app = FastAPI(
    title="Sydney Housing Price API",
    description=(
        "Predicts the sale price of a residential property in Mosman, Parramatta or "
        "Blacktown. Built for SIT720 task 8.1D. This is a decision support tool, not "
        "a valuation. Every response carries a confidence band and a warning list."
    ),
    version="1.0.0",
)

# The Streamlit app runs on a different port and origin, so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
META = {}


MODEL_SOURCE = "not loaded"


@app.on_event("startup")
def load_model():
    """Load once at startup rather than per request.

    The saved pipeline is a pickle, so it is tied to the scikit-learn version that
    created it. Hosting platforms do not always give you the version you pinned, and
    a pickle that will not unpickle would take the whole service down. Ridge on 120
    rows trains in under a second, so if the artefact is missing or refuses to load
    we rebuild it from the CSV instead. Verified to produce identical predictions.
    """
    global MODEL, META, MODEL_SOURCE

    if META_PATH.exists():
        META = json.loads(META_PATH.read_text())
    else:
        log.warning("metadata missing at %s", META_PATH)

    if MODEL_PATH.exists():
        try:
            MODEL = joblib.load(MODEL_PATH)
            MODEL_SOURCE = "loaded from models/best_model.joblib"
            log.info("loaded %s trained on %s rows",
                     META.get("model_name"), META.get("trained_on_rows"))
            return
        except Exception as exc:
            log.warning("could not unpickle %s (%s), retraining from CSV",
                        MODEL_PATH, exc)
    else:
        log.warning("no model artefact at %s, retraining from CSV", MODEL_PATH)

    csv = ROOT / "data" / "sydney_housing_raw.csv"
    if not csv.exists():
        log.error("no dataset at %s either, cannot serve predictions", csv)
        return

    try:
        from backend.train import train_from_csv
        MODEL = train_from_csv(csv)
        MODEL_SOURCE = "retrained from data/sydney_housing_raw.csv at startup"
        log.info("retrained Ridge pipeline from %s", csv)
    except Exception as exc:
        log.error("retraining failed: %s", exc)


# ---------------------------------------------------------------------------
# request and response shapes
# ---------------------------------------------------------------------------
class PropertyIn(BaseModel):
    suburb: str = Field(..., description="Mosman, Parramatta or Blacktown")
    property_type: str = Field(..., description="House, Townhouse or Apartment")
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=8)
    car_spaces: int = Field(0, ge=0, le=8)
    land_size_sqm: float = Field(0.0, ge=0, le=5000)
    internal_area_sqm: float = Field(..., gt=0, le=1500)
    year_built: int = Field(1990, ge=1850, le=2030)
    has_pool: int = Field(0, ge=0, le=1)
    has_aircon: int = Field(1, ge=0, le=1)
    renovated: int = Field(0, ge=0, le=1)
    water_view: int = Field(0, ge=0, le=1)
    distance_to_station_km: float = Field(1.5, ge=0, le=20)
    sale_method: str = Field("Auction", description="Auction or Private Treaty")
    days_on_market: int = Field(30, ge=0, le=500)
    sale_date: str = Field("2025-08-01", description="YYYY-MM-DD")
    agent_description: str = Field("", description="Listing blurb, optional but it helps")

    @field_validator("suburb")
    @classmethod
    def known_suburb(cls, v):
        allowed = META.get("suburbs", ["Blacktown", "Mosman", "Parramatta"])
        if v not in allowed:
            raise ValueError(f"suburb must be one of {allowed}")
        return v

    @field_validator("property_type")
    @classmethod
    def known_type(cls, v):
        allowed = META.get("property_types", ["Apartment", "House", "Townhouse"])
        if v not in allowed:
            raise ValueError(f"property_type must be one of {allowed}")
        return v

    @field_validator("sale_method")
    @classmethod
    def known_method(cls, v):
        allowed = META.get("sale_methods", ["Auction", "Private Treaty"])
        if v not in allowed:
            raise ValueError(f"sale_method must be one of {allowed}")
        return v


class PredictionOut(BaseModel):
    predicted_price: float
    lower_80: float
    upper_80: float
    confidence: str
    warnings: List[str]
    model_name: str
    segment_uncertainty_pct: float
    comparable_median: Optional[float] = None


class BatchIn(BaseModel):
    properties: List[PropertyIn]


# ---------------------------------------------------------------------------
# prediction logic
# ---------------------------------------------------------------------------
def build_warnings(p: PropertyIn, predicted: float) -> List[str]:
    """Part 4 identified specific situations where this model should not be trusted.
    Rather than burying that in the report, the API returns it with every prediction."""
    out = []
    rng = META.get("training_price_range", {})

    if rng and predicted < rng.get("p05", 0):
        out.append(
            "This estimate is below the 5th percentile of the training data. The model "
            "has very few comparable sales this cheap and the estimate may be unreliable.")
    if rng and predicted > rng.get("p95", 1e12):
        out.append(
            "This estimate is above the 95th percentile of the training data. Prestige "
            "property sells on scarcity and outlook, which this model does not measure.")

    # the single worst failure in Part 4 was a small apartment in an expensive suburb
    defaults = META.get("suburb_defaults", {}).get(p.suburb, {})
    med = defaults.get("median_price")
    if med and p.bedrooms <= 1 and med > 2_000_000:
        out.append(
            "Small dwellings in expensive suburbs are the model's worst failure case. "
            "It tends to inherit the suburb price level and overshoot. Treat this as a "
            "ceiling rather than an estimate.")

    if p.property_type == "Townhouse":
        out.append(
            "Townhouses had the highest error rate in testing, around 20 percent in "
            "Mosman and Parramatta. There were few of them in the training data and "
            "their pricing sits awkwardly between houses and apartments.")

    if p.property_type in ("House", "Townhouse") and p.land_size_sqm <= 0:
        out.append(
            f"Land size is zero for a {p.property_type.lower()}, which is almost "
            "certainly a missing value rather than a real one. Land is a major price "
            "driver for this property type, so the estimate below is likely to be far "
            "too low. Supply a land size.")

    if p.internal_area_sqm > 0 and p.bedrooms / max(p.internal_area_sqm, 1) > 0.06:
        out.append(
            "The bedroom count is high for the internal area given. This combination is "
            "rare in the training data so the prediction is an extrapolation.")

    if not out:
        out.append(
            "No specific risk flags. This property sits in the range where the model is "
            "most reliable. The estimate is still a starting point, not a valuation.")
    return out


def predict_one(p: PropertyIn) -> PredictionOut:
    if MODEL is None:
        raise HTTPException(503, "Model not loaded. Run the notebook to create models/best_model.joblib")

    defaults = META.get("suburb_defaults", {}).get(p.suburb, {})

    row = p.model_dump()
    # these three are fixed per suburb, so the caller does not have to supply them
    row["postcode"] = defaults.get("postcode", 2000)
    row["distance_to_cbd_km"] = defaults.get("distance_to_cbd_km", 20.0)
    row["school_catchment_rating"] = defaults.get("school_catchment_rating", 7.0)

    if p.property_type == "Apartment":
        row["land_size_sqm"] = 0.0
    if not row["agent_description"].strip():
        # the model saw a blurb for every training row, so an empty string is out of
        # distribution. A neutral description keeps it closer to what it expects.
        row["agent_description"] = (
            f"{p.bedrooms} bedroom {p.property_type.lower()} in {p.suburb}.")

    frame = engineer(pd.DataFrame([row]))

    pred_log = float(MODEL.predict(frame)[0])
    predicted = float(np.exp(pred_log))

    # segment specific uncertainty, because Part 4 showed error varies a lot by
    # suburb and property type. A single global band would be misleading.
    key = f"{p.suburb}|{p.property_type}"
    sigma = META.get("segment_sigma_log", {}).get(key, META.get("global_sigma_log", 0.2))

    # 1.28 standard deviations is the 80 percent interval for a normal distribution.
    # Because the model works in log space this becomes a multiplicative band, which
    # is the right shape for prices.
    lower = float(np.exp(pred_log - 1.28 * sigma))
    upper = float(np.exp(pred_log + 1.28 * sigma))

    band_pct = (upper - lower) / predicted * 100
    if band_pct < 35:
        confidence = "High"
    elif band_pct < 55:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return PredictionOut(
        predicted_price=round(predicted, -3),
        lower_80=round(lower, -3),
        upper_80=round(upper, -3),
        confidence=confidence,
        warnings=build_warnings(p, predicted),
        model_name=META.get("model_name", "unknown"),
        segment_uncertainty_pct=round(band_pct, 1),
        comparable_median=defaults.get("median_price"),
    )


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Sydney Housing Price API",
        "status": "ok" if MODEL is not None else "model not loaded",
        "docs": "/docs",
        "endpoints": ["/health", "/metadata", "/predict", "/predict/batch"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_name": META.get("model_name"),
        "model_source": MODEL_SOURCE,
    }


@app.get("/metadata")
def metadata():
    """Everything the front end needs to build its form and show model quality."""
    if not META:
        raise HTTPException(503, "Metadata not loaded")
    return {
        "model_name": META["model_name"],
        "trained_on_rows": META["trained_on_rows"],
        "suburbs": META["suburbs"],
        "property_types": META["property_types"],
        "sale_methods": META["sale_methods"],
        "metrics": META["best_metrics"],
        "all_model_metrics": META["cv_metrics"],
        "suburb_defaults": META["suburb_defaults"],
        "training_price_range": META["training_price_range"],
    }


@app.post("/predict", response_model=PredictionOut)
def predict(p: PropertyIn):
    return predict_one(p)


@app.post("/predict/batch")
def predict_batch(payload: BatchIn):
    """Used by the CSV upload tab in the Streamlit app."""
    if len(payload.properties) > 500:
        raise HTTPException(400, "Maximum 500 properties per batch")
    return {"predictions": [predict_one(p) for p in payload.properties],
            "count": len(payload.properties)}
