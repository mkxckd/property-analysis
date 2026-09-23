"""
Live fair-value API: serves the same fitted model that run_pipeline.py
trains, so estimates aren't limited to the precomputed grid the static
dashboard uses.

    python run_pipeline.py            # trains + saves models/fair_value.joblib
    uvicorn api.main:app --reload     # then serve it

Endpoints:
    GET  /health     model loaded? which areas/types does it know?
    GET  /metrics    the pipeline's evaluation.json (held-out metrics etc.)
    POST /estimate   fair rent for a listing described by its features
    POST /score      same, plus a listed price -> residual + fraud flag
    GET  /           the dashboard (frontend/), if present
"""
import json
import os
import sys
from functools import lru_cache
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generate_data import AREAS, PROPERTY_TYPES  # noqa: E402
from pipeline.valuation_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_derived_features  # noqa: E402

MODEL_PATH = os.environ.get("SOUQPULSE_MODEL_PATH", os.path.join(ROOT, "models", "fair_value.joblib"))
METRICS_PATH = os.path.join(ROOT, "results", "evaluation.json")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

# Same threshold run_pipeline.py evaluates against ground truth.
FRAUD_THRESHOLD = -0.30

AREA_COORDS = {name: (lat, lon) for name, lat, lon, _ in AREAS}
TOWER_TYPES = {"Studio", "1BR Apartment", "2BR Apartment", "3BR Apartment"}

AreaName = Literal[tuple(AREA_COORDS)]  # type: ignore[valid-type]
PropertyType = Literal[tuple(PROPERTY_TYPES)]  # type: ignore[valid-type]


class ListingFeatures(BaseModel):
    area: AreaName
    property_type: PropertyType
    size_sqft: float = Field(gt=100, le=20000)
    bedrooms: Optional[int] = Field(default=None, ge=0, le=10,
                                    description="defaults to the typical count for the property type")
    furnished: bool = False
    building_age_years: float = Field(default=8, ge=0, le=100)
    floor: Optional[int] = Field(default=None, ge=0, le=200,
                                 description="defaults to 15 for apartments, 0 for townhouses/villas")


class ScoreRequest(ListingFeatures):
    price_aed_month: float = Field(gt=0)


class Estimate(BaseModel):
    predicted_price_aed_month: float
    inputs: dict


class Score(Estimate):
    price_aed_month: float
    residual_aed: float
    residual_pct: float
    flag: Literal["underpriced", "overpriced", "fair"]
    suspicious: bool


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _feature_row(f: ListingFeatures) -> dict:
    beds_min, beds_max, _ = PROPERTY_TYPES[f.property_type]
    lat, lon = AREA_COORDS[f.area]
    return {
        "area": f.area,
        "property_type": f.property_type,
        "size_sqft": f.size_sqft,
        "bedrooms": f.bedrooms if f.bedrooms is not None else (
            beds_min if f.property_type == "Studio" else round((beds_min + beds_max) / 2)),
        "furnished": int(f.furnished),
        "building_age_years": f.building_age_years,
        "floor": f.floor if f.floor is not None else (15 if f.property_type in TOWER_TYPES else 0),
        "lat": lat,
        "lon": lon,
    }


def _predict(f: ListingFeatures) -> tuple[float, dict]:
    try:
        pipe = load_model()
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet: run `python run_pipeline.py` first.")
    row = _feature_row(f)
    X = add_derived_features(pd.DataFrame([row]))[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred = float(round(pipe.predict(X)[0]))
    inputs = {k: v for k, v in row.items() if k not in ("lat", "lon")}
    return pred, inputs


app = FastAPI(title="SouqPulse fair-value API", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "areas": list(AREA_COORDS),
        "property_types": list(PROPERTY_TYPES),
    }


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(503, "No evaluation results yet: run `python run_pipeline.py` first.")
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/estimate", response_model=Estimate)
def estimate(features: ListingFeatures):
    pred, inputs = _predict(features)
    return Estimate(predicted_price_aed_month=pred, inputs=inputs)


@app.post("/score", response_model=Score)
def score(req: ScoreRequest):
    pred, inputs = _predict(req)
    residual = req.price_aed_month - pred
    residual_pct = round(residual / pred, 4)
    # Same bands as the dashboard's live feed (pipeline/streaming_sim.py).
    if residual_pct <= FRAUD_THRESHOLD:
        flag = "underpriced"
    elif residual_pct >= -FRAUD_THRESHOLD:
        flag = "overpriced"
    else:
        flag = "fair"
    return Score(
        predicted_price_aed_month=pred, inputs=inputs, price_aed_month=req.price_aed_month,
        residual_aed=round(residual, 1), residual_pct=residual_pct, flag=flag,
        suspicious=residual_pct <= FRAUD_THRESHOLD,
    )


# Mounted last so the API routes above take precedence over static files.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")
