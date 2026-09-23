"""API tests against the real trained model. Skipped until
`python run_pipeline.py` has produced models/fair_value.joblib."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import MODEL_PATH, ROOT, app  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="run `python run_pipeline.py` first")

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)

BASE = {"area": "Dubai Marina", "property_type": "1BR Apartment", "size_sqft": 800}


def test_health():
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert "Palm Jumeirah" in body["areas"]


def test_estimate_matches_precomputed_grid():
    # The dashboard's estimator grid and the API must be the same model.
    with open(os.path.join(ROOT, "frontend", "data", "fair_value_grid.json"), encoding="utf-8") as fh:
        grid = json.load(fh)
    point = next(r for r in grid if r["area"] == "Business Bay" and r["property_type"] == "2BR Apartment")
    resp = client.post("/estimate", json={
        "area": point["area"], "property_type": point["property_type"],
        "size_sqft": point["size_sqft"], "furnished": bool(point["furnished"]),
    })
    assert resp.status_code == 200
    assert resp.json()["predicted_price_aed_month"] == point["predicted_price_aed_month"]


def test_estimate_is_monotonic_in_size_and_area():
    small = client.post("/estimate", json={**BASE, "size_sqft": 650}).json()["predicted_price_aed_month"]
    large = client.post("/estimate", json={**BASE, "size_sqft": 950}).json()["predicted_price_aed_month"]
    cheap_area = client.post("/estimate", json={**BASE, "area": "Dubai Silicon Oasis"}).json()["predicted_price_aed_month"]
    assert large > small
    assert client.post("/estimate", json=BASE).json()["predicted_price_aed_month"] > cheap_area


def test_score_flags_lowball_listing():
    fair = client.post("/estimate", json=BASE).json()["predicted_price_aed_month"]
    body = client.post("/score", json={**BASE, "price_aed_month": round(fair * 0.5)}).json()
    assert body["flag"] == "underpriced" and body["suspicious"] is True
    assert body["residual_pct"] == pytest.approx(-0.5, abs=0.01)

    body = client.post("/score", json={**BASE, "price_aed_month": fair}).json()
    assert body["flag"] == "fair" and body["suspicious"] is False


@pytest.mark.parametrize("bad", [
    {**BASE, "area": "Atlantis"},
    {**BASE, "property_type": "Castle"},
    {**BASE, "size_sqft": -5},
    {"area": "Dubai Marina"},
])
def test_invalid_input_is_rejected(bad):
    assert client.post("/estimate", json=bad).status_code == 422


def test_dashboard_is_served():
    resp = client.get("/")
    assert resp.status_code == 200 and "SouqPulse" in resp.text
