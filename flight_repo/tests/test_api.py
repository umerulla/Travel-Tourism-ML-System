"""
API tests for the Flight Price Prediction service.
Run from the repo root:  pytest -q
These run inside Jenkins' "Test API" stage so the build fails on regressions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flight_api"))

import pytest  # noqa: E402
import app as flight_app  # noqa: E402


@pytest.fixture
def client():
    flight_app.app.config["TESTING"] = True
    with flight_app.app.test_client() as c:
        yield c


def _valid_payload():
    # Pull one known-good value from each fitted encoder so the test is data-agnostic.
    return {
        "from": flight_app.ENCODERS["from"].classes_[0],
        "to": flight_app.ENCODERS["to"].classes_[0],
        "flightType": flight_app.ENCODERS["flightType"].classes_[0],
        "agency": flight_app.ENCODERS["agency"].classes_[0],
        "time": 1.5,
        "distance": 800,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"


def test_predict_valid(client):
    r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    assert isinstance(r.get_json()["predicted_price"], float)


def test_predict_is_order_independent(client):
    payload = _valid_payload()
    reordered = dict(reversed(list(payload.items())))
    a = client.post("/predict", json=payload).get_json()["predicted_price"]
    b = client.post("/predict", json=reordered).get_json()["predicted_price"]
    assert a == b  # feature order comes from schema, not JSON key order


def test_missing_field_returns_400(client):
    r = client.post("/predict", json={"from": "X"})
    assert r.status_code == 400


def test_unknown_category_returns_400(client):
    p = _valid_payload(); p["from"] = "NOWHERE-CITY"
    r = client.post("/predict", json=p)
    assert r.status_code == 400


def test_negative_numeric_returns_400(client):
    p = _valid_payload(); p["time"] = -5
    r = client.post("/predict", json=p)
    assert r.status_code == 400


def test_non_json_returns_415(client):
    r = client.post("/predict", data="not json")
    assert r.status_code == 415
