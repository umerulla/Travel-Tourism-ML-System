"""API tests for the gender service. Uses a stub predictor so the suite runs
without torch/transformers in CI's fast path; the live model is exercised in
the notebook and the container smoke test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gender_api"))
import pytest
import app as gender_app


@pytest.fixture
def client():
    # Inject a deterministic stub predictor (no torch needed).
    gender_app.PREDICT_FN = lambda names: [("female", 0.97) for _ in names]
    gender_app.app.config["TESTING"] = True
    with gender_app.app.test_client() as c:
        yield c


def test_health(client):
    assert client.get("/health").status_code == 200


def test_predict_single(client):
    r = client.post("/predict", json={"name": "Priya"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["predicted_gender"] == "female"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_batch(client):
    r = client.post("/predict", json={"names": ["Priya", "Rahul"]})
    assert r.status_code == 200
    assert len(r.get_json()["results"]) == 2


def test_missing_name_400(client):
    assert client.post("/predict", json={}).status_code == 400


def test_empty_name_400(client):
    assert client.post("/predict", json={"name": "  "}).status_code == 400


def test_non_json_415(client):
    assert client.post("/predict", data="x").status_code == 415
