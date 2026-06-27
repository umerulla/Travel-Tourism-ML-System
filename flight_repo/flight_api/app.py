"""
Flight Price Prediction REST API
--------------------------------
Serves the trained XGBoost regressor for real-time flight price prediction.

Design decisions that address the review feedback:
  * Feature ordering is NOT taken from the JSON payload order. It is read from
    `feature_schema.json` and enforced explicitly, so the request is order-independent.
  * All artifact paths are resolved relative to THIS file (BASE_DIR), so the app
    behaves identically in Colab, Docker, and Kubernetes.
  * Every input is validated (presence, type, range, and categorical vocabulary)
    and bad input returns HTTP 400 with a clear message instead of a 500 crash.
  * Errors are caught and returned as structured JSON with the correct status code.
"""

import os
import json
import logging

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ----------------------------------------------------------------------------
# Paths (resolved relative to this file -> consistent across all environments)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "feature_schema.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("flight-api")

# ----------------------------------------------------------------------------
# Load schema + model + encoders ONCE at startup
# ----------------------------------------------------------------------------
with open(SCHEMA_PATH, "r") as f:
    SCHEMA = json.load(f)

FEATURE_ORDER = SCHEMA["feature_order"]            # canonical, fixed order
CATEGORICAL = SCHEMA["categorical_features"]
NUMERIC = SCHEMA["numeric_features"]

MODEL = joblib.load(os.path.join(ARTIFACT_DIR, SCHEMA["model_file"]))
ENCODERS = {
    col: joblib.load(os.path.join(ARTIFACT_DIR, fname))
    for col, fname in SCHEMA["encoder_files"].items()
}

log.info("Loaded model + %d encoders. Feature order: %s", len(ENCODERS), FEATURE_ORDER)

app = Flask(__name__)


def _validate_and_encode(payload):
    """Validate a request payload and return a single-row DataFrame in the
    canonical feature order. Raises ValueError with a user-readable message."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}. "
                         f"Required: {FEATURE_ORDER}")

    row = {}

    # Categorical features: must exist in the encoder's known vocabulary.
    for col in CATEGORICAL:
        value = payload[col]
        encoder = ENCODERS[col]
        known = set(encoder.classes_.tolist())
        if value not in known:
            sample = sorted(known)[:10]
            raise ValueError(
                f"Unknown value '{value}' for '{col}'. "
                f"Expected one of (showing up to 10): {sample}"
            )
        row[col] = int(encoder.transform([value])[0])

    # Numeric features: must be a non-negative number.
    for col in NUMERIC:
        value = payload[col]
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{col}' must be numeric, got '{value}'.")
        if num < 0:
            raise ValueError(f"Field '{col}' must be non-negative, got {num}.")
        row[col] = num

    # Build DataFrame with columns in the EXACT canonical order.
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Flight Price Prediction API",
        "status": "running",
        "endpoints": {"/health": "GET", "/predict": "POST (JSON)"},
        "required_fields": FEATURE_ORDER,
    })


@app.route("/health", methods=["GET"])
def health():
    """Liveness/readiness probe target for Kubernetes."""
    return jsonify({"status": "healthy", "model_loaded": MODEL is not None}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415
    try:
        payload = request.get_json(silent=True)
        features_df = _validate_and_encode(payload)
        prediction = float(MODEL.predict(features_df)[0])
        return jsonify({
            "predicted_price": round(prediction, 2),
            "currency": "model-units",
            "input": payload,
        }), 200
    except ValueError as ve:
        # Expected, user-facing validation problem.
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        log.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
