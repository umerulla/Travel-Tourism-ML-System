"""
Gender Classification REST API (BERT)
-------------------------------------
Serves the fine-tuned BERT model saved in ./bert_gender_classifier.

Fixes applied vs. the original notebook:
  * ONE model is served: the fine-tuned BERT classifier. The throwaway
    RandomForest-on-10-sample-names model and the duplicate app.py / Gradio
    variants are removed, so there is no ambiguity about what is deployed.
  * Model path is resolved relative to this file (works in Colab/Docker/K8s).
  * Heavy imports (torch/transformers) are lazy, so the module can be imported
    and unit-tested without loading the full model.
  * Input is validated: name must be a non-empty string; returns 400 otherwise.
  * Returns the predicted label AND a confidence score (softmax probability).
"""

import os
import logging

from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("GENDER_MODEL_DIR", os.path.join(BASE_DIR, "bert_gender_classifier"))
LABELS = {0: "male", 1: "female"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gender-api")

# Pluggable predictor: predict_fn(list[str]) -> list[(label:str, prob:float)]
# Left as None until first use so the module imports without torch.
PREDICT_FN = None


def _load_real_predictor():
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Loaded BERT model from %s on %s", MODEL_DIR, device)

    def predict_fn(names):
        enc = tokenizer(names, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1)
        out = []
        for i, p in zip(idx.tolist(), probs.tolist()):
            out.append((LABELS[i], float(p[i])))
        return out

    return predict_fn


def get_predictor():
    global PREDICT_FN
    if PREDICT_FN is None:
        PREDICT_FN = _load_real_predictor()
    return PREDICT_FN


app = Flask(__name__)


def _extract_names(payload):
    """Accept either {"name": "X"} or {"names": ["X", "Y"]}. Validate types."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    if "names" in payload:
        names = payload["names"]
        if not isinstance(names, list) or not names:
            raise ValueError("'names' must be a non-empty list of strings.")
    elif "name" in payload:
        names = [payload["name"]]
    else:
        raise ValueError("Provide a 'name' (string) or 'names' (list of strings).")
    for n in names:
        if not isinstance(n, str) or not n.strip():
            raise ValueError("Each name must be a non-empty string.")
    return [n.strip() for n in names]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Gender Classification API (BERT)",
        "status": "running",
        "endpoints": {"/health": "GET", "/predict": "POST (JSON)"},
        "usage": {"name": "string"},
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415
    try:
        names = _extract_names(request.get_json(silent=True))
        results = get_predictor()(names)
        payload = [
            {"name": n, "predicted_gender": label, "confidence": round(prob, 4)}
            for n, (label, prob) in zip(names, results)
        ]
        return jsonify(payload[0] if len(payload) == 1 else {"results": payload}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
