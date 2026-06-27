"""
Hotel recommendation engine.
Loads PERSISTED artifacts (trained once) instead of retraining on every launch.

Artifacts (in ./artifacts):
  svd_model.pkl   - trained Surprise SVD collaborative-filtering model
  content.pkl     - {hotels, cosine_sim, name_to_idx} for content-based filtering
  meta.pkl        - {all_hotels, user_seen, hotel_info, rmse}

Public API:
  collaborative_recommend(user_id, top_n)
  content_similar(hotel_name, top_n)
  hybrid_recommend(user_id, top_n)
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE_DIR, "artifacts")

with open(os.path.join(ART, "svd_model.pkl"), "rb") as f:
    _SVD = pickle.load(f)
with open(os.path.join(ART, "content.pkl"), "rb") as f:
    _CONTENT = pickle.load(f)
with open(os.path.join(ART, "meta.pkl"), "rb") as f:
    _META = pickle.load(f)

ALL_HOTELS = _META["all_hotels"]
USER_SEEN = _META["user_seen"]
HOTEL_INFO = _META["hotel_info"]
RMSE = _META.get("rmse")


def collaborative_recommend(user_id, top_n=5):
    """Predict ratings for hotels the user hasn't booked, return the best.
    If the user has already booked every hotel (common here -- only 9 exist),
    fall back to ranking ALL hotels by predicted preference."""
    user_id = int(user_id)
    seen = USER_SEEN.get(user_id, set())
    unseen = [h for h in ALL_HOTELS if h not in seen]
    candidates = unseen if unseen else ALL_HOTELS  # fallback for power users
    scored = [(h, float(_SVD.predict(user_id, h).est)) for h in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def content_similar(hotel_name, top_n=5):
    """Return hotels most similar to a given hotel by name/place text features."""
    name_to_idx = _CONTENT["name_to_idx"]
    if hotel_name not in name_to_idx:
        return []
    idx = name_to_idx[hotel_name]
    sims = list(enumerate(_CONTENT["cosine_sim"][idx]))
    sims.sort(key=lambda x: x[1], reverse=True)
    hotels = _CONTENT["hotels"]
    return [hotels[i] for i, _ in sims[1:top_n + 1]]


def hybrid_recommend(user_id, top_n=5):
    """Blend CF and content: take CF-preferred hotels, expand with similar ones."""
    cf = collaborative_recommend(user_id, top_n=max(top_n, 5))
    result, seen = [], set()
    for hotel, _ in cf:
        for cand in [hotel] + content_similar(hotel, top_n=2):
            if cand not in seen:
                seen.add(cand)
                result.append(cand)
            if len(result) >= top_n:
                return result
    return result
