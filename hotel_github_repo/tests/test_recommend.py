import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hotel_app"))
import recommend as rec

def test_artifacts_loaded():
    assert rec.RMSE is not None and len(rec.ALL_HOTELS) > 0

def test_collaborative_returns_topn():
    recs = rec.collaborative_recommend(0, 3)
    assert len(recs) == 3
    assert all(isinstance(h, str) and isinstance(s, float) for h, s in recs)

def test_collaborative_sorted_desc():
    recs = rec.collaborative_recommend(5, 5)
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)

def test_hybrid_returns_unique():
    recs = rec.hybrid_recommend(0, 5)
    assert len(recs) == len(set(recs)) and len(recs) > 0

def test_content_similar_excludes_self():
    h = rec.ALL_HOTELS[0]
    assert h not in rec.content_similar(h, 3)
