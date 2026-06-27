# Hotel Recommendation System (Recommendation Module)

Hybrid recommender: SVD collaborative filtering + TF-IDF content-based filtering,
served through a Streamlit app. The model is trained ONCE and persisted; the app
loads artifacts at startup (no retraining on launch).

## Structure
```
hotel_app/  recommend.py, hotel_app.py, requirements.txt, artifacts/
tests/      test_recommend.py
docker/     Dockerfile
notebooks/  Hotel_Recommendation_System_Modified.ipynb
```

## Run
```bash
cd hotel_app && pip install -r requirements.txt
streamlit run hotel_app.py        # opens on :8501
```
Artifacts (`artifacts/*.pkl`) are produced by the notebook's persistence cell.

## Docker
```bash
docker build -t hotel-recommender -f docker/Dockerfile .
docker run -p 8501:8501 hotel-recommender
```
