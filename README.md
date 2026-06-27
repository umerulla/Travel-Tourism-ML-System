# Travel Tourism ML System

An end-to-end Machine Learning system built for the travel and tourism domain.
Covers three core ML use cases — flight price prediction, gender classification,
and hotel recommendation — each with a fully productionized MLOps stack.

---

## Business Problem

Travel platforms need to predict prices, understand users, and recommend hotels
in real time. This project builds and deploys production-ready ML models for all
three use cases using industry-standard MLOps tools.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | Python, scikit-learn, XGBoost, BERT (HuggingFace), Surprise SVD |
| API | Flask, Gunicorn |
| UI | Streamlit |
| Containerization | Docker |
| Orchestration | Kubernetes (minikube) |
| Workflow Automation | Apache Airflow |
| CI/CD | Jenkins |
| Experiment Tracking | MLflow |
| Testing | pytest |

---

## Repository Structure

```
Travel-Tourism-ML-System/
│
├── Flight_Price_Prediction.ipynb        # Main flight regression notebook
├── Gender_Classification_Model.ipynb    # Main gender classification notebook
├── Hotel_Recommendation_System.ipynb    # Main hotel recommendation notebook
│
├── flight_repo/                         # Flight price prediction module
│   ├── notebooks/
│   │   └── flight_price_prediction.ipynb
│   ├── flight_api/
│   │   ├── app.py                       # Flask REST API
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── artifacts/
│   │       ├── xgb_flight_regressor.joblib
│   │       ├── feature_schema.json
│   │       ├── from_label_encoder.joblib
│   │       ├── to_label_encoder.joblib
│   │       ├── flightType_label_encoder.joblib
│   │       └── agency_label_encoder.joblib
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── dags/
│   │   └── flight_training_dag.py       # Airflow DAG
│   ├── jenkins/
│   │   └── Jenkinsfile
│   ├── tests/
│   │   └── test_api.py
│   ├── docker/
│   │   └── build_and_test_docker.sh
│   └── README.md
│
├── gender_github_repo/                  # Gender classification module
│   ├── notebooks/
│   │   └── Gender_Classification_Model.ipynb
│   ├── gender_api/
│   │   ├── app.py                       # Flask REST API
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── MODEL_README.txt
│   │   └── bert_gender_classifier/      # Fine-tuned BERT weights (Git LFS)
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── jenkins/
│   │   └── Jenkinsfile
│   ├── tests/
│   │   └── test_api.py
│   ├── docker/
│   │   └── build_and_test_docker.sh
│   └── README.md
│
├── hotel_github_repo/                   # Hotel recommendation module
│   ├── notebooks/
│   │   └── Hotel_Recommendation_System_Modified.ipynb
│   ├── hotel_app/
│   │   ├── hotel_app.py                 # Streamlit UI
│   │   ├── recommend.py                 # Collaborative + content + hybrid logic
│   │   ├── requirements.txt
│   │   └── artifacts/
│   │       ├── svd_model.pkl
│   │       ├── content.pkl
│   │       └── meta.pkl
│   ├── docker/
│   │   └── Dockerfile
│   ├── tests/
│   │   └── test_recommend.py
│   └── README.md
│
└── README.md                            # This file
```

---

## Module 1 — Flight Price Prediction

**Problem:** Predict the price of a flight given origin, destination, flight type, agency, duration, and distance.

**Model:** XGBoost Regressor trained on `flights.csv`

**API:** Flask REST API with schema-pinned feature order, full input validation, and structured error responses

**Run locally:**
```bash
cd flight_repo/flight_api
pip install -r requirements.txt
python app.py
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"from":"Sao Paulo (SP)","to":"Recife (PE)","flightType":"firstClass","agency":"Rainbow","time":1.8,"distance":1500}'
```

**Docker:**
```bash
cd flight_repo
bash docker/build_and_test_docker.sh
```

**Kubernetes:**
```bash
minikube start
eval $(minikube docker-env)
docker build -t flight-price-api:latest -f flight_api/Dockerfile flight_api
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
```

**Airflow:**
```bash
cp flight_repo/dags/flight_training_dag.py $AIRFLOW_HOME/dags/
airflow dags test flight_price_training_pipeline 2024-01-01
```

**Jenkins:** Point a pipeline job at `flight_repo/jenkins/Jenkinsfile`

---

## Module 2 — Gender Classification

**Problem:** Classify a user's gender from their name using a fine-tuned BERT model.

**Model:** BERT (bert-base-uncased) fine-tuned on the users dataset

**API:** Flask REST API returning predicted label + softmax confidence score

**Run locally:**
```bash
cd gender_github_repo/gender_api
pip install -r requirements.txt
python app.py
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

---

## Module 3 — Hotel Recommendation

**Problem:** Recommend hotels to users based on their booking history and hotel features.

**Model:** Hybrid recommender — SVD collaborative filtering + TF-IDF content-based filtering

**UI:** Streamlit app with three tabs (Collaborative, Hybrid, Similar Hotels)

**RMSE:** 0.8902

**Run locally:**
```bash
cd hotel_github_repo/hotel_app
pip install -r requirements.txt
streamlit run hotel_app.py
```

**Docker:**
```bash
cd hotel_github_repo
docker build -t hotel-recommender -f docker/Dockerfile .
docker run -p 8501:8501 hotel-recommender
```

---

## Datasets

| Dataset | Key Fields |
|---|---|
| flights.csv | travelCode, userCode, from, to, flightType, price, time, distance, agency, date |
| users.csv | code, company, name, gender, age |
| hotels.csv | travelCode, userCode, name, place, days, price, total, date |

Flights and hotels link to users via `userCode`. Flights and hotels link to each other via `travelCode`.

---

## MLOps Pipeline

```
flights.csv
    │
    ▼
Airflow DAG ──► preprocess ──► train XGBoost ──► evaluate ──► MLflow (log metrics + model)
                                                                      │
Jenkins CI ──► pytest ──► docker build ──► smoke test /health         │
                                    │                                 │
                                    ▼                                 ▼
                             Docker image ──► Kubernetes (2 replicas, health probes)
```
