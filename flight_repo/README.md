# Flight Price Prediction — MLOps Capstone (Regression Module)

End-to-end ML system that predicts flight ticket price from the `flights.csv`
dataset and serves it through a production-style REST API, containerized with
Docker, deployable to Kubernetes, orchestrated with Airflow, tracked with
MLflow, and built via a Jenkins CI/CD pipeline.

## Repository structure
```
.
├── flight_api/
│   ├── app.py                 # Flask REST API (schema-pinned, validated)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── artifacts/             # model + label encoders + feature_schema.json
├── k8s/
│   ├── deployment.yaml        # 2 replicas, health probes, resource limits
│   └── service.yaml           # NodePort, selector matches deployment labels
├── dags/
│   └── flight_training_dag.py # real Airflow training pipeline (5 tasks)
├── jenkins/
│   └── Jenkinsfile            # checkout -> test -> build -> smoke test
├── tests/
│   └── test_api.py            # pytest suite run inside CI
├── docker/
│   └── build_and_test_docker.sh
└── notebooks/
    └── flight_price_prediction.ipynb
```

## 1. Run the API locally
```bash
cd flight_api
pip install -r requirements.txt
python app.py            # serves on :8000
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"from":"Sao Paulo (SP)","to":"Recife (PE)","flightType":"firstClass","agency":"Rainbow","time":1.8,"distance":1500}'
```

## 2. Docker
```bash
bash docker/build_and_test_docker.sh
```

## 3. Kubernetes (minikube)
```bash
minikube start
eval $(minikube docker-env)
docker build -t flight-price-api:latest -f flight_api/Dockerfile flight_api
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
minikube service flight-price-service --url
```

## 4. Airflow
Copy `dags/flight_training_dag.py` into `$AIRFLOW_HOME/dags/`, place `flights.csv`
in `$AIRFLOW_HOME/data/`, then:
```bash
airflow dags test flight_price_training_pipeline 2024-01-01
```

## 5. Jenkins
Create a Pipeline job pointing at this repo using `jenkins/Jenkinsfile`.

## 6. MLflow
Metrics and the model are logged to the `FlightFarePredictionExp` experiment;
launch `mlflow ui` to inspect runs.

## API contract
`POST /predict` — JSON body with all six fields (order does not matter):
`from`, `to`, `flightType`, `agency` (categorical) and `time`, `distance` (numeric).
Returns `{"predicted_price": <float>, ...}`. Invalid input returns HTTP 400.
