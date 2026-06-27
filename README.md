# Gender Classification — BERT (Classification Module)

Fine-tuned `bert-base-uncased` predicts gender from a user's name, served via a
Flask REST API, containerized with Docker, deployable to Kubernetes, tracked
with MLflow, and built via Jenkins CI/CD.

## Structure
```
gender_api/   app.py, requirements.txt, Dockerfile, bert_gender_classifier/
k8s/          deployment.yaml, service.yaml
tests/        test_api.py
jenkins/      Jenkinsfile
docker/       build_and_test_docker.sh
notebooks/    Gender_Classification_Model.ipynb
```

## Run locally
```bash
cd gender_api && pip install -r requirements.txt && python app.py
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"name":"Priya"}'
# -> {"name":"Priya","predicted_gender":"female","confidence":0.97}
```
Batch: `{"names":["Priya","Rahul"]}`. Invalid input returns HTTP 400.

## Docker / K8s / Jenkins / MLflow
See `docker/build_and_test_docker.sh`, `k8s/*.yaml`, `jenkins/Jenkinsfile`.
MLflow uses a SQLite backend (`sqlite:///mlflow.db`); view with `mlflow ui`.
