#!/usr/bin/env bash
# Run locally (Docker Desktop / any Docker host). Screenshot the output for your report.
set -euo pipefail
IMAGE="flight-price-api:latest"
echo "==> Building image"
docker build -t "$IMAGE" -f flight_api/Dockerfile flight_api
echo "==> Running container"
docker run -d --rm -p 8000:8000 --name flight-api "$IMAGE"
sleep 5
echo "==> Health check"; curl -s http://localhost:8000/health; echo
echo "==> Prediction"
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"from":"Sao Paulo (SP)","to":"Recife (PE)","flightType":"firstClass","agency":"Rainbow","time":1.8,"distance":1500}'; echo
echo "==> Stopping container"; docker stop flight-api
