#!/usr/bin/env bash
set -euo pipefail
IMAGE="gender-api:latest"
docker build -t "$IMAGE" -f gender_api/Dockerfile gender_api
docker run -d --rm -p 8000:8000 --name gender-api "$IMAGE"
sleep 25
curl -s http://localhost:8000/health; echo
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"name":"Priya"}'; echo
docker stop gender-api
