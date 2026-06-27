FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY hotel_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hotel_app/ ./hotel_app/
EXPOSE 8501
CMD ["streamlit","run","hotel_app/hotel_app.py","--server.port=8501","--server.address=0.0.0.0"]
