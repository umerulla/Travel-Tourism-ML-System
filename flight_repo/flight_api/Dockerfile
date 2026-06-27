# Lean, production-style image for the Flight Price Prediction API
# Base: python:3.11-slim keeps image size small (~150MB vs ~900MB for full python)
FROM python:3.11-slim

# Avoid .pyc files / enable unbuffered logs (important for Docker log streaming)
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

WORKDIR /app

# Install deps FIRST — this layer is cached separately from app code.
# If only app.py changes, Docker reuses this layer and skips pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model artifacts
COPY app.py .
COPY artifacts/ ./artifacts/

# Run as non-root user for security — never run containers as root in production
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check — Docker marks container unhealthy if /health fails 3 times
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

# Gunicorn = production WSGI server (2 workers). Never use Flask dev server in production.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "app:app"]
