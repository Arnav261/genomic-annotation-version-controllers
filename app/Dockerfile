# Multi-stage image for smaller, reproducible builds
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build deps (if needed for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
# Install gunicorn and the modern uvicorn worker (uvicorn.workers is deprecated)
RUN pip install --no-cache-dir -r requirements.txt gunicorn uvicorn-worker

# Copy the rest of your app
COPY . /app

# Expose the port Render/Cloud Run expect
ENV PORT=8000
EXPOSE 8000

# Use Gunicorn with uvicorn worker for FastAPI
# NOTE: if your app module is not 'main:app', change it below.
CMD ["gunicorn", "-k", "uvicorn_worker.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "180"]
