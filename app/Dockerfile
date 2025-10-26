# Multi-stage Dockerfile (expects requirements at app/requirements.txt)
FROM python:3.11-slim as builder
ENV DEBIAN_FRONTEND=noninteractive

# Install build deps we may need for wheels (faiss/hdbscan etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates cmake libopenblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching. Path in repo: app/requirements.txt
COPY app/requirements.txt /app/requirements.txt

# Install python packages in builder (speed up later stages)
RUN pip install --upgrade pip setuptools wheel \
    && pip --no-cache-dir install -r /app/requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Create persistent directories used by app at runtime
RUN mkdir -p /app/data/chains /app/model_data /app/model_cache

# Environment defaults
ENV LIFTOVER_CHAIN_DIR=/app/data/chains
ENV MODEL_CACHE_DIR=/app/model_cache
ENV VECTOR_INDEX_PATH=/app/model_data/faiss.index
ENV VECTOR_META_DB=/app/model_data/faiss_meta.db

EXPOSE 8000


# Use gunicorn+uvicorn for production; uvicorn for local testing
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]

# Use env $PORT provided by Render; fallback to 8000 for local testing.
CMD ["sh", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120"]