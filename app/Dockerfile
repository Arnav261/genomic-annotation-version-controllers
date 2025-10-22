FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("Model pre-download skipped or failed:", e)
PY

COPY . .

EXPOSE 8000  

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker app.main:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120"]


# Multi-stage Dockerfile
FROM python:3.11-slim as builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip --no-cache-dir install -r /app/app/requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app
RUN mkdir -p /app/data/chains /app/model_data /app/model_cache
ENV LIFTOVER_CHAIN_DIR=/app/data/chains
ENV MODEL_CACHE_DIR=/app/model_cache
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]