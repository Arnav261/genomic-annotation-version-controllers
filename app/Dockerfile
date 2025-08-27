FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OPTIONAL: Pre-download an embedding model used by your resolver (edit if different)
# This prevents slow cold-starts and avoids download failures in runtime.
RUN python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("Model pre-download skipped or failed:", e)
PY

COPY . .

EXPOSE 10000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:application", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120"]
