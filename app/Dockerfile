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
