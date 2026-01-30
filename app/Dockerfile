# Minimal production Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for optional bio packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install pip separately and requirements
RUN pip install --upgrade pip
# Install base + prod
RUN pip install -r requirements/base.txt
RUN pip install -r requirements/prod.txt

# Create a non-root user
RUN useradd --create-home appuser
USER appuser
ENV HOME=/home/appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]