# Genomic Resonance — Public Deployment Guide

This document contains the production-ready deployment instructions and operational checks for the Genomic Resonance application.

IMPORTANT: follow the steps exactly and do them in order. This file is self-contained for deployment.

## 1. System prerequisites
- Ubuntu 20.04+ or macOS with Python 3.10
- At least 4 GB RAM and ~2 GB free disk for chain files + references
- Docker (optional but recommended for production container)

## 2. Directory layout (required)
- app/ (FastAPI application)
- app/data/chains/ — put UCSC chain files here (do NOT commit)
- app/data/reference/ — NCBI/Ensembl/ClinVar/GENCODE reference JSONs
- models/ — trained model artifacts (created at runtime)
- app/data/app.db — SQLite DB (created on first run)

## 3. Required external files (download before first run)
- UCSC chain files (example):
  - https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
  - decompress to `app/data/chains/hg19ToHg38.over.chain`
- Reference datasets (recommended for full validation):
  - NCBI RefSeq GFF/JSON: ftp://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/
  - ClinVar VCF: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/
  - GENCODE annotations: https://www.gencodegenes.org/
- Place reference JSONs in `app/data/reference/` with names expected by the ValidationEngine (see `app/services/validation_engine.py`).

## 4. Install Python dependencies (recommended)
1. Create venv:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. Install base deps:
   ```
   pip install -r app/requirements-base.txt
   ```
3. Install production extras (torch CPU wheel — choose appropriate instruction for your platform):
   ```
   # For CPU-only PyTorch (Linux example)
   pip install --index-url https://download.pytorch.org/whl/cpu/ torch
   pip install -r app/requirements-prod.txt
   ```
4. Install dev/test deps (optional):
   ```
   pip install -r app/requirements-dev.txt
   ```

## 5. Environment variables (create `.env` in repo root)
```
API_KEY_ADMIN=your_admin_api_key_here
CHAIN_DIR=app/data/chains
REF_DIR=app/data/reference
MODEL_DIR=models
DB_PATH=app/data/app.db
RUN_INTEGRATION=0
ALLOWED_ORIGINS=http://localhost:3000
MAX_UPLOAD_MB=200
```

## 6. Prepare chain files & references (scripts included)
- Chain helper:
  ```
  ./app/scripts/download_chains.sh hg19 hg38 app/data/chains
  gunzip app/data/chains/hg19ToHg38.over.chain.gz
  ```
- Reference helper (sample):
  ```
  python app/scripts/download_references.py  # inspect script before running
  ```

## 7. Initialize DB & seed API key
- The application will auto-seed `API_KEY_ADMIN` into the SQLite DB on startup if set in `.env`.
- To create additional keys, use the simple CLI (example):
  ```python
  from app.db import SessionLocal, APIKey
  s = SessionLocal()
  s.add(APIKey(key="another_key", name="user1", is_active=True))
  s.commit()
  s.close()
  ```

## 8. Run server (development)
- Local:
  ```
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
- Docker:
  ```
  docker build -t genomic-resonance:latest .
  docker run --rm -p 8000:8000 -e API_KEY_ADMIN=changeme -v $(pwd)/app/data:/app/app/data genomic-resonance:latest
  ```

## 9. Running tests
- Unit tests (no network):
  ```
  pytest -q
  ```
- Integration tests (require network/chain files):
  ```
  export RUN_INTEGRATION=1
  pytest -q
  ```

## 10. Production checklist (must-dos before public exposure)
1. Set `ALLOWED_ORIGINS` to your trusted frontends (do not allow `*`).
2. Use a secret manager for `API_KEY_ADMIN` and other secrets.
3. Add rate limiting (API gateway or middleware).
4. Front door TLS termination (platform-managed TLS recommended).
5. Logging aggregation & monitoring (centralized logs).
6. Run dependency scanner (pip-audit) and update insecure packages.
7. Rotate the `API_KEY_ADMIN` immediately after first deployment.

---
## 11. Troubleshooting quick-checks
- If liftover returns errors: verify chain file exists in `app/data/chains/` with the expected name and file size > 1 KB.
- If Ensembl calls fail: check outbound network, Ensembl REST availability, and recent HTTP 429 responses.
- If VCF conversion fails on large files: increase `MAX_UPLOAD_MB` and prefer background job processing or use chunked streaming.
