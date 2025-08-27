# Deploying the Genomic Annotation API

This kit gives you a reproducible way to deploy your FastAPI app to **Render** (or **Google Cloud Run**) with Docker.

## 1) Prepare your repo

- Ensure your ASGI app is exposed as `main:app` (or update Docker CMD).
- Pin `requirements.txt` and include `uvicorn-worker` and `gunicorn`.
- Add CORS in your app using `CORS_ALLOW_ORIGINS`.
- Add a `/healthz` endpoint returning `{"status":"ok"}`.

## 2) Render (one-click-ish)

- Commit `Dockerfile` and `render.yaml` to the repo root.
- Create a **Web Service** from GitHub on Render.
- Render will build the Dockerfile and run `gunicorn`.
- Set env vars (`CORS_ALLOW_ORIGINS`, etc.).
- (Optional) Use **Upstash Redis** for job queue/cache and set `UPSTASH_*` env vars.

## 3) Cloud Run (container-first)

- Build & push your image:
  ```bash
  gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT/REPO/genomic-annotation:latest
  ```
- Deploy:
  ```bash
  gcloud run deploy genomic-annotation --image us-central1-docker.pkg.dev/PROJECT/REPO/genomic-annotation:latest --region us-central1 --allow-unauthenticated --port 8000
  ```

## 4) Static front-end

Use the `/web` folder in this kit for a minimal HTML UI (upload, status poll). Deploy it as a **Static Site** on Render or Netlify and point it to your API URL.

## 5) Production checklist (short)

- ✅ HTTPS + Custom domain
- ✅ CORS restricted to your domain(s)
- ✅ Limits on upload size & concurrency
- ✅ Background jobs for long tasks
- ✅ Request logging + error tracking
- ✅ Reproducible data (versioned chain/GTF with checksums)
- ✅ Privacy statement: no PHI; auto-delete uploads after N days
