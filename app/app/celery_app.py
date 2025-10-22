from celery import Celery
import os

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery(
    "genomic_annotation",
    broker=REDIS_URL,
    backend=CELERY_BACKEND,
)

# Optional: load config from environment or a module
celery_app.conf.update(
    task_track_started=True,
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    result_extended=True,
    worker_max_tasks_per_child=100,
)