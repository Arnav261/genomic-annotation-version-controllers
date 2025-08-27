# Optional Gunicorn config file (if you prefer a file over CLI flags)
import multiprocessing

workers = int(multiprocessing.cpu_count() * 0.5) or 1
worker_class = "uvicorn_worker.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = 180
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
