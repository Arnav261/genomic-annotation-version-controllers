import multiprocessing

# Point Gunicorn to your FastAPI app in main.py
wsgi_app = "app.main:app"


workers = int(multiprocessing.cpu_count() * 0.5) or 1
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:10000"   # Render expects dynamic port, defaults to 10000
timeout = 180
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
