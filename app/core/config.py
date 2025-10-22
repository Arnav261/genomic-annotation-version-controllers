
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENSEMBL_REST_BASE: str = "https://rest.ensembl.org"
    ENSEMBL_REST_GRCH37_BASE: str = "https://grch37.rest.ensembl.org"
    ENSEMBL_REQUEST_TIMEOUT: int = 20
    ENSEMBL_USER_AGENT: str = "GAVC/1.0 (FastAPI)"

    class Config:
        env_file = ".env"
        extra = "ignore"   

settings = Settings()

"""
Configuration constants and helpers. Edit environment variables in production.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Where to find UCSC chain files for liftover
LIFTOVER_CHAIN_DIR = os.environ.get("LIFTOVER_CHAIN_DIR", str(ROOT / "data" / "chains"))

# Where HF and SBERT model caches should be stored (if desired)
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", str(ROOT / "model_cache"))

# Default SBERT model key
DEFAULT_SBET_MODEL_KEY = os.environ.get("DEFAULT_SBET_MODEL_KEY", "sbert")

# Production server settings
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
WORKERS = int(os.environ.get("WEB_WORKERS", "2"))

# Logging level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")