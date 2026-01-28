"""
Application configuration and environment-backed settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "app" / "data")
    CHAIN_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "app" / "data" / "chains")
    REF_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "app" / "data" / "reference")
    REFERENCE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "app" / "data" / "reference")
    MODEL_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    DB_PATH: str = Field(default="app/data/app.db")

    # Legacy/compatibility fields (from .env)
    DATABASE_URL: str = Field(default="sqlite:///./app/data/gavc.db")
    UPLOAD_DIR: str = Field(default="./app/data/uploads")
    LIFTOVER_MODE: str = Field(default="ucsc")
    ENSEMBL_BASE_URL: str = Field(default="https://rest.ensembl.org")
    ENSEMBL_GRCH37_BASE_URL: str = Field(default="https://grch37.rest.ensembl.org")

    # Ensembl
    ENSEMBL_REST_BASE: str = Field(default="https://rest.ensembl.org")
    ENSEMBL_REST_GRCH37_BASE: str = Field(default="https://grch37.rest.ensembl.org")
    ENSEMBL_REQUEST_TIMEOUT: int = Field(default=10)
    ENSEMBL_USER_AGENT: str = Field(default="GAVC/1.0 (FastAPI)")

    # Liftover chain file URLs mapping (can be extended)
    CHAIN_FILE_URLS: dict = Field(
        default_factory=lambda: {
            "hg19ToHg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
            "hg38ToHg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz",
        }
    )

    # API & Security
    API_KEY_ADMIN: str | None = Field(default=None, env="API_KEY_ADMIN")
    MAX_UPLOAD_MB: int = Field(default=200)
    RUN_INTEGRATION: bool = Field(default=False, env="RUN_INTEGRATION")

    # ML settings
    MODEL_DEFAULT_FILENAME: str = Field(default="confidence_model.pt")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()