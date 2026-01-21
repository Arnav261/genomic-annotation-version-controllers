"""
Configuration Management with Validation
Centralizes all configuration with environment variable validation
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseSettings, validator, Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Application
    APP_NAME: str = "Genomic Liftover System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")
    API_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./data/genomic_liftover.db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=5, env="DB_POOL_SIZE")
    
    # File Paths
    DATA_DIR: Path = Field(default=Path("./data"), env="DATA_DIR")
    CHAIN_DIR: Path = Field(default=Path("./data/chains"), env="CHAIN_DIR")
    REFERENCE_DIR: Path = Field(default=Path("./data/reference"), env="REFERENCE_DIR")
    VALIDATION_DIR: Path = Field(default=Path("./data/validation"), env="VALIDATION_DIR")
    MODEL_DIR: Path = Field(default=Path("./data/models"), env="MODEL_DIR")
    
    # Chain File URLs
    CHAIN_FILE_URLS: dict = {
        "hg19ToHg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        "hg38ToHg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz",
        "hg18ToHg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg18/liftOver/hg18ToHg19.over.chain.gz",
        "hg18ToHg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg18/liftOver/hg18ToHg38.over.chain.gz",
    }
    
    # Reference Database URLs
    NCBI_GENE_URL: str = "ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    ENSEMBL_REST_URL: str = "https://rest.ensembl.org"
    UCSC_MYSQL_HOST: str = "genome-mysql.soe.ucsc.edu"
    
    # Processing
    MAX_BATCH_SIZE: int = Field(default=10000, env="MAX_BATCH_SIZE")
    JOB_TIMEOUT_SECONDS: int = Field(default=3600, env="JOB_TIMEOUT")
    JOB_CLEANUP_DAYS: int = Field(default=7, env="JOB_CLEANUP_DAYS")
    WORKER_POOL_SIZE: int = Field(default=4, env="WORKER_POOL_SIZE")
    
    # Machine Learning
    ML_MODEL_VERSION: str = "1.0.0"
    ML_CONFIDENCE_THRESHOLD: float = 0.7
    ML_TRAINING_BATCH_SIZE: int = 32
    ML_VALIDATION_SPLIT: float = 0.2
    USE_GPU: bool = Field(default=False, env="USE_GPU")
    
    # Validation
    VALIDATION_TOLERANCE_BP: int = 100
    VALIDATION_SAMPLE_SIZE: Optional[int] = None
    RUN_BENCHMARKS_ON_STARTUP: bool = False
    
    # Caching
    ENABLE_CACHING: bool = Field(default=True, env="ENABLE_CACHING")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL")
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[Path] = None
    
    # Security
    API_KEY_REQUIRED: bool = Field(default=False, env="API_KEY_REQUIRED")
    API_KEYS: List[str] = Field(default=[], env="API_KEYS")
    
    # Feature Flags
    ENABLE_ENSEMBLE_LIFTOVER: bool = True
    ENABLE_BAYESIAN_CONFIDENCE: bool = True
    ENABLE_ACTIVE_LEARNING: bool = True
    ENABLE_EXPLAINABLE_AI: bool = True
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("API_KEYS", pre=True)
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator("DATA_DIR", "CHAIN_DIR", "REFERENCE_DIR", "VALIDATION_DIR", "MODEL_DIR")
    def create_directories(cls, v):
        """Ensure directories exist"""
        if v:
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return v
    
    def validate_paths(self) -> bool:
        """Validate all required paths exist and are accessible"""
        required_paths = [
            self.DATA_DIR,
            self.CHAIN_DIR,
            self.REFERENCE_DIR,
            self.MODEL_DIR
        ]
        
        for path in required_paths:
            if not path.exists():
                raise ValueError(f"Required path does not exist: {path}")
            if not os.access(path, os.R_OK | os.W_OK):
                raise ValueError(f"Insufficient permissions for path: {path}")
        
        return True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.validate_paths()
    return settings


# Global settings instance
settings = get_settings()