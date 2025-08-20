
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

