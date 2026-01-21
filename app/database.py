"""
Simple SQLAlchemy SQLite models and helpers for jobs, API keys, labels.
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from app.config import settings
import os

DB_URL = f"sqlite:///{settings.DB_PATH}"
os.makedirs(settings.DATA_DIR, exist_ok=True)

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, nullable=False)
    name = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, nullable=False)
    job_type = Column(String(64), nullable=False)
    status = Column(String(32), default="queued")
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    results = Column(JSON, nullable=True)
    errors = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(String(128), nullable=False)
    label = Column(String(64), nullable=False)
    annotator = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)


# Initialize DB on import if not present
init_db()