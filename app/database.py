"""
Database Configuration with Connection Pooling
Provides SQLAlchemy session management with proper cleanup
"""
from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Create engine with appropriate pooling
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite-specific configuration
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DEBUG
    )
else:
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=10,
        pool_pre_ping=True,
        echo=settings.DEBUG
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()


# Event listeners for connection management
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Configure connection on connect"""
    if settings.DATABASE_URL.startswith("sqlite"):
        # Enable foreign keys for SQLite
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Verify connection on checkout"""
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("SELECT 1")
        except Exception:
            raise pool.InvalidRequestError()
        finally:
            cursor.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints
    Provides database session with automatic cleanup
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    Use in non-FastAPI code
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from app.models.database_models import (
        Job, LiftoverResult, ValidationRecord,
        BenchmarkResult, MLModel, ChainFile
    )
    
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def cleanup_old_jobs(days: int = None):
    """Clean up old completed jobs"""
    from datetime import datetime, timedelta
    from app.models.database_models import Job
    
    days = days or settings.JOB_CLEANUP_DAYS
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_context() as db:
        deleted_count = db.query(Job).filter(
            Job.completed_at < cutoff_date,
            Job.status.in_(["completed", "failed"])
        ).delete()
        
        logger.info(f"Cleaned up {deleted_count} old jobs")
        return deleted_count