"""
Database Models for Persistence
All entities that need to be stored permanently
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, Index, UniqueConstraint, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from app.database import Base


class JobStatus(PyEnum):
    """Job execution status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(PyEnum):
    """Types of jobs"""
    LIFTOVER_SINGLE = "liftover_single"
    LIFTOVER_BATCH = "liftover_batch"
    VCF_CONVERSION = "vcf_conversion"
    VALIDATION = "validation"
    BENCHMARKING = "benchmarking"


class Job(Base):
    """Job tracking for async operations"""
    __tablename__ = "jobs"
    
    id = Column(String(32), primary_key=True, index=True)
    job_type = Column(Enum(JobType), nullable=False, index=True)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED, index=True)
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime, default=func.now(), index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Parameters and results
    parameters = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # User tracking (optional)
    user_id = Column(String(64), nullable=True, index=True)
    api_key_hash = Column(String(64), nullable=True)
    
    # Relationships
    liftover_results = relationship(
        "LiftoverResult",
        back_populates="job",
        cascade="all, delete-orphan"
    )
    validation_records = relationship(
        "ValidationRecord",
        back_populates="job",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_job_status_created', 'status', 'created_at'),
        Index('idx_job_user_status', 'user_id', 'status'),
    )
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Calculate processing time"""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "progress_percent": round(self.progress_percent, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_seconds": self.processing_time_seconds,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "error_message": self.error_message
        }


class LiftoverResult(Base):
    """Individual liftover result"""
    __tablename__ = "liftover_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(32), ForeignKey("jobs.id"), nullable=False, index=True)
    
    # Input coordinates
    input_chrom = Column(String(16), nullable=False)
    input_pos = Column(Integer, nullable=False)
    input_build = Column(String(16), nullable=False)
    
    # Output coordinates
    output_chrom = Column(String(16), nullable=True)
    output_pos = Column(Integer, nullable=True)
    output_build = Column(String(16), nullable=False)
    
    # Quality metrics
    success = Column(Boolean, nullable=False)
    confidence_score = Column(Float, nullable=True)
    chain_score = Column(Float, nullable=True)
    
    # ML predictions
    ml_confidence = Column(Float, nullable=True)
    ensemble_confidence = Column(Float, nullable=True)
    
    # Flags
    ambiguous_mapping = Column(Boolean, default=False)
    in_repeat_region = Column(Boolean, default=False)
    in_sv_region = Column(Boolean, default=False)
    near_assembly_gap = Column(Boolean, default=False)
    
    # Additional data
    metadata = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    job = relationship("Job", back_populates="liftover_results")
    
    __table_args__ = (
        Index('idx_liftover_input', 'input_chrom', 'input_pos', 'input_build'),
        Index('idx_liftover_output', 'output_chrom', 'output_pos', 'output_build'),
        Index('idx_liftover_success', 'success'),
    )


class ValidationRecord(Base):
    """Validation test record"""
    __tablename__ = "validation_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(32), ForeignKey("jobs.id"), nullable=True, index=True)
    
    # Gene information
    gene_id = Column(String(64), nullable=False, index=True)
    gene_symbol = Column(String(64), nullable=False, index=True)
    source_database = Column(String(32), nullable=False)
    
    # Expected vs actual
    expected_chrom = Column(String(16), nullable=False)
    expected_pos = Column(Integer, nullable=False)
    expected_build = Column(String(16), nullable=False)
    
    actual_chrom = Column(String(16), nullable=True)
    actual_pos = Column(Integer, nullable=True)
    
    # Metrics
    success = Column(Boolean, nullable=False)
    error_bp = Column(Integer, nullable=True)
    error_percent = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Context
    region_type = Column(String(32), nullable=True)
    gene_size = Column(Integer, nullable=True)
    
    # Flags
    in_repeat = Column(Boolean, default=False)
    in_sv = Column(Boolean, default=False)
    near_gap = Column(Boolean, default=False)
    
    # Timing
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Additional data
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    job = relationship("Job", back_populates="validation_records")
    
    __table_args__ = (
        Index('idx_validation_gene', 'gene_symbol', 'source_database'),
        Index('idx_validation_success', 'success'),
    )


class BenchmarkResult(Base):
    """Benchmark comparison results"""
    __tablename__ = "benchmark_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Benchmark metadata
    benchmark_name = Column(String(128), nullable=False, index=True)
    tool_name = Column(String(64), nullable=False)
    tool_version = Column(String(32), nullable=True)
    
    # Dataset information
    dataset_name = Column(String(128), nullable=False)
    dataset_size = Column(Integer, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Error metrics
    mean_error_bp = Column(Float, nullable=True)
    median_error_bp = Column(Float, nullable=True)
    p95_error_bp = Column(Float, nullable=True)
    max_error_bp = Column(Float, nullable=True)
    
    # Performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    throughput_items_per_second = Column(Float, nullable=True)
    
    # Statistical tests
    statistical_significance = Column(JSON, nullable=True)
    
    # Detailed results
    detailed_metrics = Column(JSON, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_benchmark_name', 'benchmark_name', 'tool_name'),
        UniqueConstraint(
            'benchmark_name', 'tool_name', 'dataset_name', 'created_at',
            name='uq_benchmark_run'
        ),
    )


class MLModel(Base):
    """Machine learning model registry"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model metadata
    model_name = Column(String(128), nullable=False, index=True)
    model_version = Column(String(32), nullable=False)
    model_type = Column(String(64), nullable=False)
    
    # Training information
    training_dataset_size = Column(Integer, nullable=True)
    training_date = Column(DateTime, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    
    # Performance metrics
    train_accuracy = Column(Float, nullable=True)
    train_auc = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    val_auc = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_auc = Column(Float, nullable=True)
    
    # Model artifacts
    model_path = Column(String(256), nullable=False)
    scaler_path = Column(String(256), nullable=True)
    feature_names = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    created_by = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_model_name_version', 'model_name', 'model_version'),
        UniqueConstraint('model_name', 'model_version', name='uq_model_version'),
    )


class ChainFile(Base):
    """Chain file registry"""
    __tablename__ = "chain_files"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Chain file identification
    name = Column(String(64), nullable=False, unique=True, index=True)
    from_build = Column(String(16), nullable=False)
    to_build = Column(String(16), nullable=False)
    
    # File information
    file_path = Column(String(256), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    checksum_md5 = Column(String(32), nullable=True)
    
    # Download metadata
    source_url = Column(String(512), nullable=True)
    download_date = Column(DateTime, nullable=True)
    
    # Usage statistics
    times_used = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Status
    is_available = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_chain_builds', 'from_build', 'to_build'),
    )