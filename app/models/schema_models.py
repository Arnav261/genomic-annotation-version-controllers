"""
Pydantic Models for API Request/Response Validation
Ensures type safety and automatic documentation
"""
from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class AssemblyBuild(str, Enum):
    """Supported genome assemblies"""
    HG18 = "hg18"
    HG19 = "hg19"
    HG38 = "hg38"
    GRCH37 = "GRCh37"
    GRCH38 = "GRCh38"


class Strand(str, Enum):
    """DNA strand"""
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "."


class JobStatusEnum(str, Enum):
    """Job status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfidenceLevel(str, Enum):
    """Confidence interpretation"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class CoordinateInput(BaseModel):
    """Single genomic coordinate"""
    chrom: str = Field(..., description="Chromosome (e.g., chr17 or 17)")
    pos: int = Field(..., gt=0, description="Position (1-based)")
    strand: Optional[Strand] = Field(default=Strand.PLUS, description="Strand")
    
    @validator("chrom")
    def normalize_chromosome(cls, v):
        """Normalize chromosome name"""
        v = v.strip()
        if not v.lower().startswith("chr") and v not in ["X", "Y", "M", "MT"]:
            if v.isdigit() or v in ["X", "Y", "M"]:
                v = f"chr{v}"
        return v


class LiftoverRequest(BaseModel):
    """Single coordinate liftover request"""
    coordinate: CoordinateInput
    from_build: AssemblyBuild = Field(..., description="Source assembly")
    to_build: AssemblyBuild = Field(..., description="Target assembly")
    include_ml_confidence: bool = Field(default=True, description="Include ML confidence prediction")
    include_bayesian: bool = Field(default=False, description="Include Bayesian confidence intervals")
    use_ensemble: bool = Field(default=True, description="Use ensemble liftover method")
    
    @root_validator(skip_on_failure=True)
    def validate_builds(cls, values):
        """Ensure from and to builds are different"""
        if values.get("from_build") == values.get("to_build"):
            raise ValueError("Source and target builds must be different")
        return values


class BatchLiftoverRequest(BaseModel):
    """Batch liftover request"""
    coordinates: List[CoordinateInput] = Field(..., min_items=1, max_items=10000)
    from_build: AssemblyBuild
    to_build: AssemblyBuild
    include_ml_confidence: bool = True
    include_bayesian: bool = False
    use_ensemble: bool = True


class ConfidenceMetrics(BaseModel):
    """Confidence prediction metrics"""
    score: float = Field(..., ge=0.0, le=1.0)
    level: ConfidenceLevel
    interpretation: str
    recommendation: str
    clinical_threshold: bool
    research_threshold: bool
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[Dict[str, float]] = None


class BayesianConfidence(BaseModel):
    """Bayesian confidence intervals"""
    mean: float
    median: float
    std: float
    credible_interval_95_lower: float
    credible_interval_95_upper: float
    probability_correct: float = Field(..., ge=0.0, le=1.0)


class LiftoverResponse(BaseModel):
    """Single liftover response"""
    success: bool
    original: CoordinateInput
    original_build: str
    lifted_chrom: Optional[str] = None
    lifted_pos: Optional[int] = None
    lifted_build: str
    lifted_strand: Optional[Strand] = None
    chain_score: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    ml_confidence: Optional[ConfidenceMetrics] = None
    bayesian_confidence: Optional[BayesianConfidence] = None
    ensemble_confidence: Optional[float] = None
    ambiguous: bool = False
    alternative_mappings: Optional[List[Dict[str, Any]]] = None
    in_repeat_region: bool = False
    in_sv_region: bool = False
    near_assembly_gap: bool = False
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    method: str


class JobResponse(BaseModel):
    """Job creation response"""
    job_id: str
    status: JobStatusEnum
    job_type: str
    total_items: int
    created_at: datetime
    status_url: str
    results_url: Optional[str] = None
    estimated_completion_seconds: Optional[float] = None


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatusEnum
    job_type: str
    total_items: int
    processed_items: int
    failed_items: int
    progress_percent: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    results_available: bool = False
    results_url: Optional[str] = None
    download_urls: Optional[Dict[str, str]] = None
    summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Single validation result"""
    gene_id: str
    gene_symbol: str
    source_database: str
    expected_chrom: str
    expected_pos: int
    actual_chrom: Optional[str]
    actual_pos: Optional[int]
    success: bool
    error_bp: Optional[int]
    error_percent: Optional[float]
    confidence_score: Optional[float]
    region_type: Optional[str]
    in_repeat: bool = False
    in_sv: bool = False
    near_gap: bool = False
    notes: Optional[str]


class ValidationReport(BaseModel):
    """Complete validation report"""
    total_tests: int
    successful: int
    failed: int
    success_rate: float
    mean_error_bp: float
    median_error_bp: float
    p95_error_bp: float
    max_error_bp: float
    std_error_bp: float
    per_chromosome: Dict[str, Dict[str, Any]]
    per_region_type: Dict[str, Dict[str, Any]]
    per_database: Dict[str, Dict[str, Any]]
    confidence_calibration: Dict[str, Dict[str, Any]]
    statistical_significance: Optional[Dict[str, Any]] = None
    methodology: Dict[str, str]
    detailed_results: Optional[List[ValidationResult]] = None


class BenchmarkComparison(BaseModel):
    """Benchmark comparison result"""
    tool_name: str
    tool_version: Optional[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mean_error_bp: float
    median_error_bp: float
    p95_error_bp: float
    processing_time_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    significantly_different: Optional[bool] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


class BenchmarkReport(BaseModel):
    """Complete benchmark report"""
    benchmark_name: str
    dataset_name: str
    dataset_size: int
    tools: List[BenchmarkComparison]
    best_accuracy: str
    best_speed: str
    best_memory: str
    statistical_tests: Dict[str, Any]
    detailed_metrics: Dict[str, Any]
    generated_at: datetime


class HealthResponse(BaseModel):
    """System health check response"""
    status: str
    version: str
    timestamp: datetime
    uptime_seconds: float
    services: Dict[str, bool]
    database_connected: bool
    cache_available: bool
    max_batch_size: int
    ml_models_loaded: int
    chain_files_available: int
    features_enabled: Dict[str, bool]