"""
Genomic Coordinate Liftover Service - Resonance
Professional Research-Grade Bioinformatics Platform
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse, PlainTextResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
import uuid
import os
import json
import csv
from io import StringIO
import numpy as np

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config and database
from app.config import settings
from app.database import SessionLocal, APIKey, Job

# Import services
try:
    from app.services.real_liftover import RealLiftoverService
    from app.services.ensembl_liftover import EnsemblLiftover
    from app.services.feature_extractor import FeatureExtractor
    from app.services.confidence_predictor import ConfidencePredictor
    from app.services.vcf_converter import VCFConverter
    HAS_SERVICES = True
except ImportError as e:
    logger.error(f"Failed to import services: {e}")
    HAS_SERVICES = False

SERVICES: Dict[str, Any] = {}
startup_time = time.time()
job_storage: Dict[str, Any] = {}

# Initialize FastAPI
app = FastAPI(
    title="Resonance - Genomic Coordinate Liftover",
    description="Professional research-grade genomic coordinate conversion platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_services():
    """Initialize all services at startup"""
    global SERVICES
    
    logger.info("Initializing services...")
    
    if not HAS_SERVICES:
        logger.error("Services not available - imports failed")
        return False
    
    try:
        # Initialize chain-based liftover
        try:
            SERVICES["chain_liftover"] = RealLiftoverService(chain_dir=str(settings.CHAIN_DIR))
            logger.info("✓ Chain liftover initialized")
        except Exception as e:
            logger.warning(f"Chain liftover failed: {e}")
            SERVICES["chain_liftover"] = None
        
        # Initialize Ensembl liftover with chain fallback
        try:
            SERVICES["liftover"] = EnsemblLiftover(
                fallback=SERVICES.get("chain_liftover")
            )
            logger.info("✓ Ensembl liftover initialized")
        except Exception as e:
            logger.warning(f"Ensembl liftover failed, using chain only: {e}")
            SERVICES["liftover"] = SERVICES.get("chain_liftover")
        
        # Initialize feature extractor
        try:
            SERVICES["feature_extractor"] = FeatureExtractor(
                data_dir=str(settings.REF_DIR)
            )
            logger.info("✓ Feature extractor initialized")
        except Exception as e:
            logger.warning(f"Feature extractor failed: {e}")
            SERVICES["feature_extractor"] = None
        
        # Initialize confidence predictor
        try:
            cp = ConfidencePredictor(model_path=str(settings.MODEL_DIR / "confidence_model.pkl"))
            SERVICES["confidence_predictor"] = cp
            logger.info(f"✓ Confidence predictor initialized (trained: {cp.is_trained})")
        except Exception as e:
            logger.warning(f"Confidence predictor failed: {e}")
            SERVICES["confidence_predictor"] = None
        
        # Initialize VCF converter
        if SERVICES.get("liftover"):
            try:
                SERVICES["vcf_converter"] = VCFConverter(SERVICES["liftover"])
                logger.info("✓ VCF converter initialized")
            except Exception as e:
                logger.warning(f"VCF converter failed: {e}")
                SERVICES["vcf_converter"] = None
        else:
            SERVICES["vcf_converter"] = None
        
        # Check if we have at least basic liftover
        if SERVICES.get("liftover"):
            logger.info("✓ Core services operational")
            return True
        else:
            logger.error("✗ Core liftover service failed to initialize")
            return False
            
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


# Job management
class BatchJob:
    """Background job tracker"""
    def __init__(self, job_id: str, total_items: int, job_type: str):
        self.job_id = job_id
        self.total_items = total_items
        self.processed_items = 0
        self.status = "queued"
        self.job_type = job_type
        self.results = []
        self.start_time = datetime.now()
        self.end_time = None
        self.errors = []
        self.metadata = {}
        self.warnings = []
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "job_id": self.job_id,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "status": self.status,
            "job_type": self.job_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


@app.get("/", response_class=HTMLResponse)
def landing_page():
    db = SessionLocal()
    try:
        active_jobs = db.query(Job).filter(
            Job.status.in_(["queued", "processing"])
        ).count()
    except:
        active_jobs = 0
    finally:
        db.close()

    return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Resonance – Genomic Liftover</title>

<style>
:root {{
    --navy: #001f3f;
    --light: #f7f9fb;
    --border: #d0d7de;
}}

body {{
    margin: 0;
    font-family: "Times New Roman", Georgia, serif;
    background: var(--light);
    color: #000;
}}

header {{
    background: var(--navy);
    color: #fff;
    padding: 2.5rem 3rem;
}}

header h1 {{
    margin: 0;
    font-size: 2.6rem;
    letter-spacing: 0.05em;
}}

header p {{
    margin-top: 0.5rem;
    opacity: 0.9;
}}

main {{
    max-width: 1200px;
    margin: auto;
    padding: 3rem;
}}

.section {{
    background: #fff;
    border: 2px solid var(--navy);
    padding: 2rem;
    margin-bottom: 2.5rem;
}}

.section h2 {{
    margin-top: 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}}

label {{
    display: block;
    margin-top: 1rem;
    font-weight: bold;
}}

input {{
    width: 100%;
    padding: 0.6rem;
    margin-top: 0.3rem;
}}

button {{
    margin-top: 1.5rem;
    padding: 0.7rem 1.8rem;
    background: var(--navy);
    color: white;
    border: none;
    cursor: pointer;
    font-size: 1rem;
}}

pre {{
    background: #f5f5f5;
    border: 1px solid var(--border);
    padding: 1rem;
    margin-top: 1.5rem;
    white-space: pre-wrap;
    font-family: monospace;
}}
</style>
</head>

<body>

<header>
<h1>RESONANCE</h1>
<p>Genomic Coordinate Liftover & Validation Platform</p>
</header>

<main>

<section class="section">
<h2>System Status</h2>
<ul>
<li>Active jobs: {active_jobs}</li>
<li>ML confidence: {"Available" if SERVICES.get("confidence_predictor") else "Unavailable"}</li>
<li>VCF processing: {"Enabled" if SERVICES.get("vcf_converter") else "Disabled"}</li>
</ul>
</section>

<section class="section">
<h2>Live Coordinate Conversion</h2>

<label>Chromosome</label>
<input id="chrom" value="chr17">

<label>Position</label>
<input id="pos" value="41196312">

<button onclick="run()">Convert</button>

<pre id="out"></pre>
</section>

</main>

<script>
async function run() {{
    const chrom = document.getElementById("chrom").value;
    const pos = document.getElementById("pos").value;

    const r = await fetch(
        `/liftover/single?chrom=${{chrom}}&pos=${{pos}}&from_build=hg19&to_build=hg38`,
        {{ method: 'POST' }}
    );
    document.getElementById("out").textContent =
        JSON.stringify(await r.json(), null, 2);
}}
</script>

</body>
</html>
""")


@app.get("/health")
def health():
    db_ok = True
    try:
        s = SessionLocal()
        s.execute("SELECT 1")
        s.close()
    except Exception:
        db_ok = False

    return {
        "status": "ok" if SERVICES.get("liftover") and db_ok else "degraded",
        "uptime_seconds": int(time.time() - startup_time),
        "services": {
            "liftover": SERVICES.get("liftover") is not None,
            "vcf": SERVICES.get("vcf_converter") is not None,
            "ml": SERVICES.get("confidence_predictor") is not None,
        }
    }


@app.post("/liftover/single")
async def liftover_single(
    chrom: str = Query(...),
    pos: int = Query(..., ge=1),
    from_build: str = Query("hg19"),
    to_build: str = Query("hg38"),
    strand: str = Query("+"),
    include_ml: bool = Query(True)
):
    """Convert single genomic coordinate"""
    if not SERVICES.get('liftover'):
        raise HTTPException(status_code=503, detail="Liftover service unavailable")
    
    try:
        result = SERVICES['liftover'].convert_coordinate(chrom, pos, from_build, to_build, strand)
        
        if include_ml and SERVICES.get('feature_extractor') and SERVICES.get('confidence_predictor'):
            try:
                features = SERVICES['feature_extractor'].extract_features(
                    chrom, pos, from_build, to_build, result
                )
                
                ml_confidence = SERVICES['confidence_predictor'].predict_confidence(features.to_array())
                
                # Ensure ml_confidence is a proper float between 0 and 1
                ml_confidence = float(ml_confidence)
                ml_confidence = max(0.0, min(1.0, ml_confidence))
                
                interpretation = SERVICES['confidence_predictor'].interpret_confidence(ml_confidence)
                
                result['ml_analysis'] = {
                    'confidence_score': ml_confidence,
                    'interpretation': interpretation,
                    'model_type': 'gradient_boosting',
                    'model_status': 'operational'
                }
            except Exception as e:
                logger.error(f"ML confidence failed: {e}")
                result['ml_analysis'] = {
                    'error': str(e), 
                    'model_status': 'error',
                    'confidence_score': None,
                    'interpretation': None
                }
        
        # Ensure chain confidence is also properly formatted
        if 'confidence' in result and result['confidence'] is not None:
            result['confidence'] = float(result['confidence'])
            result['confidence'] = max(0.0, min(1.0, result['confidence']))
        
        return result
        
    except Exception as e:
        logger.error(f"Liftover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    """Get job processing status"""
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
    
    response = {
        "job_id": job_id,
        "job_type": job.job_type,
        "status": job.status,
        "progress_percent": round(progress, 2),
        "processed_items": job.processed_items,
        "total_items": job.total_items,
        "start_time": job.start_time.isoformat(),
        "errors": job.errors,
        "warnings": job.warnings
    }
    
    if job.status == "completed":
        response["end_time"] = job.end_time.isoformat() if job.end_time else None
        response["processing_time_seconds"] = (
            (job.end_time - job.start_time).total_seconds()
            if job.end_time else None
        )
        response["metadata"] = job.metadata
        response["results_count"] = len(job.results)
        response["export_options"] = {
            "json": f"/export/{job_id}/json",
            "csv": f"/export/{job_id}/csv"
        }
    
    return response


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("=" * 80)
    logger.info("Resonance - Genomic Coordinate Liftover Service v1.0.0")
    logger.info("=" * 80)
    
    # Initialize services
    success = initialize_services()
    
    # Log service status
    for service_name, service in SERVICES.items():
        status = 'Available' if service else 'Unavailable'
        logger.info(f"  {service_name}: {status}")
    
    if success:
        logger.info("✓ All critical systems operational")
    else:
        logger.warning("⚠ Operating in limited mode")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)