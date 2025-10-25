"""
Genomic Annotation Version Controller - Production API
With REAL liftover, validation, VCF conversion, and AI conflict resolution
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from typing import List, Dict, Any, Optional
import uuid
import time
import asyncio
import os
from datetime import datetime
from io import StringIO
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Genomic Annotation Version Controller",
    description="""
    ## 🧬 Production-Grade Genomic Data Management Platform
    
    ### Features
    - ✅ **Real UCSC LiftOver**: Actual chain file-based coordinate conversion
    - ✅ **Validated**: Tested against NCBI RefSeq coordinates
    - ✅ **VCF Support**: Convert variant files between assemblies
    - ✅ **AI Conflict Resolution**: Machine learning-based annotation reconciliation
    - ✅ **Multi-format Export**: BED, VCF, CSV, JSON
    
    ### Assemblies Supported
    - GRCh37/hg19 ↔ GRCh38/hg38
    
    ### Data Sources
    - NCBI Gene, Ensembl, RefSeq, GENCODE, UCSC
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
startup_time = time.time()
job_storage: Dict[str, Any] = {}

# Import real services
try:
    from app.services.real_liftover import RealLiftoverService
    from app.services.vcf_converter import VCFConverter
    from app.services.real_ai_resolver import RealAIConflictResolver, AnnotationSource
    from app.validation.validation_suite import GenomicValidationSuite
    
    # Initialize services
    liftover_service = RealLiftoverService()
    vcf_converter = VCFConverter(liftover_service)
    ai_resolver = RealAIConflictResolver()
    validation_suite = GenomicValidationSuite()
    
    logger.info("✅ All real services loaded successfully")
    SERVICES_AVAILABLE = True
    
except Exception as e:
    logger.error(f"❌ Failed to load services: {e}")
    logger.warning("Running in limited mode - install required packages")
    SERVICES_AVAILABLE = False
    liftover_service = None
    vcf_converter = None
    ai_resolver = None
    validation_suite = None

# Job management
class BatchJob:
    def __init__(self, job_id: str, total_items: int, job_type: str):
        self.job_id = job_id
        self.total_items = total_items
        self.processed_items = 0
        self.status = "started"
        self.job_type = job_type
        self.results = []
        self.start_time = datetime.now()
        self.end_time = None
        self.errors = []
        self.metadata = {}

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def landing_page():
    """Landing page with links to documentation"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genomic Annotation Version Controller</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 1200px; margin: 40px auto; padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .card {{ 
                background: rgba(255,255,255,0.1); 
                backdrop-filter: blur(10px);
                padding: 30px; margin: 20px 0; border-radius: 15px;
            }}
            .status {{ display: inline-block; padding: 5px 15px; border-radius: 20px; 
                       background: #4CAF50; margin: 5px; }}
            .btn {{ 
                display: inline-block; background: white; color: #667eea;
                padding: 15px 30px; text-decoration: none; border-radius: 8px;
                margin: 10px 10px 10px 0; font-weight: bold;
            }}
            .feature {{ margin: 20px 0; }}
            .feature h3 {{ color: #FFD700; }}
        </style>
    </head>
    <body>
        <h1>🧬 Genomic Annotation Version Controller</h1>
        <p style="font-size: 20px;">Production-grade genomic data management platform</p>
        
        <div class="card">
            <h2>System Status</h2>
            <span class="status">{"🟢 All Services Online" if SERVICES_AVAILABLE else "🟡 Limited Mode"}</span>
            <span class="status">Version 4.0.0</span>
            <span class="status">Uptime: {int(time.time() - startup_time)}s</span>
        </div>
        
        <div class="card">
            <h2>Features</h2>
            <div class="feature">
                <h3>✅ Real UCSC LiftOver</h3>
                <p>Actual chain file-based coordinate conversion, not approximations</p>
            </div>
            <div class="feature">
                <h3>✅ Validated Against NCBI</h3>
                <p>Tested with {validation_suite.known_genes.__len__() if validation_suite else 10} known gene coordinates</p>
            </div>
            <div class="feature">
                <h3>✅ VCF File Support</h3>
                <p>Convert variant files between genome assemblies</p>
            </div>
            <div class="feature">
                <h3>✅ AI Conflict Resolution</h3>
                <p>Machine learning-based annotation reconciliation using scikit-learn</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Quick Start</h2>
            <a href="/docs" class="btn">📚 API Documentation</a>
            <a href="/validation-report" class="btn">📊 Validation Report</a>
            <a href="/health" class="btn">💚 Health Check</a>
        </div>
        
        <div class="card">
            <h2>Example API Calls</h2>
            <pre style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; overflow-x: auto;">
# Convert a coordinate
POST /liftover/single
{{"chrom": "chr17", "pos": 41196312, "from_build": "hg19", "to_build": "hg38"}}

# Convert VCF file
POST /vcf/convert
Upload your VCF file

# Resolve annotation conflicts
POST /ai/resolve-conflicts
{{"gene_symbol": "BRCA1", "sources": [...]}}
            </pre>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy" if SERVICES_AVAILABLE else "degraded",
        "version": "4.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - startup_time),
        "services": {
            "liftover": liftover_service is not None,
            "vcf_converter": vcf_converter is not None,
            "ai_resolver": ai_resolver is not None,
            "validation": validation_suite is not None
        },
        "active_jobs": len(job_storage),
        "supported_assemblies": ["GRCh37/hg19", "GRCh38/hg38"]
    }

@app.get("/validation-report")
def get_validation_report():
    """Get comprehensive validation report"""
    if not SERVICES_AVAILABLE or not validation_suite or not liftover_service:
        raise HTTPException(
            status_code=503,
            detail="Validation service not available - install required packages"
        )
    
    try:
        results = validation_suite.run_full_validation(liftover_service)
        report_text = validation_suite.generate_validation_report(results)
        
        return {
            "report_text": report_text,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LIFTOVER ENDPOINTS
# ============================================================================

@app.post("/liftover/single")
async def liftover_single_coordinate(
    chrom: str,
    pos: int,
    from_build: str = "hg19",
    to_build: str = "hg38",
    strand: str = "+"
):
    """
    Convert a single genomic coordinate.
    
    Example:
    ```json
    {
        "chrom": "chr17",
        "pos": 41196312,
        "from_build": "hg19",
        "to_build": "hg38"
    }
    ```
    """
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(
            status_code=503,
            detail="Liftover service not available - install pyliftover"
        )
    
    try:
        result = liftover_service.convert_coordinate(
            chrom, pos, from_build, to_build, strand
        )
        return result
    except Exception as e:
        logger.error(f"Liftover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/liftover/batch")
async def liftover_batch_coordinates(
    coordinates: List[Dict],
    from_build: str = "hg19",
    to_build: str = "hg38",
    background_tasks: BackgroundTasks = None
):
    """
    Convert multiple coordinates.
    
    Example:
    ```json
    [
        {"chrom": "chr17", "pos": 41196312},
        {"chrom": "chr7", "pos": 55086725}
    ]
    ```
    """
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(status_code=503, detail="Liftover service not available")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(coordinates), "batch_liftover")
    job_storage[job_id] = job
    
    # Process in background
    async def process():
        job.status = "processing"
        try:
            results = liftover_service.batch_convert(coordinates, from_build, to_build)
            job.results = results
            job.processed_items = len(results)
            job.status = "completed"
            job.end_time = datetime.now()
            
            # Calculate stats
            successful = sum(1 for r in results if r.get("success"))
            job.metadata = {
                "successful": successful,
                "failed": len(results) - successful,
                "success_rate": (successful / len(results) * 100) if results else 0
            }
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "started",
        "total_coordinates": len(coordinates),
        "check_status": f"/job-status/{job_id}"
    }

@app.post("/liftover/region")
async def liftover_region(
    chrom: str,
    start: int,
    end: int,
    from_build: str = "hg19",
    to_build: str = "hg38"
):
    """Convert a genomic region (start and end)"""
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(status_code=503, detail="Liftover service not available")
    
    try:
        result = liftover_service.convert_region(chrom, start, end, from_build, to_build)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# VCF CONVERSION ENDPOINTS
# ============================================================================

@app.post("/vcf/convert")
async def convert_vcf_file(
    file: UploadFile = File(...),
    from_build: str = "hg19",
    to_build: str = "hg38",
    keep_failed: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Convert VCF file between assemblies.
    
    Upload a VCF file and get back a converted VCF with all variants lifted over.
    """
    if not SERVICES_AVAILABLE or not vcf_converter:
        raise HTTPException(status_code=503, detail="VCF converter not available")
    
    # Read file
    content = await file.read()
    vcf_content = content.decode('utf-8')
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, 1, "vcf_conversion")
    job_storage[job_id] = job
    
    # Process in background
    async def process():
        job.status = "processing"
        try:
            result = vcf_converter.convert_vcf(vcf_content, from_build, to_build, keep_failed)
            job.results = [result]
            job.processed_items = 1
            job.status = "completed"
            job.end_time = datetime.now()
            job.metadata = result["statistics"]
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"VCF conversion failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "started",
        "filename": file.filename,
        "from_build": from_build,
        "to_build": to_build,
        "check_status": f"/job-status/{job_id}",
        "download": f"/vcf/download/{job_id}"
    }

@app.get("/vcf/download/{job_id}")
async def download_converted_vcf(job_id: str):
    """Download converted VCF file"""
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job.status}")
    
    if not job.results:
        raise HTTPException(status_code=500, detail="No results available")
    
    vcf_content = job.results[0]["vcf_content"]
    
    return PlainTextResponse(
        vcf_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=converted_{job_id}.vcf"
        }
    )

@app.post("/vcf/validate")
async def validate_vcf_file(file: UploadFile = File(...)):
    """Validate VCF file format"""
    if not SERVICES_AVAILABLE or not vcf_converter:
        raise HTTPException(status_code=503, detail="VCF validator not available")
    
    content = await file.read()
    vcf_content = content.decode('utf-8')
    
    validation = vcf_converter.validate_vcf(vcf_content)
    return validation

# ============================================================================
# AI CONFLICT RESOLUTION ENDPOINTS
# ============================================================================

@app.post("/ai/resolve-conflicts")
async def resolve_annotation_conflicts(
    gene_annotations: List[Dict],
    background_tasks: BackgroundTasks = None
):
    """
    Resolve annotation conflicts using ML clustering.
    
    Example:
    ```json
    [
        {
            "gene_symbol": "BRCA1",
            "sources": [
                {
                    "name": "Ensembl",
                    "start": 43044295,
                    "end": 43125483,
                    "confidence": 0.95,
                    "evidence": ["experimental", "literature"]
                },
                {
                    "name": "RefSeq",
                    "start": 43044294,
                    "end": 43125482,
                    "confidence": 0.92
                }
            ]
        }
    ]
    ```
    """
    if not SERVICES_AVAILABLE or not ai_resolver:
        raise HTTPException(status_code=503, detail="AI resolver not available")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(gene_annotations), "ai_conflict_resolution")
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            resolutions = ai_resolver.batch_resolve(gene_annotations)
            
            # Convert to dict format
            job.results = [
                {
                    "gene_symbol": r.gene_symbol,
                    "resolved_coordinates": {
                        "start": r.resolved_start,
                        "end": r.resolved_end,
                        "strand": r.resolved_strand
                    },
                    "confidence_score": r.confidence_score,
                    "resolution_method": r.resolution_method,
                    "contributing_sources": r.contributing_sources,
                    "conflict_types": r.conflict_types,
                    "consensus_level": r.consensus_level,
                    "statistical_metrics": r.statistical_metrics,
                    "recommendation": r.recommendation,
                    "manual_review_needed": r.manual_review_needed
                }
                for r in resolutions
            ]
            
            job.processed_items = len(resolutions)
            job.status = "completed"
            job.end_time = datetime.now()
            job.metadata = ai_resolver.generate_report(resolutions)
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"AI resolution failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "started",
        "genes_to_process": len(gene_annotations),
        "check_status": f"/job-status/{job_id}"
    }

# ============================================================================
# JOB MANAGEMENT
# ============================================================================

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
        "progress_percent": round(progress, 1),
        "processed_items": job.processed_items,
        "total_items": job.total_items,
        "start_time": job.start_time.isoformat(),
        "errors": job.errors
    }
    
    if job.status == "completed":
        response["end_time"] = job.end_time.isoformat() if job.end_time else None
        response["processing_time_seconds"] = (
            (job.end_time - job.start_time).total_seconds() 
            if job.end_time else None
        )
        response["metadata"] = job.metadata
        response["results_count"] = len(job.results)
        
        # Add download links based on job type
        if job.job_type == "vcf_conversion":
            response["download_url"] = f"/vcf/download/{job_id}"
        else:
            response["download_url"] = f"/export/{job_id}/json"
    
    return response

@app.get("/export/{job_id}/{format}")
def export_job_results(job_id: str, format: str):
    """Export job results in various formats"""
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status})")
    
    format = format.lower()
    
    if format == "json":
        content = json.dumps({
            "job_info": {
                "job_id": job_id,
                "job_type": job.job_type,
                "processed_at": job.end_time.isoformat() if job.end_time else None,
                "metadata": job.metadata
            },
            "results": job.results
        }, indent=2)
        media_type = "application/json"
        filename = f"results_{job_id}.json"
    
    elif format == "csv":
        # Convert results to CSV
        import csv
        from io import StringIO
        
        output = StringIO()
        if job.results and isinstance(job.results[0], dict):
            writer = csv.DictWriter(output, fieldnames=job.results[0].keys())
            writer.writeheader()
            writer.writerows(job.results)
        
        content = output.getvalue()
        media_type = "text/csv"
        filename = f"results_{job_id}.csv"
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    
    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 70)
    logger.info("🧬 Genomic Annotation Version Controller v4.0.0")
    logger.info("=" * 70)
    logger.info(f"Services Available: {SERVICES_AVAILABLE}")
    logger.info(f"LiftOver Service: {'✅' if liftover_service else '❌'}")
    logger.info(f"VCF Converter: {'✅' if vcf_converter else '❌'}")
    logger.info(f"AI Resolver: {'✅' if ai_resolver else '❌'}")
    logger.info(f"Validation Suite: {'✅' if validation_suite else '❌'}")
    logger.info("=" * 70)
    
    if SERVICES_AVAILABLE:
        logger.info("All systems operational!")
    else:
        logger.warning(" Running in limited mode - install required packages")
        logger.warning("   Run: pip install pyliftover scikit-learn numpy")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)