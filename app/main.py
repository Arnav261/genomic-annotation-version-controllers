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
from app.database import SessionLocal, Job
import numpy as np
from app.config import settings
startup_time = time.time()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


job_storage: Dict[str, Any] = {}
SERVICES: Dict[str, Any] = {}

def init_services():
    try:
        SERVICES["chain_liftover"] = RealLiftover(chain_dir=str(settings.CHAIN_DIR))
    except Exception:
        SERVICES["chain_liftover"] = None

    try:
        SERVICES["liftover"] = EnsemblLiftover(
            fallback=SERVICES.get("chain_liftover")
        )
    except Exception:
        SERVICES["liftover"] = SERVICES.get("chain_liftover")

    if FeatureExtractor:
        try:
            SERVICES["feature_extractor"] = FeatureExtractor(
                data_dir=str(settings.REF_DIR)
            )
        except Exception:
            SERVICES["feature_extractor"] = None
    else:
        SERVICES["feature_extractor"] = None

    if ConfidencePredictor:
        try:
            cp = ConfidencePredictor(model_dir=str(settings.MODEL_DIR))
            cp.load_model_if_exists()
            SERVICES["confidence_predictor"] = cp
        except Exception:
            SERVICES["confidence_predictor"] = None
    else:
        SERVICES["confidence_predictor"] = None

    if SERVICES.get("liftover"):
        SERVICES["vcf_converter"] = VCFConverter(SERVICES["liftover"])
    else:
        SERVICES["vcf_converter"] = None

def initialize_services() -> bool:
    """
    Backwards-compatible wrapper that initializes services and returns True/False.
    """
    try:
        init_services()
        return True
    except Exception as e:
        logger.exception("Service initialization failed: %s", e)
        return False
SERVICES_AVAILABLE = initialize_services()
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

try:
    from app.ml.feature_extractor import FeatureExtractor
except ImportError:
    FeatureExtractor = None


@app.get("/", response_class=HTMLResponse)
def landing_page():
    db = SessionLocal()
    try:
        active_jobs = db.query(Job).filter(
            Job.status.in_(["queued", "processing"])
        ).count()
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
/* existing JS logic preserved */
async function run() {{
    const chrom = document.getElementById("chrom").value;
    const pos = document.getElementById("pos").value;

    const r = await fetch(
        `/liftover/single?chrom=${{chrom}}&pos=${{pos}}`
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



@app.api_route("/liftover/single", methods=["GET", "POST"])
async def liftover_single(
    chrom: str = Query(...),
    pos: int = Query(..., ge=1),
    from_build: str = Query("hg19"),
    to_build: str = Query("hg38"),
    strand: str = Query("+"),
    include_ml: bool = Query(True)
):
    """Convert single genomic coordinate"""
    if not SERVICES_AVAILABLE or not SERVICES.get('liftover'):
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


@app.post("/liftover/batch")
async def liftover_batch(
    coordinates: List[Dict],
    from_build: str = Query("hg19"),
    to_build: str = Query("hg38"),
    include_ml: bool = Query(True),
    background_tasks: BackgroundTasks = None
):
    """Batch coordinate conversion"""
    if not SERVICES_AVAILABLE or not SERVICES.get('liftover'):
        raise HTTPException(status_code=503, detail="Liftover service unavailable")
    
    if not coordinates:
        raise HTTPException(status_code=400, detail="No coordinates provided")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(coordinates), "batch_liftover_ml")
    job_storage[job_id] = job
    
    logger.info(f"Created batch job {job_id} with {len(coordinates)} coordinates")
    
    async def process():
        job.status = "processing"
        db.commit()
        try:
            results = []
            
            for i, coord in enumerate(coordinates):
                result = SERVICES['liftover'].convert_coordinate(
                    coord.get("chrom", ""),
                    coord.get("pos", 0),
                    from_build,
                    to_build
                )
                job.processed_items = i + 1
                db.commit()

                # Normalize confidence score
                if 'confidence' in result and result['confidence'] is not None:
                    result['confidence'] = float(result['confidence'])
                    result['confidence'] = max(0.0, min(1.0, result['confidence']))
                
                if include_ml and SERVICES.get('feature_extractor') and SERVICES.get('confidence_predictor'):
                    try:
                        features = SERVICES['feature_extractor'].extract_features(
                            coord.get("chrom", ""),
                            coord.get("pos", 0),
                            from_build,
                            to_build,
                            result
                        )
                        
                        ml_confidence = SERVICES['confidence_predictor'].predict_confidence(features.to_array())
                        ml_confidence = float(ml_confidence)
                        ml_confidence = max(0.0, min(1.0, ml_confidence))
                        
                        result['ml_confidence'] = ml_confidence
                        result['ml_interpretation'] = SERVICES['confidence_predictor'].interpret_confidence(ml_confidence)
                    except:
                        pass
                
                results.append(result)
                job.processed_items = i + 1
            
            job.results = results
            job.status = "completed"
            db.commit()
            job.end_time = datetime.now()
            
            successful = sum(1 for r in results if r.get("success"))
            job.job_metadata = {
                "successful": successful,
                "failed": len(results) - successful,
                "success_rate": round((successful / len(results) * 100), 2),
                "ml_predictions": sum(1 for r in results if 'ml_confidence' in r),
                "from_build": from_build,
                "to_build": to_build
            }
            
            logger.info(f"Job {job_id} completed: {successful}/{len(results)} successful")
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"Job {job_id} failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_coordinates": len(coordinates),
        "ml_enabled": include_ml,
        "estimated_time_seconds": len(coordinates) * 0.2,
        "status_endpoint": f"/job-status/{job_id}",
        "export_endpoints": {
            "json": f"/export/{job_id}/json",
            "csv": f"/export/{job_id}/csv"
        }
    }


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
        response["metadata"] = job.job_metadata
        response["results_count"] = len(job.results)
        response["export_options"] = {
            "json": f"/export/{job_id}/json",
            "csv": f"/export/{job_id}/csv"
        }
    
    return response


@app.get("/export/{job_id}/{format}")
def export_results(
    job_id: str,
    format: str = FastAPIPath(..., regex=r"^(json|csv)$")
):
    """Export job results"""
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job.status}"
        )
    
    if format == "json":
        content = json.dumps({
            "job_info": {
                "job_id": job_id,
                "job_type": job.job_type,
                "completed_at": job.end_time.isoformat() if job.end_time else None,
                "processing_time_seconds": (
                    (job.end_time - job.start_time).total_seconds()
                    if job.end_time else None
                ),
                "metadata": job.job_metadata
            },
            "results": job.results
        }, indent=2)
        
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=results_{job_id}.json"}
        )
    
    elif format == "csv":
        output = StringIO()
        
        if job.results and isinstance(job.results[0], dict):
            flattened = []
            for result in job.results:
                flat = {}
                for key, value in result.items():
                    if isinstance(value, (dict, list)):
                        flat[key] = json.dumps(value)
                    else:
                        flat[key] = value
                flattened.append(flat)
            
            if flattened:
                writer = csv.DictWriter(output, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
        
        content = output.getvalue()
        
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=results_{job_id}.csv"}
        )


@app.get("/validation-report")
def validation_report():
    """Generate validation report"""
    if not SERVICES.get('validation_engine') or not SERVICES.get('liftover'):
        return {
            "status": "limited",
            "message": "Full validation unavailable",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        from app.validation.validation_suite import GenomicValidationSuite
        validation_suite = GenomicValidationSuite()
        
        results = validation_suite.run_full_validation(SERVICES['liftover'])
        report_text = validation_suite.generate_validation_report(results)
        
        return {
            "validation_report": report_text,
            "summary": results["summary"],
            "statistics": results["statistics"],
            "methodology": results["methodology"],
            "detailed_results": results["detailed_results"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Resonance - Genomic Coordinate Liftover Service v1.0.0")
    logger.info(f"Services Status: {SERVICES_AVAILABLE}")
    
    for service_name, service in SERVICES.items():
        status = 'Available' if service else 'Unavailable'
        logger.info(f"  {service_name}: {status}")
    
    if SERVICES_AVAILABLE:
        logger.info("All critical systems operational")
    else:
        logger.warning("Operating in limited mode")

@app.post("/vcf/convert")
async def convert_vcf_file(
    file: UploadFile = File(..., description="VCF file to convert"),
    from_build: str = Query("hg19", description="Source assembly"),
    to_build: str = Query("hg38", description="Target assembly"),
    keep_failed: bool = Query(False, description="Include variants that failed liftover"),
    background_tasks: BackgroundTasks = None
):
    """
    Convert VCF file between genome assemblies.
    
    Uploads VCF file and converts all variants to target assembly.
    Preserves sample information, genotypes, and INFO fields.
    
    Processing is asynchronous. Use job_id to check progress and download results.
    """
    if not SERVICES_AVAILABLE or not vcf_converter:
        raise HTTPException(status_code=503, detail="VCF converter unavailable")
    
    # Validate file extension
    if not file.filename.endswith(('.vcf', '.vcf.gz')):
        raise HTTPException(status_code=400, detail="File must be VCF format (.vcf or .vcf.gz)")
    
    content = await file.read()
    vcf_content = content.decode('utf-8')
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, 1, "vcf_conversion")
    job.job_metadata["original_filename"] = file.filename
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            result = vcf_converter.convert_vcf(vcf_content, from_build, to_build, keep_failed)
            job.results = [result]
            job.processed_items = 1
            job.status = "completed"
            job.end_time = datetime.now()
            job.job_metadata.update(result["statistics"])
            
            if result["statistics"]["failed_conversion"] > 0:
                job.warnings.append(
                    f"{result['statistics']['failed_conversion']} variants failed conversion"
                )
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"VCF conversion failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "original_filename": file.filename,
        "from_build": from_build,
        "to_build": to_build,
        "status_endpoint": f"/job-status/{job_id}",
        "download_endpoint": f"/vcf/download/{job_id}"
    }

@app.get("/vcf/download/{job_id}")
async def download_converted_vcf(job_id: str):
    """Download converted VCF file"""
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if not job.results:
        raise HTTPException(status_code=500, detail="No results available")
    
    vcf_content = job.results[0]["vcf_content"]
    original_filename = job.job_metadata.get("original_filename", "input.vcf")
    output_filename = f"converted_{original_filename}"
    
    return PlainTextResponse(
        vcf_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename={output_filename}"
        }
    )

@app.post("/vcf/validate")
async def validate_vcf_file(
    file: UploadFile = File(..., description="VCF file to validate")
):
    """
    Validate VCF file format compliance.
    
    Checks for required headers, column structure, and variant line formatting.
    """
    if not SERVICES_AVAILABLE or not vcf_converter:
        raise HTTPException(status_code=503, detail="VCF validator unavailable")
    
    content = await file.read()
    vcf_content = content.decode('utf-8')
    
    validation = vcf_converter.validate_vcf(vcf_content)
    
    return {
        "filename": file.filename,
        "validation_result": validation,
        "timestamp": datetime.now().isoformat()
    }



@app.post("/semantic/reconcile")
async def reconcile_semantic_annotations(
    gene_symbol: str = Query(..., description="Gene symbol"),
    annotations: List[Dict] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Reconcile conflicting gene descriptions using semantic analysis.
    
    Request body format:
    [
        {
            "description": "BRCA1 DNA repair associated",
            "source": "NCBI",
            "biological_process": ["DNA repair"],
            "confidence": 0.95
        },
        {
            "description": "breast cancer type 1 susceptibility protein",
            "source": "UniProt",
            "molecular_function": ["protein binding"],
            "confidence": 0.97
        }
    ]
    
    Returns reconciled description with consensus biological terms.
    """
    if not SERVICES_AVAILABLE or not semantic_engine:
        raise HTTPException(status_code=503, detail="Semantic reconciliation unavailable")
    
    if not annotations or len(annotations) == 0:
        raise HTTPException(status_code=400, detail="No annotations provided")
    
    # Convert to SemanticAnnotation objects
    from app.services.semantic_reconciliation import SemanticAnnotation
    
    semantic_annotations = []
    for ann in annotations:
        semantic_ann = SemanticAnnotation(
            gene_symbol=gene_symbol,
            description=ann.get("description", ""),
            source=ann.get("source", "Unknown"),
            biological_process=ann.get("biological_process"),
            molecular_function=ann.get("molecular_function"),
            cellular_component=ann.get("cellular_component"),
            protein_domains=ann.get("protein_domains"),
            confidence=ann.get("confidence", 0.8)
        )
        semantic_annotations.append(semantic_ann)
    
    try:
        result = semantic_engine.reconcile_annotations(gene_symbol, semantic_annotations)
        return result
    except Exception as e:
        logger.error(f"Semantic reconciliation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/semantic/batch-reconcile")
async def batch_reconcile_semantic(
    gene_annotations: Dict[str, List[Dict]],
    background_tasks: BackgroundTasks = None
):
    """
    Batch reconciliation for multiple genes.
    
    Request body maps gene symbols to annotation lists.
    Processing is asynchronous.
    """
    if not SERVICES_AVAILABLE or not semantic_engine:
        raise HTTPException(status_code=503, detail="Semantic reconciliation unavailable")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(gene_annotations), "semantic_reconciliation")
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            from app.services.semantic_reconciliation import SemanticAnnotation
            
            # Convert all annotations
            converted = {}
            for gene, anns in gene_annotations.items():
                semantic_anns = []
                for ann in anns:
                    semantic_ann = SemanticAnnotation(
                        gene_symbol=gene,
                        description=ann.get("description", ""),
                        source=ann.get("source", "Unknown"),
                        biological_process=ann.get("biological_process"),
                        molecular_function=ann.get("molecular_function"),
                        confidence=ann.get("confidence", 0.8)
                    )
                    semantic_anns.append(semantic_ann)
                converted[gene] = semantic_anns
            
            results = semantic_engine.batch_reconcile(converted)
            report = semantic_engine.generate_reconciliation_report(results)
            
            job.results = list(results.values())
            job.processed_items = len(results)
            job.status = "completed"
            job.end_time = datetime.now()
            job.job_metadata = report
            
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"Batch semantic reconciliation failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "genes_to_process": len(gene_annotations),
        "status_endpoint": f"/job-status/{job_id}"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)