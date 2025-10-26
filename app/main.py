from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import logging.config 
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from pathlib import Path
import os, io, csv, json
import threading
import uuid
import time
import asyncio
import os
from datetime import datetime
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
    ## Professional Research-Grade Bioinformatics Platform
    
    ### Core Capabilities
    
    **Coordinate Liftover**
    - UCSC chain file-based coordinate conversion
    - Support for GRCh37/hg19 to GRCh38/hg38
    - Validated against NCBI RefSeq coordinates
    - Batch processing with progress tracking
    
    **VCF File Processing**
    - Parse and convert variant files between assemblies
    - Format validation and quality control
    - Preservation of sample and genotype data
    
    **Semantic Reconciliation**
    - Natural language processing of gene descriptions
    - Biological term extraction and normalization
    - Multi-source annotation consensus
    
    **Conflict Resolution**
    - Machine learning-based clustering (DBSCAN, Agglomerative)
    - Statistical confidence metrics
    - Evidence-weighted decision making
    
    ### Quality Assurance
    
    All coordinate conversions validated against known NCBI Gene coordinates
    for major genes including BRCA1, TP53, EGFR, CFTR, APOE, KRAS, BRCA2,
    MYC, PTEN, and HBB.
    
    ### Data Sources
    
    Integrates data from NCBI Gene, Ensembl, RefSeq, GENCODE, UCSC Genome
    Browser, UniProt, and HGNC.
    
    ### Citation
    
    If you use this tool in your research, please cite:
    [Citation information to be added]
    
    ### Support
    
    For issues, feature requests, or questions, please visit:
    [Repository URL]
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Arnav Asher",
        "email": "arnavasher007@gmail.com",
    },
    
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

# Import services
try:
    from app.services.real_liftover import RealLiftoverService
    from app.services.vcf_converter import VCFConverter
    from app.services.real_ai_resolver import RealAIConflictResolver
    from app.services.semantic_reconciliation import SemanticReconciliationEngine
    from app.validation.validation_suite import GenomicValidationSuite
    
    # Initialize services
    liftover_service = RealLiftoverService()
    vcf_converter = VCFConverter(liftover_service)
    ai_resolver = RealAIConflictResolver()
    semantic_engine = SemanticReconciliationEngine()
    validation_suite = GenomicValidationSuite()
    
    logger.info("All services initialized successfully")
    SERVICES_AVAILABLE = True
    
except Exception as e:
    logger.error(f"Service initialization failed: {e}")
    logger.warning("Operating in limited mode")
    SERVICES_AVAILABLE = False
    liftover_service = None
    vcf_converter = None
    ai_resolver = None
    semantic_engine = None
    validation_suite = None

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



@app.get("/", response_class=HTMLResponse)
def landing_page():
    """Professional landing page"""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Genomic Annotation Version Controller</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            header {{
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            h1 {{
                color: #1e3c72;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            .subtitle {{
                color: #666;
                font-size: 1.2em;
            }}
            .card {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .status-bar {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }}
            .status-item {{
                background: #f8f9fa;
                padding: 15px 25px;
                border-radius: 8px;
                text-align: center;
                flex: 1;
                min-width: 150px;
            }}
            .status-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .status-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #1e3c72;
            }}
            .feature {{
                margin: 20px 0;
                padding: 20px;
                border-left: 4px solid #1e3c72;
                background: #f8f9fa;
            }}
            .feature h3 {{
                color: #1e3c72;
                margin-bottom: 10px;
            }}
            .btn {{
                display: inline-block;
                background: #1e3c72;
                color: white;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 10px 10px 0;
                font-weight: bold;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #2a5298;
            }}
            .btn-secondary {{
                background: #6c757d;
            }}
            .btn-secondary:hover {{
                background: #5a6268;
            }}
            footer {{
                text-align: center;
                color: white;
                margin-top: 40px;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Genomic Annotation Version Controller</h1>
                <p class="subtitle">Professional Research-Grade Bioinformatics Platform</p>
            </header>
            
            <div class="card">
                <h2>System Status</h2>
                <div class="status-bar">
                    <div class="status-item">
                        <div class="status-label">System Status</div>
                        <div class="status-value">{"Operational" if SERVICES_AVAILABLE else "Limited"}</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Version</div>
                        <div class="status-value">4.0.0</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Uptime</div>
                        <div class="status-value">{int((time.time() - startup_time) / 3600)}h</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Active Jobs</div>
                        <div class="status-value">{len(job_storage)}</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Core Capabilities</h2>
                
                <div class="feature">
                    <h3>Coordinate Liftover</h3>
                    <p>Production-grade coordinate conversion using UCSC chain files. 
                    Validated against NCBI RefSeq coordinates with demonstrated accuracy 
                    exceeding 95% for major genes.</p>
                </div>
                
                <div class="feature">
                    <h3>VCF File Processing</h3>
                    <p>Complete VCF file parsing, validation, and conversion between 
                    genome assemblies. Maintains sample information and genotype data 
                    integrity.</p>
                </div>
                
                <div class="feature">
                    <h3>Semantic Reconciliation</h3>
                    <p>Natural language processing and biological term extraction to 
                    reconcile conflicting gene descriptions across multiple annotation 
                    databases.</p>
                </div>
                
                <div class="feature">
                    <h3>AI Conflict Resolution</h3>
                    <p>Machine learning-based annotation conflict resolution using 
                    DBSCAN and agglomerative clustering with evidence-weighted 
                    decision making.</p>
                </div>
            </div>
            
            <div class="card">
                <h2>Access Documentation</h2>
                <a href="/docs" class="btn">API Documentation</a>
                <a href="/validation-report" class="btn btn-secondary">Validation Report</a>
                <a href="/health" class="btn btn-secondary">System Health</a>
            </div>
            
            <div class="card">
                <h2>Supported Assemblies</h2>
                <p>GRCh37 (hg19) ↔ GRCh38 (hg38)</p>
                
                <h2 style="margin-top: 30px;">Data Sources</h2>
                <p>NCBI Gene, Ensembl, RefSeq, GENCODE, UCSC Genome Browser, 
                UniProt, HGNC</p>
            </div>
            
            <footer>
                <p>Genomic Annotation Version Controller v4.0.0</p>
                <p>Professional Research-Grade Bioinformatics Platform</p>
            </footer>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
def health_check():
    """System health check endpoint"""
    return {
        "status": "operational" if SERVICES_AVAILABLE else "degraded",
        "version": "4.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - startup_time),
        "services": {
            "liftover": liftover_service is not None,
            "vcf_converter": vcf_converter is not None,
            "ai_resolver": ai_resolver is not None,
            "semantic_reconciliation": semantic_engine is not None,
            "validation": validation_suite is not None
        },
        "active_jobs": len(job_storage),
        "supported_assemblies": ["GRCh37/hg19", "GRCh38/hg38"],
        "data_sources": [
            "NCBI Gene", "Ensembl", "RefSeq", "GENCODE", 
            "UCSC Genome Browser", "UniProt", "HGNC"
        ]
    }

@app.get("/demo", response_class=HTMLResponse)
def demo_interface():
    """Interactive demo interface"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Demo - Genomic Annotation Version Controller</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h1, h2 { color: #1e3c72; margin-bottom: 20px; }
            .demo-section {
                margin: 30px 0;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            .input-group {
                margin: 15px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            input, select, textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            button {
                background: #1e3c72;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background: #2a5298;
            }
            .result-box {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                white-space: pre-wrap;
                font-family: monospace;
                max-height: 400px;
                overflow-y: auto;
            }
            .loading {
                display: none;
                color: #1e3c72;
                font-style: italic;
            }
            .error {
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .success {
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Interactive Demo</h1>
                <p>Test the Genomic Annotation Version Controller API in real-time.</p>
            </div>

            <!-- Single Coordinate Liftover -->
            <div class="card">
                <div class="demo-section">
                    <h2>1. Single Coordinate Liftover</h2>
                    <p>Convert a single genomic coordinate between assemblies.</p>
                    
                    <div class="input-group">
                        <label>Chromosome:</label>
                        <input type="text" id="liftover-chrom" value="chr17" placeholder="chr17">
                    </div>
                    
                    <div class="input-group">
                        <label>Position:</label>
                        <input type="number" id="liftover-pos" value="41196312" placeholder="41196312">
                    </div>
                    
                    <div class="input-group">
                        <label>From Assembly:</label>
                        <select id="liftover-from">
                            <option value="hg19">hg19 (GRCh37)</option>
                            <option value="hg38">hg38 (GRCh38)</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>To Assembly:</label>
                        <select id="liftover-to">
                            <option value="hg38">hg38 (GRCh38)</option>
                            <option value="hg19">hg19 (GRCh37)</option>
                        </select>
                    </div>
                    
                    <button onclick="testLiftover()">Convert Coordinate</button>
                    <div class="loading" id="liftover-loading">Processing...</div>
                    <div class="result-box" id="liftover-result"></div>
                </div>
            </div>

            <!-- Validation Report -->
            <div class="card">
                <div class="demo-section">
                    <h2>2. System Validation Report</h2>
                    <p>View accuracy validation against NCBI RefSeq coordinates.</p>
                    
                    <button onclick="getValidation()">Get Validation Report</button>
                    <div class="loading" id="validation-loading">Loading...</div>
                    <div class="result-box" id="validation-result"></div>
                </div>
            </div>

            <!-- Batch Liftover -->
            <div class="card">
                <div class="demo-section">
                    <h2>3. Batch Coordinate Conversion</h2>
                    <p>Convert multiple coordinates at once.</p>
                    
                    <div class="input-group">
                        <label>Coordinates (JSON format):</label>
                        <textarea id="batch-coords" rows="6">
[
  {"chrom": "chr17", "pos": 41196312},
  {"chrom": "chr7", "pos": 55086725},
  {"chrom": "chr17", "pos": 7571720}
]</textarea>
                    </div>
                    
                    <button onclick="testBatchLiftover()">Start Batch Job</button>
                    <div class="loading" id="batch-loading">Processing...</div>
                    <div class="result-box" id="batch-result"></div>
                </div>
            </div>

            <div class="card">
                <p style="text-align: center; color: #666;">
                    <strong>Note:</strong> This demo uses live API calls. Results may take a few seconds.
                    <br>
                    For full API documentation, visit <a href="/docs" style="color: #1e3c72;">/docs</a>
                </p>
            </div>
        </div>

        <script>
            async function testLiftover() {
                const chrom = document.getElementById('liftover-chrom').value;
                const pos = document.getElementById('liftover-pos').value;
                const fromBuild = document.getElementById('liftover-from').value;
                const toBuild = document.getElementById('liftover-to').value;
                
                const resultDiv = document.getElementById('liftover-result');
                const loadingDiv = document.getElementById('liftover-loading');
                
                resultDiv.textContent = '';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch(
                        `/liftover/single?chrom=${chrom}&pos=${pos}&from_build=${fromBuild}&to_build=${toBuild}`
                        , { method: 'POST' }
                    );
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    
                    if (data.success) {
                        resultDiv.innerHTML = `<div class="success">Conversion Successful!</div>`;
                        resultDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                    } else {
                        resultDiv.innerHTML = `<div class="error">Conversion Failed</div>`;
                        resultDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                    }
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }

            async function getValidation() {
                const resultDiv = document.getElementById('validation-result');
                const loadingDiv = document.getElementById('validation-loading');
                
                resultDiv.textContent = '';
                loadingDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/validation-report');
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    
                    resultDiv.innerHTML = `<div class="success">Validation Report Generated</div>`;
                    resultDiv.innerHTML += `<pre>${data.validation_report}</pre>`;
                    resultDiv.innerHTML += `<h3>Summary:</h3>`;
                    resultDiv.innerHTML += `<pre>${JSON.stringify(data.summary, null, 2)}</pre>`;
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }

            async function testBatchLiftover() {
                const coordsText = document.getElementById('batch-coords').value;
                const resultDiv = document.getElementById('batch-result');
                const loadingDiv = document.getElementById('batch-loading');
                
                resultDiv.textContent = '';
                loadingDiv.style.display = 'block';
                
                try {
                    const coordinates = JSON.parse(coordsText);
                    
                    const response = await fetch('/liftover/batch?from_build=hg19&to_build=hg38', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(coordinates)
                    });
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    
                    resultDiv.innerHTML = `<div class="success">Batch Job Started!</div>`;
                    resultDiv.innerHTML += `<p>Job ID: <strong>${data.job_id}</strong></p>`;
                    resultDiv.innerHTML += `<p>Check status at: <a href="/job-status/${data.job_id}" target="_blank">/job-status/${data.job_id}</a></p>`;
                    resultDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """)

@app.get("/validation-report")
def get_validation_report():
    """
    Comprehensive validation report.
    
    Returns validation results for known gene coordinates tested against
    NCBI RefSeq database. Includes accuracy metrics and detailed results.
    """
    if not SERVICES_AVAILABLE or not validation_suite or not liftover_service:
        raise HTTPException(
            status_code=503,
            detail="Validation service unavailable"
        )
    
    try:
        results = validation_suite.run_full_validation(liftover_service)
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


@app.post("/liftover/single")
async def liftover_single_coordinate(
    chrom: str = Query(..., description="Chromosome (e.g., chr17 or 17)"),
    pos: int = Query(..., description="Position (1-based)", ge=1),
    from_build: str = Query("hg19", description="Source assembly (hg19, hg38, GRCh37, GRCh38)"),
    to_build: str = Query("hg38", description="Target assembly"),
    strand: str = Query("+", description="Strand (+, -, or unspecified)")
):
    """
    Convert single genomic coordinate between assemblies.
    
    Uses UCSC LiftOver chain files for accurate coordinate conversion.
    
    Example request:
        POST /liftover/single?chrom=chr17&pos=41196312&from_build=hg19&to_build=hg38
    
    Returns:
        - success: boolean indicating conversion success
        - lifted_chrom: converted chromosome
        - lifted_pos: converted position
        - confidence: mapping confidence score
        - method: conversion method used
    """
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(
            status_code=503,
            detail="Liftover service unavailable"
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
    coordinates: List[Dict] = None,
    from_build: str = Query("hg19", description="Source assembly"),
    to_build: str = Query("hg38", description="Target assembly"),
    background_tasks: BackgroundTasks = None
):
    """
    Convert multiple coordinates in batch mode.
    
    Request body format:
    [
        {"chrom": "chr17", "pos": 41196312},
        {"chrom": "chr7", "pos": 55086725, "strand": "+"}
    ]
    
    Processing is performed asynchronously. Use returned job_id to check status.
    """
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(status_code=503, detail="Liftover service unavailable")
    
    if not coordinates:
        raise HTTPException(status_code=400, detail="No coordinates provided")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(coordinates), "batch_liftover")
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            results = liftover_service.batch_convert(coordinates, from_build, to_build)
            job.results = results
            job.processed_items = len(results)
            job.status = "completed"
            job.end_time = datetime.now()
            
            successful = sum(1 for r in results if r.get("success"))
            job.metadata = {
                "successful": successful,
                "failed": len(results) - successful,
                "success_rate": round((successful / len(results) * 100), 2) if results else 0,
                "from_build": from_build,
                "to_build": to_build
            }
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error(f"Batch liftover failed: {e}")
    
    background_tasks.add_task(process)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_coordinates": len(coordinates),
        "estimated_time_seconds": len(coordinates) * 0.1,
        "status_endpoint": f"/job-status/{job_id}"
    }

@app.post("/liftover/region")
async def liftover_region(
    chrom: str = Query(..., description="Chromosome"),
    start: int = Query(..., description="Region start (1-based)", ge=1),
    end: int = Query(..., description="Region end (inclusive)", ge=1),
    from_build: str = Query("hg19", description="Source assembly"),
    to_build: str = Query("hg38", description="Target assembly")
):
    """
    Convert genomic region (start and end coordinates).
    
    Both boundaries are converted and confidence metrics provided.
    """
    if not SERVICES_AVAILABLE or not liftover_service:
        raise HTTPException(status_code=503, detail="Liftover service unavailable")
    
    if start >= end:
        raise HTTPException(status_code=400, detail="Start position must be less than end position")
    
    try:
        result = liftover_service.convert_region(chrom, start, end, from_build, to_build)
        return result
    except Exception as e:
        logger.error(f"Region liftover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    job.metadata["original_filename"] = file.filename
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            result = vcf_converter.convert_vcf(vcf_content, from_build, to_build, keep_failed)
            job.results = [result]
            job.processed_items = 1
            job.status = "completed"
            job.end_time = datetime.now()
            job.metadata.update(result["statistics"])
            
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
    original_filename = job.metadata.get("original_filename", "input.vcf")
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
            job.metadata = report
            
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



@app.post("/ai/resolve-conflicts")
async def resolve_annotation_conflicts(
    gene_annotations: List[Dict],
    background_tasks: BackgroundTasks = None
):
    """
    Resolve annotation conflicts using machine learning.
    
    Request body format:
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
    
    Uses DBSCAN or agglomerative clustering for consensus determination.
    """
    if not SERVICES_AVAILABLE or not ai_resolver:
        raise HTTPException(status_code=503, detail="AI resolver unavailable")
    
    if not gene_annotations:
        raise HTTPException(status_code=400, detail="No gene annotations provided")
    
    job_id = uuid.uuid4().hex[:8]
    job = BatchJob(job_id, len(gene_annotations), "ai_conflict_resolution")
    job_storage[job_id] = job
    
    async def process():
        job.status = "processing"
        try:
            resolutions = ai_resolver.batch_resolve(gene_annotations)
            
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
        "status": "queued",
        "genes_to_process": len(gene_annotations),
        "status_endpoint": f"/job-status/{job_id}"
    }



@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    """
    Retrieve job processing status.
    
    Returns current status, progress, and results when completed.
    """
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
        
        # Add download endpoints
        if job.job_type == "vcf_conversion":
            response["download_url"] = f"/vcf/download/{job_id}"
        else:
            response["export_options"] = {
                "json": f"/export/{job_id}/json",
                "csv": f"/export/{job_id}/csv"
            }
    
    return response

@app.get("/export/{job_id}/{format}")
def export_job_results(
    job_id: str,
    format: str = Path(..., pattern="^(json|csv)$", description="Export format")
):
    """
    Export job results in specified format.
    
    Supported formats: json, csv
    """
    job = job_storage.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if format == "json":
        content = json.dumps({
            "job_info": {
                "job_id": job_id,
                "job_type": job.job_type,
                "processed_at": job.end_time.isoformat() if job.end_time else None,
                "processing_time_seconds": (
                    (job.end_time - job.start_time).total_seconds() 
                    if job.end_time else None
                ),
                "metadata": job.metadata
            },
            "results": job.results
        }, indent=2)
        media_type = "application/json"
        filename = f"results_{job_id}.json"
    
    elif format == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        if job.results and isinstance(job.results[0], dict):
            # Flatten nested structures for CSV
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
        media_type = "text/csv"
        filename = f"results_{job_id}.csv"
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )



@app.on_event("startup")
async def startup_event():
    """Log application startup"""
    logger.info("=" * 80)
    logger.info("Genomic Annotation Version Controller v4.0.0")
    logger.info("Professional Research-Grade Bioinformatics Platform")
    logger.info("=" * 80)
    logger.info(f"Services Available: {SERVICES_AVAILABLE}")
    logger.info(f"LiftOver Service: {'Available' if liftover_service else 'Unavailable'}")
    logger.info(f"VCF Converter: {'Available' if vcf_converter else 'Unavailable'}")
    logger.info(f"AI Resolver: {'Available' if ai_resolver else 'Unavailable'}")
    logger.info(f"Semantic Engine: {'Available' if semantic_engine else 'Unavailable'}")
    logger.info(f"Validation Suite: {'Available' if validation_suite else 'Unavailable'}")
    logger.info("=" * 80)
    
    if SERVICES_AVAILABLE:
        logger.info("All systems operational")
    else:
        logger.warning("Operating in limited mode")
        logger.warning("Install required packages: pip install pyliftover scikit-learn")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)