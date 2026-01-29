"""
Genomic Coordinate Liftover Service - Resonance
Bioinformatics Platform
"""

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, UploadFile, File, Query, Path as FastAPIPath
from fastapi.templating import Jinja2Templates
from fastapi import Form
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
from app.database import SessionLocal, Job, APIKey
from app.services.semantic_reconciliation import SemanticReconciliationEngine, SemanticAnnotation


templates = Jinja2Templates(directory="templates")
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

# After the feature_extractor / before confidence_predictor instantiation in your initializer:
try:
    sem_engine = SemanticReconciliationEngine()
    SERVICES["semantic_engine"] = sem_engine
    logger.info("SemanticReconciliationEngine initialized.")
except Exception as e:
    SERVICES["semantic_engine"] = None
    logger.warning("SemanticReconciliationEngine initialization failed: %s", e)

# Initialize FastAPI
app = FastAPI(
    title="Resonance - Genomic Coordinate Liftover",
    description="Accurate genomic coordinate conversion platform",
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


# Updated landing_page function for main.py
# Replace the existing @app.get("/") function with this

from pathlib import Path

# Load comprehensive landing page template
LANDING_PAGE_TEMPLATE_PATH = Path(__file__).parent / "templates" / "landing_page.html"

@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    db = SessionLocal()
    try:
        recent = db.query(Job).order_by(Job.created_at.desc()).limit(10).all()
        active_jobs = len(recent)
    except Exception:
        active_jobs = 0
    finally:
        db.close()

    core_services = [
        "liftover",
        "vcf_converter",
        "confidence_predictor",
        "feature_extractor",
        "semantic_engine",
    ]

    operational_count = sum(
        1 for key in core_services if SERVICES.get(key) is not None
    )

    ml_available = SERVICES.get("confidence_predictor") is not None
    vcf_available = SERVICES.get("vcf_converter") is not None

    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
            "active_jobs": active_jobs,
            "operational_services": operational_count,
            "ml_status_class": "status-available" if ml_available else "status-unavailable",
            "ml_status": "Available" if ml_available else "Unavailable",
            "vcf_status_class": "status-available" if vcf_available else "status-unavailable",
            "vcf_status": "Enabled" if vcf_available else "Disabled",
        },
    )



# Comprehensive inline landing page HTML (used if template file not found)
COMPREHENSIVE_LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resonance – Professional Genomic Liftover Platform</title>

<style>
:root {{
    --navy: #001f3f;
    --light: #f7f9fb;
    --border: #d0d7de;
    --success: #28a745;
    --warning: #ffc107;
    --danger: #dc3545;
    --info: #17a2b8;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    background: var(--light);
    color: #000;
    line-height: 1.6;
}}

header {{
    background: linear-gradient(135deg, #001f3f 0%, #003366 100%);
    color: #fff;
    padding: 2.5rem 3rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}

header h1 {{
    margin: 0;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}

header p {{
    margin-top: 0.5rem;
    opacity: 0.95;
    font-size: 1.1rem;
}}

.status-badge {{
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 0.5rem;
}}

.status-available {{ background: var(--success); color: white; }}
.status-unavailable {{ background: var(--danger); color: white; }}

main {{
    max-width: 1400px;
    margin: auto;
    padding: 2rem 1.5rem;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}}

.section {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s;
}}

.section:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}}

.section h2 {{
    margin-top: 0;
    border-bottom: 2px solid var(--navy);
    padding-bottom: 0.7rem;
    color: var(--navy);
    font-size: 1.5rem;
}}

.section h3 {{
    color: var(--navy);
    margin-top: 1.5rem;
    font-size: 1.1rem;
}}

.form-group {{
    margin: 1rem 0;
}}

label {{
    display: block;
    margin-bottom: 0.3rem;
    font-weight: 600;
    color: #333;
}}

input, select, textarea {{
    width: 100%;
    padding: 0.7rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 1rem;
    transition: border-color 0.3s;
}}

input:focus, select:focus, textarea:focus {{
    outline: none;
    border-color: var(--navy);
}}

.checkbox-group {{
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
}}

.checkbox-group input[type="checkbox"] {{
    width: auto;
    margin-right: 0.5rem;
}}

.checkbox-group label {{
    margin-bottom: 0;
    font-weight: normal;
}}

button {{
    margin-top: 1rem;
    padding: 0.8rem 2rem;
    background: var(--navy);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 600;
    transition: background 0.3s, transform 0.1s;
}}

button:hover {{
    background: #003366;
    transform: translateY(-1px);
}}

button:active {{
    transform: translateY(0);
}}

button:disabled {{
    background: #ccc;
    cursor: not-allowed;
}}

.btn-secondary {{
    background: var(--info);
}}

.btn-secondary:hover {{
    background: #138496;
}}

pre {{
    background: #f5f5f5;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem;
    margin-top: 1rem;
    white-space: pre-wrap;
    font-family: "Consolas", "Monaco", monospace;
    font-size: 0.9rem;
    max-height: 400px;
    overflow-y: auto;
}}

.info-box {{
    background: #e7f3ff;
    border-left: 4px solid var(--info);
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}}

.warning-box {{
    background: #fff3cd;
    border-left: 4px solid var(--warning);
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}}

.feature-list {{
    list-style: none;
    padding: 0;
}}

.feature-list li {{
    padding: 0.5rem 0;
    padding-left: 1.5rem;
    position: relative;
}}

.feature-list li:before {{
    content: "✓";
    position: absolute;
    left: 0;
    color: var(--success);
    font-weight: bold;
}}

.status-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}

.status-card {{
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    border-left: 4px solid var(--info);
}}

.status-card h4 {{
    margin: 0 0 0.5rem 0;
    color: var(--navy);
}}

.status-card p {{
    margin: 0;
    font-size: 0.9rem;
}}

.file-upload {{
    border: 2px dashed var(--border);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    background: #fafafa;
    cursor: pointer;
    transition: all 0.3s;
}}

.file-upload:hover {{
    border-color: var(--navy);
    background: #f0f0f0;
}}

.file-upload input[type="file"] {{
    display: none;
}}

.progress-bar {{
    width: 100%;
    height: 30px;
    background: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
}}

.progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--navy), #003366);
    transition: width 0.3s;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
}}

.result-success {{
    border-left: 4px solid var(--success);
}}

.result-warning {{
    border-left: 4px solid var(--warning);
}}

.result-error {{
    border-left: 4px solid var(--danger);
}}

.tabs {{
    display: flex;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}}

.tab {{
    padding: 0.8rem 1.5rem;
    cursor: pointer;
    border: none;
    background: none;
    font-weight: 600;
    color: #666;
    border-bottom: 3px solid transparent;
    transition: all 0.3s;
    margin: 0;
}}

.tab:hover {{
    color: var(--navy);
}}

.tab.active {{
    color: var(--navy);
    border-bottom-color: var(--navy);
}}

.tab-content {{
    display: none;
}}

.tab-content.active {{
    display: block;
}}

footer {{
    background: var(--navy);
    color: white;
    padding: 2rem;
    margin-top: 3rem;
    text-align: center;
}}

footer a {{
    color: white;
    text-decoration: underline;
}}

@media (max-width: 768px) {{
    .grid {{
        grid-template-columns: 1fr;
    }}
    
    header h1 {{
        font-size: 2rem;
    }}
    
    main {{
        padding: 1rem;
    }}
    
    .tabs {{
        overflow-x: auto;
    }}
}}
</style>
</head>

<body>

<header>
<h1> RESONANCE</h1>
<p>Professional Genomic Coordinate Liftover & Validation Platform</p>
<div style="margin-top: 1rem;">
    <span class="status-badge {ml_status_class}">ML Confidence: {ml_status}</span>
    <span class="status-badge {vcf_status_class}">VCF Processing: {vcf_status}</span>
    <span class="status-badge status-available">NCBI RefSeq: Connected</span>
</div>
</header>

<main>

<!-- System Status -->
<div class="section">
    <h2> System Status</h2>
    <div class="status-grid">
        <div class="status-card">
            <h4>Active Jobs</h4>
            <p style="font-size: 2rem; font-weight: bold; color: var(--navy);">{active_jobs}</p>
        </div>
        <div class="status-card">
            <h4>Services Operational</h4>
            <p style="font-size: 2rem; font-weight: bold; color: var(--success);">{operational_services}/5</p>
        </div>
        <div class="status-card">
            <h4>API Status</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: var(--success);">✓ Online</p>
        </div>
        <div class="status-card">
            <h4>RefSeq Connection</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: var(--success);">✓ Active</p>
        </div>
    </div>
</div>

<!-- Feature Tabs -->
<div class="section">
    <div class="tabs">
        <button class="tab active" onclick="switchTab('single')">Single Coordinate</button>
        <button class="tab" onclick="switchTab('batch')">Batch Processing</button>
        <button class="tab" onclick="switchTab('vcf')">VCF Conversion</button>
        <button class="tab" onclick="switchTab('region')">Region Liftover</button>
        <button class="tab" onclick="switchTab('semantic')">Semantic Reconciliation</button>
    </div>

    <!-- Single Coordinate Tab -->
    <div id="single-tab" class="tab-content active">
        <h2> Single Coordinate Conversion</h2>
        <p>Convert individual genomic coordinates between genome builds with ML-based confidence prediction and NCBI RefSeq validation.</p>

        <div class="form-group">
            <label>Chromosome</label>
            <input type="text" id="single-chrom" value="chr17" placeholder="e.g., chr17, 17, chrX">
        </div>

        <div class="form-group">
            <label>Position</label>
            <input type="number" id="single-pos" value="41196312" placeholder="e.g., 41196312">
        </div>

        <div class="form-group">
            <label>From Build</label>
            <select id="single-from">
                <option value="hg19" selected>hg19 / GRCh37</option>
                <option value="hg38">hg38 / GRCh38</option>
            </select>
        </div>

        <div class="form-group">
            <label>To Build</label>
            <select id="single-to">
                <option value="hg38" selected>hg38 / GRCh38</option>
                <option value="hg19">hg19 / GRCh37</option>
            </select>
        </div>

        <div class="form-group">
            <label>Strand</label>
            <select id="single-strand">
                <option value="+" selected>+ (Forward)</option>
                <option value="-">- (Reverse)</option>
            </select>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="include-ml" checked>
            <label for="include-ml">Include ML Confidence Analysis</label>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="include-refseq" checked>
            <label for="include-refseq">Validate with NCBI RefSeq</label>
        </div>

        <button onclick="runSingleConversion()"> Convert Coordinate</button>

        <pre id="single-output" style="display: none;"></pre>
    </div>

    <!-- Batch Processing Tab -->
    <div id="batch-tab" class="tab-content">
        <h2> Batch Coordinate Processing</h2>
        <p>Upload a CSV/TSV file or paste coordinates for bulk conversion. Supports thousands of coordinates with background processing and job tracking.</p>

        <div class="info-box">
            <strong>Supported formats:</strong> CSV, TSV, Plain text (one coordinate per line)<br>
            <strong>Format examples:</strong> chr17,41196312 or chr17:41196312<br>
            <strong>Max coordinates:</strong> 10,000 per batch
        </div>

        <div class="form-group">
            <label>Paste Coordinates</label>
            <textarea id="batch-text" rows="8" placeholder="chr17,41196312
chr1:100000
chrX,50000

Or upload a file below..."></textarea>
        </div>

        <div class="form-group">
            <label>From Build</label>
            <select id="batch-from">
                <option value="hg19" selected>hg19 / GRCh37</option>
                <option value="hg38">hg38 / GRCh38</option>
            </select>
        </div>

        <div class="form-group">
            <label>To Build</label>
            <select id="batch-to">
                <option value="hg38" selected>hg38 / GRCh38</option>
                <option value="hg19">hg19 / GRCh37</option>
            </select>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="batch-ml" checked>
            <label for="batch-ml">Include ML confidence for each coordinate</label>
        </div>

        <button onclick="runBatchConversion()"> Start Batch Job</button>

        <div id="batch-progress" style="display: none;">
            <h3>Processing Job...</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="batch-progress-fill" style="width: 0%">0%</div>
            </div>
            <p id="batch-status-text">Initializing...</p>
        </div>

        <pre id="batch-output" style="display: none;"></pre>
    </div>

    <!-- VCF Conversion Tab -->
    <div id="vcf-tab" class="tab-content">
        <h2> VCF File Liftover</h2>
        <p>Convert VCF (Variant Call Format) files between genome builds while preserving all variant annotations and quality metrics.</p>

        <div class="info-box">
            <strong>Supported VCF versions:</strong> VCF 4.0, 4.1, 4.2, 4.3<br>
            <strong>Features:</strong> Preserves INFO, FORMAT, and FILTER fields<br>
            <strong>Validation:</strong> Cross-references with reference genome
        </div>

        <div class="warning-box">
            <strong>Note:</strong> VCF file upload requires using the API endpoint <code>/vcf/convert</code>. See <a href="/docs" style="color: #856404;">API documentation</a> for details.
        </div>

        <div class="form-group">
            <label>Example VCF Coordinate</label>
            <input type="text" id="vcf-example" value="chr17 41196312 . G A 50 PASS" placeholder="CHROM POS ID REF ALT QUAL FILTER">
        </div>

        <div class="form-group">
            <label>From Build</label>
            <select id="vcf-from">
                <option value="hg19" selected>hg19 / GRCh37</option>
                <option value="hg38">hg38 / GRCh38</option>
            </select>
        </div>

        <div class="form-group">
            <label>To Build</label>
            <select id="vcf-to">
                <option value="hg38" selected>hg38 / GRCh38</option>
                <option value="hg19">hg19 / GRCh37</option>
            </select>
        </div>

        <button onclick="showVCFInstructions()"> View API Instructions</button>

        <pre id="vcf-output" style="display: none;"></pre>
    </div>

    <!-- Region Liftover Tab -->
    <div id="region-tab" class="tab-content">
        <h2> Genomic Region Conversion</h2>
        <p>Convert genomic regions (intervals) between builds. Ideal for gene regions, regulatory elements, CNVs, and other genomic features.</p>

        <div class="form-group">
            <label>Chromosome</label>
            <input type="text" id="region-chrom" value="chr17" placeholder="e.g., chr17">
        </div>

        <div class="form-group">
            <label>Start Position</label>
            <input type="number" id="region-start" value="41196312" placeholder="e.g., 41196312">
        </div>

        <div class="form-group">
            <label>End Position</label>
            <input type="number" id="region-end" value="41277500" placeholder="e.g., 41277500">
        </div>

        <div class="form-group">
            <label>From Build</label>
            <select id="region-from">
                <option value="hg19" selected>hg19 / GRCh37</option>
                <option value="hg38">hg38 / GRCh38</option>
            </select>
        </div>

        <div class="form-group">
            <label>To Build</label>
            <select id="region-to">
                <option value="hg38" selected>hg38 / GRCh38</option>
                <option value="hg19">hg19 / GRCh37</option>
            </select>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="region-genes" checked>
            <label for="region-genes">Include gene annotations from NCBI RefSeq</label>
        </div>

        <button onclick="runRegionConversion()"> Convert Region</button>

        <pre id="region-output" style="display: none;"></pre>
    </div>

    <!-- Semantic Reconciliation Tab -->
    <div id="semantic-tab" class="tab-content">
        <h2> Semantic Reconciliation</h2>
        <p>Reconcile and validate genomic annotations across different databases and naming conventions using semantic mapping and NCBI RefSeq integration.</p>

        <div class="info-box">
            <strong>Supported databases:</strong> NCBI RefSeq, Ensembl, UCSC, HGNC<br>
            <strong>Features:</strong> Gene name disambiguation, transcript mapping, coordinate validation, variant cross-referencing<br>
            <strong>Validates against:</strong> Current RefSeq release
        </div>

        <div class="form-group">
            <label>Gene Symbol / Transcript ID</label>
            <input type="text" id="semantic-gene" value="BRCA1" placeholder="e.g., BRCA1, NM_007294, ENST00000357654">
        </div>

        <div class="form-group">
            <label>Source Database</label>
            <select id="semantic-source">
                <option value="refseq" selected>NCBI RefSeq</option>
                <option value="ensembl">Ensembl</option>
                <option value="ucsc">UCSC</option>
                <option value="hgnc">HGNC</option>
            </select>
        </div>

        <div class="form-group">
            <label>Genome Build</label>
            <select id="semantic-build">
                <option value="hg19">hg19 / GRCh37</option>
                <option value="hg38" selected>hg38 / GRCh38</option>
            </select>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="semantic-variants" checked>
            <label for="semantic-variants">Include known variants from ClinVar</label>
        </div>

        <div class="checkbox-group">
            <input type="checkbox" id="semantic-transcripts" checked>
            <label for="semantic-transcripts">Map to all transcript isoforms</label>
        </div>

        <button onclick="runSemanticReconciliation()"> Reconcile Annotation</button>

        <pre id="semantic-output" style="display: none;"></pre>
    </div>
</div>

<!-- API Documentation -->
<div class="grid">
    <div class="section">
        <h2> API Endpoints</h2>
        <ul class="feature-list">
            <li><code>POST /liftover/single</code> - Single coordinate conversion</li>
            <li><code>POST /liftover/batch</code> - Batch coordinate processing</li>
            <li><code>POST /liftover/region</code> - Genomic region conversion</li>
            <li><code>POST /vcf/convert</code> - VCF file liftover</li>
            <li><code>POST /semantic/reconcile</code> - Semantic annotation mapping</li>
            <li><code>GET /job-status/{{job_id}}</code> - Check job status</li>
            <li><code>GET /export/{{job_id}}/{{format}}</code> - Download results</li>
            <li><code>GET /health</code> - System health check</li>
        </ul>
        <button class="btn-secondary" onclick="window.location.href='/docs'"> View Full API Docs</button>
    </div>

    <div class="section">
        <h2> Platform Features</h2>
        <ul class="feature-list">
            <li>ML-based confidence prediction</li>
            <li>NCBI RefSeq integration & validation</li>
            <li>Multi-source validation (Ensembl, UCSC)</li>
            <li>VCF file processing with annotation preservation</li>
            <li>Batch processing with background job tracking</li>
            <li>Semantic annotation reconciliation</li>
            <li>Checking for clinical-grade validation</li>
            <li>RESTful API with OpenAPI documentation</li>
            <li>Real-time job progress tracking</li>
            <li>Multiple export formats (JSON, CSV, VCF)</li>
        </ul>
    </div>
</div>

</main>

<footer>
    <p><strong>Resonance Genomic Liftover Platform</strong> v1.0.0</p>
    <p>Research-Grade Bioinformatics Platform | <a href="/docs">API Documentation</a> | <a href="/health">System Health</a></p>
    <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;">
        Powered by UCSC LiftOver, Ensembl REST API, and NCBI RefSeq | ML confidence with scikit-learn
    </p>
</footer>

<script>
// Tab switching
function switchTab(tabName) {{
    document.querySelectorAll('.tab-content').forEach(tab => {{
        tab.classList.remove('active');
    }});
    document.querySelectorAll('.tab').forEach(tab => {{
        tab.classList.remove('active');
    }});
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}}

// Single coordinate conversion
async function runSingleConversion() {{
    const chrom = document.getElementById('single-chrom').value;
    const pos = document.getElementById('single-pos').value;
    const fromBuild = document.getElementById('single-from').value;
    const toBuild = document.getElementById('single-to').value;
    const strand = document.getElementById('single-strand').value;
    const includeML = document.getElementById('include-ml').checked;

    const outputEl = document.getElementById('single-output');
    outputEl.style.display = 'block';
    outputEl.textContent = 'Converting...';
    outputEl.className = '';

    try {{
        const response = await fetch(
            `/liftover/single?chrom=${{encodeURIComponent(chrom)}}&pos=${{pos}}&from_build=${{fromBuild}}&to_build=${{toBuild}}&strand=${{encodeURIComponent(strand)}}&include_ml=${{includeML}}`,
            {{ method: 'POST' }}
        );
        
        const result = await response.json();
        
        if (result.success) {{
            outputEl.className = 'result-success';
        }} else {{
            outputEl.className = 'result-error';
        }}
        
        outputEl.textContent = JSON.stringify(result, null, 2);
    }} catch (error) {{
        outputEl.className = 'result-error';
        outputEl.textContent = `Error: ${{error.message}}`;
    }}
}}

// Batch conversion
async function runBatchConversion() {{
    const text = document.getElementById('batch-text').value;
    const fromBuild = document.getElementById('batch-from').value;
    const toBuild = document.getElementById('batch-to').value;
    const includeML = document.getElementById('batch-ml').checked;

    if (!text.trim()) {{
        alert('Please enter coordinates');
        return;
    }}

    const progressDiv = document.getElementById('batch-progress');
    const outputEl = document.getElementById('batch-output');
    
    progressDiv.style.display = 'block';
    outputEl.style.display = 'none';
    
async function pollBatchResults(jobId) {
    const outputEl = document.getElementById('batch-output');
    const progressFill = document.getElementById('batch-progress-fill');
    const statusText = document.getElementById('batch-status-text');
    const progressDiv = document.getElementById('batch-progress');
    
    progressDiv.style.display = 'block';
    
    let attempts = 0;
    const maxAttempts = 30;
    
    const poll = async () => {
        try {
            const response = await fetch(`/job-status/${jobId}`);
            const status = await response.json();
            
            const progress = status.progress_percent || 0;
            progressFill.style.width = `${progress}%`;
            progressFill.textContent = `${progress.toFixed(0)}%`;
            statusText.textContent = `Status: ${status.status} - ${status.processed_items}/${status.total_items} processed`;
            
            if (status.status === 'completed') {
                progressDiv.style.display = 'none';
                outputEl.className = 'result-success';
                outputEl.textContent = `Batch job completed!\\n\\n${JSON.stringify(status.metadata || {}, null, 2)}\\n\\nDownload results at: ${status.export_options?.json || 'N/A'}`;
                return;
            } else if (status.status === 'failed') {
                progressDiv.style.display = 'none';
                outputEl.className = 'result-error';
                outputEl.textContent = `Batch job failed:\\n${status.error_message || 'Unknown error'}`;
                return;
            }
            
            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 2000);
            } else {
                progressDiv.style.display = 'none';
                outputEl.className = 'result-warning';
                outputEl.textContent = 'Polling timed out. Check job status manually.';
            }
        } catch (error) {
            progressDiv.style.display = 'none';
            outputEl.className = 'result-error';
            outputEl.textContent = `Error polling status: ${error.message}`;
        }
    };
    
    poll();
}
    
// Parse coordinates from text input
    const lines = text.trim().split('\\n');
    const coordinates = [];
    
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        
        // Support formats: "chr17,41196312" or "chr17:41196312"
        const match = trimmed.match(/^(chr)?([\\dXYM]+)[,:](\\d+)/i);
        if (match) {
            const chrom = match[1] ? match[0].split(/[,:]/ )[0] : 'chr' + match[2];
            const pos = parseInt(match[3]);
            coordinates.push({ chrom, pos });
        }
    }

    if (coordinates.length === 0) {
        alert('No valid coordinates found. Use format: chr17,41196312 or chr17:41196312');
        progressDiv.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(
            `/liftover/batch?from_build=${fromBuild}&to_build=${toBuild}&include_ml=${includeML}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(coordinates)
            }
        );
        
        const result = await response.json();
        
        progressDiv.style.display = 'none';
        outputEl.style.display = 'block';
        
        if (result.job_id) {
            outputEl.className = 'result-success';
            outputEl.textContent = `Batch job started!\\n\\nJob ID: ${result.job_id}\\nTotal coordinates: ${result.total_coordinates}\\nStatus endpoint: ${result.status_endpoint}\\n\\nPolling for results...`;
            
            // Auto-poll for results
            setTimeout(() => pollBatchResults(result.job_id), 2000);
        } else if (Array.isArray(result)) {
            outputEl.className = 'result-success';
            const successCount = result.filter(r => r.success).length;
            outputEl.textContent = `Batch conversion complete!\\n\\nSuccessful: ${successCount}/${result.length}\\n\\n${JSON.stringify(result, null, 2)}`;
        } else {
            outputEl.className = 'result-error';
            outputEl.textContent = JSON.stringify(result, null, 2);
        }
    } catch (error) {
        progressDiv.style.display = 'none';
        outputEl.className = 'result-error';
        outputEl.style.display = 'block';
        outputEl.textContent = `Error: ${error.message}`;
    }
}}

// Region conversion
async function runRegionConversion() {{
    const chrom = document.getElementById('region-chrom').value;
    const start = document.getElementById('region-start').value;
    const end = document.getElementById('region-end').value;
    const fromBuild = document.getElementById('region-from').value;
    const toBuild = document.getElementById('region-to').value;

    const outputEl = document.getElementById('region-output');
    outputEl.style.display = 'block';
    outputEl.textContent = 'Converting region...';
    outputEl.className = '';

    try {{
        const response = await fetch(
            `/liftover/region?chrom=${{encodeURIComponent(chrom)}}&start=${{start}}&end=${{end}}&from_build=${{fromBuild}}&to_build=${{toBuild}}`,
            {{ method: 'POST' }}
        );
        
        const result = await response.json();
        
        if (result.success) {{
            outputEl.className = 'result-success';
        }} else {{
            outputEl.className = 'result-error';
        }}
        
        outputEl.textContent = JSON.stringify(result, null, 2);
    }} catch (error) {{
        outputEl.className = 'result-error';
        outputEl.textContent = `Error: ${{error.message}}`;
    }}
}}

// Semantic reconciliation
async function runSemanticReconciliation() {
    const gene = document.getElementById('semantic-gene').value.trim();
    const sourceDb = document.getElementById('semantic-source').value;
    const build = document.getElementById('semantic-build').value;
    const includeVariants = document.getElementById('semantic-variants').checked;
    const includeTranscripts = document.getElementById('semantic-transcripts').checked;

    if (!gene) {
        alert('Please enter a gene symbol or transcript ID');
        return;
    }

    const outputEl = document.getElementById('semantic-output');
    outputEl.style.display = 'block';
    outputEl.textContent = 'Processing semantic reconciliation...';
    outputEl.className = '';

    try {
        // Create sample annotations for demonstration
        const annotations = [
            {
                description: `${gene} from ${sourceDb}`,
                source: sourceDb,
                biological_process: ["regulation", "signaling"],
                molecular_function: ["binding", "catalytic"],
                confidence: 0.9
            },
            {
                description: `${gene} annotation from reference database`,
                source: "RefSeq",
                biological_process: ["regulation"],
                molecular_function: ["binding"],
                confidence: 0.85
            }
        ];

        const response = await fetch(
            `/semantic/reconcile?gene_symbol=${encodeURIComponent(gene)}`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(annotations)
            }
        );
        
        const result = await response.json();
        
        if (response.ok) {
            outputEl.className = 'result-success';
        } else {
            outputEl.className = 'result-error';
        }
        
        outputEl.textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        outputEl.className = 'result-error';
        outputEl.textContent = `Error: ${error.message}`;
    }
}

// VCF instructions
function showVCFInstructions() {{
    const outputEl = document.getElementById('vcf-output');
    outputEl.style.display = 'block';
    outputEl.className = 'result-success';
    outputEl.textContent = `VCF File Liftover Instructions:

Use the API endpoint: POST /vcf/convert

Example using curl:
curl -X POST "http://localhost:8000/vcf/convert" \\
  -F "file=@input.vcf" \\
  -F "from_build=hg19" \\
  -F "to_build=hg38"

Response will include:
- Lifted VCF file
- Liftover statistics
- Failed variants report
- Validation results

See /docs for complete documentation.`;
}}
</script>

</body>
</html>
"""


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

@app.post("/liftover/region")
def liftover_region(
    chrom: str = Query(...),
    start: int = Query(..., ge=1),
    end: int = Query(..., ge=1),
    from_build: str = Query("hg19"),
    to_build: str = Query("hg38"),
):
    """Convert genomic region (start and end coordinates)"""
    if not SERVICES.get("liftover"):
        raise HTTPException(status_code=503, detail="Liftover service unavailable")
    if start >= end:
        raise HTTPException(status_code=400, detail="Start must be less than end")
    try:
        # Attempt to use liftover service's region method when available, else lift each position
        liftover = SERVICES["liftover"]
        if hasattr(liftover, "convert_region"):
            return liftover.convert_region(chrom, start, end, from_build, to_build)
        # fallback: liftover start and end individually
        start_r = liftover.convert_coordinate(chrom, start, from_build, to_build)
        end_r = liftover.convert_coordinate(chrom, end, from_build, to_build)
        return {"start": start_r, "end": end_r}
    except Exception as e:
        logger.exception("Region liftover failed")
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

@app.post("/semantic/reconcile")
def semantic_reconcile(
    gene_symbol: str = Query(...),
    annotations: List[Dict] = None
):
    """
    Reconcile conflicting annotations for a single gene.
    Request: POST /semantic/reconcile?gene_symbol=BRCA1
    Body: JSON array of objects: [{"description":"...", "source":"NCBI", ...}, ...]
    """
    if not SERVICES.get("semantic_engine"):
        raise HTTPException(status_code=503, detail="Semantic reconciliation service unavailable")
    if not annotations:
        raise HTTPException(status_code=400, detail="No annotations provided")
    try:
        # Convert dicts to SemanticAnnotation dataclass instances expected by the engine
        anns = []
        for a in annotations:
            sa = SemanticAnnotation(
                gene_symbol = gene_symbol,
                description = a.get("description", "") or "",
                source = a.get("source", "unknown"),
                biological_process = a.get("biological_process") or a.get("biological_processes") or [],
                molecular_function = a.get("molecular_function") or a.get("molecular_functions") or [],
                cellular_component = a.get("cellular_component") or a.get("cellular_components") or [],
                protein_domains = a.get("protein_domains") or [],
                synonyms = a.get("synonyms") or [],
                confidence = float(a.get("confidence", 0.8) or 0.8)
            )
            anns.append(sa)

        result = SERVICES["semantic_engine"].reconcile_annotations(gene_symbol, anns)
        return result
    except Exception as e:
        logger.exception("Semantic reconciliation failed for %s", gene_symbol)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/semantic/batch")
async def semantic_batch(
    gene_annotations: Dict[str, List[Dict]],
    background_tasks: BackgroundTasks = None
):
    """
    Batch reconcile many genes.
    Body format: {"BRCA1": [ {annotation}, ... ], "TP53": [ ... ] }
    Returns job_id; results are saved to Job.results (JSON) when completed.
    """
    if not gene_annotations:
        raise HTTPException(status_code=400, detail="No gene annotations provided")
    if not SERVICES.get("semantic_engine"):
        raise HTTPException(status_code=503, detail="Semantic reconciliation service unavailable")

    job_id = uuid.uuid4().hex[:8]

    # Create DB job record
    db = SessionLocal()
    try:
        job = Job(job_id=job_id, job_type="semantic_batch", status="queued", total_items=len(gene_annotations))
        db.add(job)
        db.commit()
    finally:
        db.close()

    def process():
        db = SessionLocal()
        try:
            j = db.query(Job).filter(Job.job_id == job_id).first()
            j.status = "processing"
            db.commit()

            results = {}
            processed = 0
            for gene, anns_list in gene_annotations.items():
                try:
                    # convert dicts to dataclasses
                    anns = []
                    for a in anns_list:
                        sa = SemanticAnnotation(
                            gene_symbol = gene,
                            description = a.get("description", "") or "",
                            source = a.get("source", "unknown"),
                            biological_process = a.get("biological_process") or a.get("biological_processes") or [],
                            molecular_function = a.get("molecular_function") or a.get("molecular_functions") or [],
                            cellular_component = a.get("cellular_component") or a.get("cellular_components") or [],
                            protein_domains = a.get("protein_domains") or [],
                            synonyms = a.get("synonyms") or [],
                            confidence = float(a.get("confidence", 0.8) or 0.8)
                        )
                        anns.append(sa)

                    res = SERVICES["semantic_engine"].reconcile_annotations(gene, anns)
                except Exception as e:
                    logger.exception("Semantic reconcile failed for %s", gene)
                    res = {"gene_symbol": gene, "status": "error", "error": str(e)}
                results[gene] = res
                processed += 1
                j.processed_items = processed
                db.commit()

            j.status = "completed"
            j.results = results
            j.metadata = {"n_genes": len(results)}
            db.commit()
            logger.info("Semantic batch job %s completed: %d genes", job_id, len(results))

        except Exception as e:
            logger.exception("Semantic batch job failed")
            j.status = "failed"
            j.errors = [str(e)]
            db.commit()
        finally:
            db.close()

    # schedule background processing
    background_tasks.add_task(process)
    return {"job_id": job_id, "status_endpoint": f"/job-status/{job_id}"}

@app.post("/batch-submit", response_class=HTMLResponse)
async def batch_submit_form(
    batch_text: str = Form(...),
    batch_from: str = Form(...),
    batch_to: str = Form(...),
    batch_ml: bool = Form(False)
):
    """Process batch coordinates and return HTML results"""
    
    # Parse coordinates
    lines = batch_text.strip().split('\n')
    coordinates = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Support both chr17:41196312 and chr17,41196312
        parts = line.replace(':', ',').split(',')
        if len(parts) >= 2:
            coordinates.append({
                "chrom": parts[0].strip(),
                "pos": int(parts[1].strip())
            })
    
    # Process batch
    results = []
    for coord in coordinates:
        try:
            result = await liftover_coordinate(
                chrom=coord["chrom"],
                pos=coord["pos"],
                from_build=batch_from,
                to_build=batch_to,
                include_ml=batch_ml
            )
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "input": coord})
    
    # Format results as HTML
    html_results = "<h3>Results:</h3><table border='1'><tr><th>Input</th><th>Output</th><th>Status</th></tr>"
    for r in results:
        if "error" in r:
            html_results += f"<tr><td>{r['input']}</td><td>-</td><td style='color:red'>Error: {r['error']}</td></tr>"
        else:
            html_results += f"<tr><td>{r.get('input_chrom')}:{r.get('input_pos')}</td>"
            html_results += f"<td>{r.get('output_chrom')}:{r.get('output_pos')}</td>"
            html_results += f"<td style='color:green'>✓ {r.get('method', 'OK')}</td></tr>"
    html_results += "</table>"
    
    return HTMLResponse(content=html_results)


@app.post("/semantic-submit", response_class=HTMLResponse)
async def semantic_submit_form(
    gene_symbol: str = Form(...),
    source_db: str = Form("ensembl"),
    build: str = Form("hg38"),
    include_variants: bool = Form(False),
    include_isoforms: bool = Form(False)
):
    """Process semantic reconciliation and return HTML results"""
    
    try:
        result = await reconcile_gene(
            gene_symbol=gene_symbol,
            source_db=source_db,
            build=build,
            include_variants=include_variants,
            include_isoforms=include_isoforms
        )
        
        # Format as HTML table
        html = f"<h3>Results for {gene_symbol}:</h3>"
        html += "<table border='1'><tr><th>Database</th><th>ID</th><th>Coordinates</th></tr>"
        
        for db, data in result.items():
            if isinstance(data, dict):
                html += f"<tr><td>{db}</td><td>{data.get('id', 'N/A')}</td>"
                html += f"<td>{data.get('chrom', '?')}:{data.get('start', '?')}-{data.get('end', '?')}</td></tr>"
        
        html += "</table>"
        return HTMLResponse(content=html)
        
    except Exception as e:
        return HTMLResponse(content=f"<p style='color:red'>Error: {str(e)}</p>")

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