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

# Global state
startup_time = time.time()
job_storage: Dict[str, Any] = {}

# Service initialization
SERVICES = {}

def initialize_services():
    """Initialize all services with error handling"""
    global SERVICES
    
    service_configs = [
        ('liftover', 'app.services.real_liftover', 'RealLiftoverService', None),
        ('vcf_converter', 'app.services.vcf_converter', 'VCFConverter', 'liftover'),
        ('feature_extractor', 'app.services.feature_extractor', 'FeatureExtractor', None),
        ('confidence_predictor', 'app.services.confidence_predictor', 'ConfidencePredictor', None),
        ('validation_engine', 'app.services.validation_engine', 'ValidationEngine', None),
    ]
    
    for service_name, module_path, class_name, dependency in service_configs:
        try:
            module = __import__(module_path, fromlist=[class_name])
            service_class = getattr(module, class_name)
            
            if dependency and dependency not in SERVICES:
                logger.warning(f"Service {service_name} skipped: dependency '{dependency}' not available")
                SERVICES[service_name] = None
                continue
            
            if dependency:
                SERVICES[service_name] = service_class(SERVICES[dependency])
            else:
                SERVICES[service_name] = service_class()
            
            logger.info(f"Service {service_name} initialized")
        except Exception as e:
            logger.error(f"Service {service_name} failed: {e}")
            SERVICES[service_name] = None
    
    return bool(SERVICES.get('liftover'))

SERVICES_AVAILABLE = initialize_services()

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
    """Professional landing page"""
    active_jobs = len([j for j in job_storage.values() if j.status in ["queued", "processing"]])
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resonance - Genomic Liftover</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Times New Roman', Times, serif;
                background: #ffffff;
                color: #000000;
                min-height: 100vh;
            }}
            .header {{
                background: #001f3f;
                color: #ffffff;
                padding: 2rem 0;
                border-bottom: 3px solid #000000;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            h1 {{
                font-size: 2.5em;
                font-weight: normal;
                letter-spacing: 2px;
                margin-bottom: 0.5rem;
            }}
            .subtitle {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content {{
                padding: 2rem 0;
            }}
            .card {{
                background: #ffffff;
                border: 2px solid #001f3f;
                padding: 2rem;
                margin-bottom: 2rem;
            }}
            .card h2 {{
                color: #001f3f;
                font-size: 1.8em;
                font-weight: normal;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid #001f3f;
                padding-bottom: 0.5rem;
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }}
            .status-item {{
                border: 1px solid #001f3f;
                padding: 1.5rem;
                text-align: center;
            }}
            .status-value {{
                font-size: 2em;
                color: #001f3f;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }}
            .status-label {{
                color: #000000;
                font-size: 0.95em;
            }}
            .demo-section {{
                margin: 2rem 0;
                padding: 1.5rem;
                border: 1px solid #001f3f;
            }}
            input, select, textarea {{
                width: 100%;
                padding: 0.75rem;
                border: 1px solid #001f3f;
                font-family: 'Times New Roman', Times, serif;
                font-size: 1rem;
                margin: 0.5rem 0;
            }}
            button {{
                background: #001f3f;
                color: #ffffff;
                padding: 0.75rem 2rem;
                border: none;
                cursor: pointer;
                font-family: 'Times New Roman', Times, serif;
                font-size: 1rem;
                margin: 0.5rem 0.5rem 0.5rem 0;
            }}
            button:hover {{
                background: #003366;
            }}
            .btn-secondary {{
                background: #666666;
            }}
            .btn-secondary:hover {{
                background: #444444;
            }}
            .result-box {{
                background: #f5f5f5;
                padding: 1rem;
                border: 1px solid #001f3f;
                margin-top: 1rem;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                max-height: 400px;
                overflow-y: auto;
                display: none;
            }}
            .loading {{
                display: none;
                color: #001f3f;
                margin-top: 1rem;
            }}
            .error {{
                color: #cc0000;
                background: #ffe6e6;
                padding: 1rem;
                border: 1px solid #cc0000;
                margin-top: 1rem;
            }}
            .success {{
                color: #006600;
                background: #e6ffe6;
                padding: 1rem;
                border: 1px solid #006600;
                margin-top: 1rem;
            }}
            .tab {{
                display: inline-block;
                padding: 0.75rem 1.5rem;
                background: #f5f5f5;
                border: 1px solid #001f3f;
                cursor: pointer;
                margin-right: 0.25rem;
            }}
            .tab.active {{
                background: #001f3f;
                color: #ffffff;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>RESONANCE</h1>
                <p class="subtitle">Genomic Coordinate Liftover Platform</p>
            </div>
        </div>
        
        <div class="container content">
            <div class="card">
                <h2>System Status</h2>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-value">{"OPERATIONAL" if SERVICES_AVAILABLE else "LIMITED"}</div>
                        <div class="status-label">System Status</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value">{active_jobs}</div>
                        <div class="status-label">Active Jobs</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value">{"YES" if SERVICES.get('confidence_predictor') else "NO"}</div>
                        <div class="status-label">ML Confidence</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value">{"YES" if SERVICES.get('vcf_converter') else "NO"}</div>
                        <div class="status-label">VCF Support</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div style="border-bottom: 1px solid #001f3f; margin-bottom: 2rem;">
                    <div class="tab active" onclick="switchTab('demo')">Live Demo</div>
                    <div class="tab" onclick="switchTab('batch')">Batch Processing</div>
                    <div class="tab" onclick="switchTab('docs')">Documentation</div>
                </div>

                <div id="demo-tab" class="tab-content active">
                    <h2>Live Demo - Single Coordinate</h2>
                    
                    <div class="demo-section">
                        <p>Example: BRCA1 gene start position (hg19 to hg38)</p>
                        
                        <label>Chromosome:</label>
                        <input type="text" id="chrom" value="chr17" placeholder="chr17">
                        
                        <label>Position:</label>
                        <input type="number" id="pos" value="41196312" placeholder="41196312">
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <label>From Build:</label>
                                <select id="from-build">
                                    <option value="hg19" selected>hg19 (GRCh37)</option>
                                    <option value="hg38">hg38 (GRCh38)</option>
                                </select>
                            </div>
                            <div>
                                <label>To Build:</label>
                                <select id="to-build">
                                    <option value="hg38" selected>hg38 (GRCh38)</option>
                                    <option value="hg19">hg19 (GRCh37)</option>
                                </select>
                            </div>
                        </div>
                        
                        <label>
                            <input type="checkbox" id="include-ml" checked> Include ML Confidence Prediction
                        </label>
                        
                        <button onclick="testLiftover()">Convert Coordinate</button>
                        <button class="btn-secondary" onclick="loadExample('BRCA1')">BRCA1</button>
                        <button class="btn-secondary" onclick="loadExample('TP53')">TP53</button>
                        <button class="btn-secondary" onclick="loadExample('EGFR')">EGFR</button>
                        
                        <div class="loading" id="loading">Processing...</div>
                        <div class="result-box" id="result"></div>
                    </div>
                </div>
                
                <div id="batch-tab" class="tab-content">
                    <h2>Batch File Upload</h2>
                    
                    <div class="demo-section">
                        <p>Upload CSV or TSV file with columns: chrom, pos</p>
                        
                        <input type="file" id="batch-file" accept=".csv,.tsv,.txt">
                        
                        <button onclick="uploadBatch()">Upload and Process</button>
                        
                        <div class="loading" id="batch-loading">Uploading...</div>
                        <div class="result-box" id="batch-result"></div>
                    </div>
                </div>
                
                <div id="docs-tab" class="tab-content">
                    <h2>Documentation</h2>
                    
                    <h3>Core Features</h3>
                    <ul style="line-height: 2; margin-left: 2rem;">
                        <li>Coordinate Liftover: UCSC chain file-based conversion with ML confidence</li>
                        <li>VCF Processing: Full variant file conversion with sample preservation</li>
                        <li>Validation: Tested against NCBI RefSeq gene coordinates</li>
                    </ul>
                    
                    <div style="margin-top: 2rem;">
                        <a href="/docs" style="display: inline-block; padding: 0.75rem 1.5rem; background: #001f3f; color: white; text-decoration: none; margin: 0.5rem;">API Documentation</a>
                        <a href="/validation-report" style="display: inline-block; padding: 0.75rem 1.5rem; background: #001f3f; color: white; text-decoration: none; margin: 0.5rem;">Validation Report</a>
                        <a href="/health" style="display: inline-block; padding: 0.75rem 1.5rem; background: #001f3f; color: white; text-decoration: none; margin: 0.5rem;">System Health</a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function switchTab(tab) {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                event.target.classList.add('active');
                document.getElementById(tab + '-tab').classList.add('active');
            }}

            function loadExample(gene) {{
                const examples = {{
                    'BRCA1': {{ chrom: 'chr17', pos: 41196312, from: 'hg19', to: 'hg38' }},
                    'TP53': {{ chrom: 'chr17', pos: 7571720, from: 'hg19', to: 'hg38' }},
                    'EGFR': {{ chrom: 'chr7', pos: 55086725, from: 'hg19', to: 'hg38' }}
                }};
                
                const ex = examples[gene];
                document.getElementById('chrom').value = ex.chrom;
                document.getElementById('pos').value = ex.pos;
                document.getElementById('from-build').value = ex.from;
                document.getElementById('to-build').value = ex.to;
            }}

            async function testLiftover() {{
                const chrom = document.getElementById('chrom').value;
                const pos = document.getElementById('pos').value;
                const fromBuild = document.getElementById('from-build').value;
                const toBuild = document.getElementById('to-build').value;
                const includeML = document.getElementById('include-ml').checked;
                
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                
                resultDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {{
                    const url = `/liftover/single?chrom=${{chrom}}&pos=${{pos}}&from_build=${{fromBuild}}&to_build=${{toBuild}}&include_ml=${{includeML}}`;
                    const response = await fetch(url, {{ method: 'POST' }});
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    
                    if (data.success) {{
                        let html = '<div class="success">Conversion Successful</div>';
                        html += `Original: ${{data.original.chrom}}:${{data.original.pos}} (${{fromBuild}})\\n`;
                        html += `Converted: ${{data.lifted_chrom}}:${{data.lifted_pos}} (${{toBuild}})\\n`;
                        html += `Chain Score: ${{(data.confidence * 100).toFixed(2)}}%\\n`;
                        
                        if (data.ml_analysis && data.ml_analysis.confidence_score !== undefined) {{
                            html += `\\nML Confidence: ${{(data.ml_analysis.confidence_score * 100).toFixed(2)}}%\\n`;
                            if (data.ml_analysis.interpretation && data.ml_analysis.interpretation.recommendation) {{
                                html += `Recommendation: ${{data.ml_analysis.interpretation.recommendation}}\\n`;
                            }}
                        }}
                        
                        html += `\\nFull Response:\\n${{JSON.stringify(data, null, 2)}}`;
                        resultDiv.innerHTML = html;
                    }} else {{
                        resultDiv.innerHTML = `<div class="error">Conversion Failed</div>\\n${{data.error || 'Unknown error'}}`;
                    }}
                }} catch (error) {{
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<div class="error">Request Error: ${{error.message}}</div>`;
                }}
            }}

            async function uploadBatch() {{
                const fileInput = document.getElementById('batch-file');
                const file = fileInput.files[0];
                
                if (!file) {{
                    alert('Please select a file');
                    return;
                }}
                
                const resultDiv = document.getElementById('batch-result');
                const loadingDiv = document.getElementById('batch-loading');
                
                resultDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                try {{
                    const text = await file.text();
                    const lines = text.trim().split('\\n');
                    
                    const separator = text.includes('\\t') ? '\\t' : ',';
                    const headers = lines[0].split(separator).map(h => h.trim().toLowerCase());
                    
                    const chromIdx = headers.findIndex(h => h === 'chrom' || h === 'chr' || h === 'chromosome');
                    const posIdx = headers.findIndex(h => h === 'pos' || h === 'position' || h === 'start');
                    
                    if (chromIdx === -1 || posIdx === -1) {{
                        throw new Error('File must have "chrom" and "pos" columns');
                    }}
                    
                    const coordinates = [];
                    for (let i = 1; i < lines.length; i++) {{
                        const parts = lines[i].split(separator);
                        if (parts.length > Math.max(chromIdx, posIdx)) {{
                            coordinates.push({{
                                chrom: parts[chromIdx].trim(),
                                pos: parseInt(parts[posIdx].trim())
                            }});
                        }}
                    }}
                    
                    const response = await fetch('/liftover/batch?from_build=hg19&to_build=hg38&include_ml=true', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(coordinates)
                    }});
                    
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    
                    let html = '<div class="success">Batch Job Created</div>';
                    html += `Job ID: ${{data.job_id}}\\n`;
                    html += `Coordinates: ${{data.total_coordinates}}\\n`;
                    html += `Status: ${{data.status}}\\n\\n`;
                    html += `Check status at: <a href="/job-status/${{data.job_id}}" target="_blank">/job-status/${{data.job_id}}</a>`;
                    
                    resultDiv.innerHTML = html;
                    
                }} catch (error) {{
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<div class="error">Error: ${{error.message}}</div>`;
                }}
            }}
        </script>
    </body>
    </html>
    """)


@app.get("/health")
def health_check():
    """System health check"""
    active_count = sum(1 for job in job_storage.values() if job.status in ["queued", "processing"])
    completed_count = sum(1 for job in job_storage.values() if job.status == "completed")
    failed_count = sum(1 for job in job_storage.values() if job.status == "failed")
    
    # Test ML model
    ml_trained = False
    if SERVICES.get('confidence_predictor'):
        try:
            test_features = np.zeros((1, 11))
            _ = SERVICES['confidence_predictor'].predict_confidence(test_features)
            ml_trained = True
        except:
            pass
    
    return {
        "status": "operational" if SERVICES_AVAILABLE else "limited",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(time.time() - startup_time),
        
        "services": {
            "liftover": SERVICES.get('liftover') is not None,
            "vcf_converter": SERVICES.get('vcf_converter') is not None,
            "feature_extraction": SERVICES.get('feature_extractor') is not None,
            "confidence_prediction": SERVICES.get('confidence_predictor') is not None,
            "ml_model_trained": ml_trained,
            "validation_engine": SERVICES.get('validation_engine') is not None,
        },
        
        "jobs": {
            "active": active_count,
            "completed": completed_count,
            "failed": failed_count,
            "total": len(job_storage)
        },
        
        "supported_assemblies": ["hg19/GRCh37", "hg38/GRCh38"],
        "active_jobs": active_count
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
        try:
            results = []
            
            for i, coord in enumerate(coordinates):
                result = SERVICES['liftover'].convert_coordinate(
                    coord.get("chrom", ""),
                    coord.get("pos", 0),
                    from_build,
                    to_build
                )
                
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
            job.end_time = datetime.now()
            
            successful = sum(1 for r in results if r.get("success"))
            job.metadata = {
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
        response["metadata"] = job.metadata
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
                "metadata": job.metadata
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)