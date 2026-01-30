# Genomic Coordinate Liftover with ML Confidence Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16966073.svg)](https://doi.org/10.5281/zenodo.16966073)  
> **Honest Implementation**: This tool provides accurate coordinate liftover with ML-based confidence prediction. It does NOT claim to be a general-purpose AI genomics platform.

## What This Tool Actually Does

### Core Features ✓

1. **Accurate Coordinate Liftover**
   - Uses UCSC LiftOver chain files (pyliftover)
   - Supports hg19 (GRCh37) ↔ hg38 (GRCh38)
   - Per-variant tracking and validation against authoritative coordinates

2. **ML Confidence Prediction**
   - Gradient boosting classifier with calibrated probability outputs
   - Interpretable feature contributions (SHAP-compatible export)
   - Expanded feature set including chain quality, repeats, SVs, GC content, and historical region success

3. **VCF File Conversion & Streaming**
   - Complete VCF 4.x parsing and validation
   - Preserves sample/genotype information
   - Streaming-friendly batch processing for large VCFs
   - Per-variant conversion status and quality metrics

4. **Scalable Batch Processing (Only Partially Functional - Repeated Separate Liftover Recommended)**
   - Parallelized batch liftover worker (configurable concurrency)
   - CLI support for batched jobs and pipeline integration
   - Optional chunked processing to limit memory footprint

5. **Deployment & Reproducibility**
   - Docker image for demo and reproducible environments
   - GitHub Actions CI for tests and linting
   - Pre-built demo on Render (where available)

6. **Expanded Validation**
   - Broader RefSeq-derived training/validation set (expanded towards ~20K genes)
   - Exportable validation reports with per-chromosome and calibration analysis

7. **Optional Integrations (Pluggable)**
   - Ensembl API connector (experimental) for cross-references
   - RepeatMasker, DGV/gnomAD-SV integration to flag problematic regions

---

## What This Tool Does NOT Do ✗

Be honest:
- This is NOT clinical-grade variant interpretation (not FDA/CLIA).
- It is not a general biomedical language model — semantic reconciliation is lightweight.
- Multi-species liftover is experimental/partial; human assemblies are the primary supported target.
- Real-time production-scale RL or deep learning components are not part of the stable core.

---

## Quick Start

### Install (recommended: virtualenv) or use Docker

Using Python virtual environment:

```bash
git clone https://github.com/Arnav261/genomic-annotation-version-controllers.git
cd genomic-annotation-version-controllers

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download UCSC chain files (one-time):

```bash
mkdir -p app/data/chains
cd app/data/chains
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
cd ../../..
```

Run the API server:

```bash
uvicorn app.main:app --reload
```

Using Docker (recommended for demo or reproducible runs):

```bash
# Build
docker build -t gavo-liftover:latest .

# Run (exposes API on 8000)
docker run --rm -p 8000:8000 \
  -v "$(pwd)/app/data/chains:/app/data/chains:ro" \
  gavo-liftover:latest

# Or run the published image (if available)
docker run --rm -p 8000:8000 arnav261/gavo-liftover:latest
```

### CLI example (batch liftover)

```bash
# liftover-cli is available in the repo to run local batch jobs
liftover-cli --input variants.vcf --output variants.lifted.vcf \
  --from hg19 --to hg38 --workers 4 --include-ml
```

### Basic Python usage (single coordinate)

```python
import requests

response = requests.post(
    "http://localhost:8000/liftover/single",
    params={
        "chrom": "chr17",
        "pos": 41196312,
        "from_build": "hg19",
        "to_build": "hg38",
        "include_ml": True
    }
)

result = response.json()
print(f"Lifted position: {result.get('lifted_pos')}")
print(f"ML confidence: {result.get('ml_analysis',{}).get('confidence_score')}")
print(f"Recommendation: {result.get('ml_analysis',{}).get('interpretation',{}).get('recommendation')}")
```

---

## Technical Details

### ML Confidence Model

- Algorithm: Gradient boosting classifier (scikit-learn) with probability calibration. Optional LightGBM backend supported for faster training/inference.
- Features (examples):
  - Chain alignment score and agreement
  - Chain gap and local chain context
  - Local GC content (±1kb)
  - RepeatMasker overlap density and low-complexity flag
  - Structural variant overlap (DGV/gnomAD-SV)
  - Segmental duplication overlap
  - Distance to assembly gaps
  - Historical region liftover success
- Explainability: SHAP export enables per-variant explanation vectors that can be included in reports.

Training and validation:
- Training dataset expanded using RefSeq-derived coordinates; ongoing efforts to grow the validated set to ~20K genes for robust calibration.
- Internal cross-validation shows improved calibration and stability compared to the initial prototype. See exported validation reports for detailed metrics.

### Validation & Reports

The `/validation-report` endpoint and exported artifacts include:
- Per-chromosome accuracy
- Confidence score calibration plots and metrics
- Error distribution statistics and edge-case summaries
- Comparison methodology and reproducible scripts used for benchmarking

---

## API Endpoints (high level)

- POST /liftover/single — single coordinate liftover (with optional ML output)
- POST /liftover/batch — JSON array of coordinates; supports streaming/async mode
- POST /liftover/stream — upload VCF for streaming conversion (recommended for large files)
- GET /validation-report — returns the latest validation artifacts
- GET /health — service and ML model health with honest status report

Example cURL (single):

```bash
curl -X POST "http://localhost:8000/liftover/single?chrom=chr17&pos=41196312&from_build=hg19&to_build=hg38&include_ml=true"
```

Batch (file upload / streaming is recommended for large sets):

```bash
curl -X POST "http://localhost:8000/liftover/batch" \
  -H "Content-Type: application/json" \
  -d '[{"chrom": "chr17", "pos": 41196312}, {"chrom": "chr7", "pos": 55086725}]'
```

---

## Development Roadmap

### Phase 1: Current State ✓
- [x] Core liftover using UCSC chain files
- [x] ML confidence prediction (calibrated gradient boosting)
- [x] VCF conversion and streaming support
- [x] FastAPI with OpenAPI docs and health checks
- [x] Docker image and basic CI

### Phase 2: Enhanced ML (In Progress - Near Completion)
- [x] Expanded RefSeq-derived training and validation dataset
- [x] SHAP-compatible explainability export
- [x] Parallel batch processing and CLI
- [ ] Further calibration with additional validated variants (Partially achieved with semantic reconciliation)

### Phase 3: Advanced Features (Planned)
- [ ] Biomedical NLP (PubMedBERT) for semantic reconciliation (experimental)
- [ ] Multi-species liftover support (beta)
- [ ] RepeatMasker and DGV/gnomAD-SV full integration 
- [ ] Real-time Ensembl API backlinking and cross-references
- [ ] Benchmarking against CrossMap and UCSC liftOver at scale

### Phase 4: Publication (Goal)
- [ ] Comprehensive benchmark and methods paper
- [ ] Performance profiling and memory/speed optimizations
- [ ] Community-driven validations and user-facing tutorials

---

## Data Sources

### Required
- **UCSC Chain Files**: coordinate mappings
  - Source: http://hgdownload.soe.ucsc.edu/goldenPath/
  - License: open access for research

- **NCBI RefSeq**: gene coordinates for training/validation
  - Source: ftp://ftp.ncbi.nlm.nih.gov/refseq/
  - License: public domain

### Optional (improves accuracy)
- **RepeatMasker** annotations
- **DGV / gnomAD-SV** for structural variant context
- **Ensembl** cross-references (experimental)
- **ClinVar** variant coordinates for clinical validation

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{asher2025genomic,
  author = {Asher, Arnav},
  title = {Genomic Coordinate Liftover with ML Confidence Prediction},
  year = {2025},
  doi = {10.5281/zenodo.16966073},
  url = {https://github.com/Arnav261/genomic-annotation-version-controllers}
}
```

---

## Contributing

This is a learning & research project — contributions and feedback are welcome.

### Helpful contributions
- Testing on additional gene sets and real VCFs
- Integration and benchmarking with other liftover tools (CrossMap, UCSC liftOver)
- Performance optimization and memory profiling
- Improvements to ML model, calibration, and feature engineering
- Better docs and example pipelines

How to contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests where applicable
4. Submit a pull request with a clear description

---

## Known Issues & Limitations

### Current Limitations
1. Assembly support: Primary support remains hg19 ↔ hg38 (human). Multi-species is experimental.
2. ML model: While training data has been expanded, more high-quality validated variants are needed for production-grade calibration.
3. Semantic features: Still lightweight; PubMedBERT integration is planned but not production-ready.
4. Resource usage: Large VCF streaming requires adequate memory and I/O; tune worker count accordingly.

### Edge Cases
- Centromeres, telomeres, and PAR regions remain problematic.
- Large structural variants, complex rearrangements, or regions adjacent to assembly gaps may fail liftover or be assigned low confidence.
- X/Y pseudoautosomal edge cases require cautious interpretation.

---

## Support

- Issues: [GitHub Issues](https://github.com/Arnav261/genomic-annotation-version-controllers/issues)
- Email: arnavasher007@gmail.com
- Demo: [genomic-annotation-version-controller.onrender.com](https://genomic-annotation-version-controller.onrender.com) (subject to availability)

---

## Acknowledgments

- UCSC Genome Browser: chain files and liftover resources
- NCBI: RefSeq coordinates
- pyliftover, scikit-learn, FastAPI
- Community contributors and early testers

---

## Honest Self-Assessment

What works well:
- Accurate and validated liftover for many genic regions
- ML confidence gives interpretable uncertainty that is useful for filtering and triage
- Docker and CLI make reproducible demos and batch processing straightforward

What needs improvement:
- Larger, high-quality training/validation datasets for production-grade calibration
- Full integration with large annotation sources (RepeatMasker, gnomAD-SV)
- Multi-species and advanced NLP features remain experimental

Rating: 7/10 for research use; with additional validation, documentation, and benchmarks this can rise for production research pipelines.

---

**Last Updated:** 2026-01-30  
**Version:** 5.0.0  
**Status:** Active Development
