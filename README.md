# Genomic Annotation Version Controller (Prototype)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16966073.svg)](https://doi.org/10.5281/zenodo.16966073)

## Overview
This project is a **proof-of-concept** tool for managing genomic annotation differences between Ensembl releases and across GRCh37 and GRCh38 genome builds. It demonstrates software architecture patterns for genomic data processing while exploring automated conflict resolution approaches.

## Why I Built This
I developed this project independently as a high school student to learn about computational genomics and software engineering. My goal was to understand the technical challenges researchers face when working with evolving genome annotations and to build a functional system that demonstrates potential solutions.

I am sharing this with the genomics community to request feedback, technical critique, and guidance on both the biological accuracy and software engineering approaches.

## Technical Architecture

### Core Engineering Features
- **RESTful API Design**: OpenAPI/Swagger documentation with proper HTTP status codes
- **Asynchronous Job Queue**: Background task processing with real-time status monitoring  
- **Multiple Export Formats**: CSV, BED, VCF, and JSON with proper MIME types
- **Comprehensive Error Handling**: Graceful degradation and informative error messages
- **CORS Support**: Configurable cross-origin resource sharing for web applications
- **Health Monitoring**: System status endpoints with uptime tracking

### Current Implementation Status
-  **API Framework**: Fully functional FastAPI application with proper routing
-  **Job Management**: Background task processing with status tracking
-  **Data Export**: Multi-format export with appropriate headers
-  **Live Ensembl Integration**: Real queries to Ensembl REST API for gene metadata
- **Coordinate Liftover**: Prototype implementation using simplified coordinate transformation
-  **Conflict Resolution**: K-means clustering for annotation grouping (real implementation)
-  **Validation Framework**: Testing against known NCBI coordinate pairs

## Features
- Real-time gene annotation lookup via Ensembl REST API
- Batch coordinate processing with progress monitoring
- Automated annotation quality assessment using configurable metrics
- K-means clustering for grouping similar annotation sources
- Transparent job tracking with detailed status reporting
- Multi-format data export (CSV, BED, VCF, JSON)

## Current Limitations & Known Issues
- **Coordinate Liftover**: Uses simplified offset calculation rather than UCSC LiftOver chain files
- **Limited Validation**: Tested on small dataset of major genes; requires comprehensive benchmarking
- **Prototype AI Components**: Clustering implementation is functional but simplified
- **No Authentication**: Currently designed for research/demo use without user management
- **Memory Storage**: Jobs stored in-memory; production would require persistent storage

## Validation Results
Current testing against known NCBI RefSeq coordinates shows:
- Prototype coordinate transformation provides directionally correct results
- Quality assessment metrics identify potential annotation issues
- Clustering successfully groups similar annotations from multiple sources

**Note**: These are preliminary results from a prototype implementation. Production use would require extensive validation against established genomic databases and liftover tools.

## Technical Dependencies
- **FastAPI**: Web framework with automatic API documentation
- **Pandas**: Data manipulation and export functionality  
- **NumPy**: Numerical computations for clustering algorithms
- **Requests**: HTTP client for external API integration
- **Gradio**: Optional web interface for demonstrations

## Try the Demo
- **Web Interface**: [genomic-annotation-version-controller.onrender.com](https://genomic-annotation-version-controller.onrender.com)
- **API Documentation**: Available at `/docs` endpoint
- **Health Check**: Available at `/health` endpoint

## Code Structure
```
├── main.py                 # FastAPI application and routing
├── ai_conflict_resolver.py # Clustering and conflict resolution logic
├── demo_ai_conflicts.py    # Demonstration scripts
└── README.md              # This file
```

## Installation & Usage
```bash
# Install dependencies
pip install fastapi pandas numpy requests uvicorn

# Run the application
uvicorn main:app --reload

# Access API documentation
curl http://localhost:8000/docs
```

## Seeking Technical Feedback
I would appreciate input on:
- **Software Architecture**: Is the API design following genomics community best practices?
- **Biological Accuracy**: Are my assumptions about annotation conflicts reasonable?
- **Algorithm Implementation**: How can the clustering approach be improved?
- **Validation Strategy**: What benchmarking datasets would be most valuable?
- **Production Considerations**: What would be needed for real research use?

## Future Development Priorities
1. Integration with UCSC LiftOver for accurate coordinate conversion
2. Comprehensive validation against larger genomic datasets  
3. Enhanced clustering algorithms with genomic-specific distance metrics
4. Persistent storage and user session management
5. Integration with additional annotation databases (RefSeq, GENCODE)

## How to Cite
If you find this work useful or provide feedback, please cite:
Asher, Arnav. (2025). Genomic Annotation Version Controller (Prototype). Zenodo. https://doi.org/10.5281/zenodo.16966073

## Contributing
This is a learning project where I welcome:
- Technical code reviews and suggestions
- Biological accuracy corrections
- Algorithm improvement recommendations
- Testing with real genomic datasets
- Guidance on genomics community standards

Please open issues or contact me directly with feedback.