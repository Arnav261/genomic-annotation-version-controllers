```markdown
# Genomic Annotation Version Controllers (FastAPI + AI + PLM + Vector DB + RL)

This repository provides:
- Coordinate liftover (pyliftover primary, UCSC liftOver as fallback)
- Deterministic and AI-powered semantic conflict resolution (SBERT / HF-based)
- Embedding-based genomic contextualizer (text, protein, nucleotide PLMs)
- Vector storage (default FAISS) for semantic queries, clustering, outlier detection, merge suggestions
- Reinforcement learning scaffold for learning conflict-resolution policies from curator feedback
- Simple static frontend demo and ingestion/egestion scripts

Highlights & Safety Notes (short)
- Liftover:
  - Uses UCSC chain files (must download separately). Chain files are authoritative; do not invent mappings.
  - pyliftover is used for convenience. If pyliftover can't map, the UCSC `liftOver` binary is used (if present).
  - Safety: Always verify critical mappings before downstream interpretation. Ambiguous mappings require manual curation.

- Semantic AI (text + SBERT):
  - Uses sentence-transformers (SBERT) for high-quality sentence embeddings where available.
  - For biomedical text, consider using domain models (BioBERT, PubMedBERT, SciBERT, or biomedical SBERTs).
  - Safety: Semantic similarity does not equal biological equivalence. Use confidence metrics and manual review.

- Sequence PLMs (ProtBERT / ProtTrans / ESM / DNABERT):
  - Provide biologically-informed embeddings for protein/DNA sequences.
  - ESM models (Meta AI) and ProtTrans are large — use GPU for production; support optional ESM if installed.
  - Safety: These embeddings suggest similarity but do not replace experimental validation. Treat merge suggestions as proposals.

- Vector DB (FAISS / Milvus / Weaviate):
  - FAISS is the default for local experiments (fast, embeddable).
  - For multi-host production, replace with Milvus or Weaviate (connector provided hooks).
  - Safety: Ensure vector-store persistence and backups; index rebuilds may be required for deletes.

- RL (Stable-Baselines3 / Ray RLlib):
  - Scaffold supports training offline RL agents from curator feedback.
  - Training must be done off-request (separate worker); DO NOT train models in web request threads.
  - Safety: Reward design is critical — noisy rewards lead to poor policies. Always test in simulation and monitor.

Quick start (local)
1. Download chain files (do not commit).
   Example: hg19 -> hg38
   wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz -P app/data/chains/

2. Create venv and install:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r app/requirements.txt

3. Run server:
   ./run.sh
   or
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Open demo:
   http://localhost:8000

Ingestion & Egestion scripts
- scripts/ingest.py: flexible ingest script (CSV/TSV/JSON/JSONL); auto-detects description/seq columns; batches embedding and indexing into FAISS.
- scripts/egest.py: export vectors and metadata to CSV/JSON; optionally export embeddings as .npy.

Design notes
- Modular: embedding backends, vector store abstraction, semantic layer, and RL scaffold are decoupled.
- Production: mount persistent volumes for MODEL_CACHE_DIR and LIFTOVER_CHAIN_DIR; use GPU for heavy embedding workloads; offload long training jobs to separate compute.

Interpretation checklist for biologists
- Always check confidence metrics:
  - support_count, confidence_score (evidence-weight), cluster_semantic_confidence, global_entropy.
- Flags:
  - ambiguous liftover mappings
  - cluster low confidence or high entropy
  - embeddings outliers flagged by the system
- Manual curation: recommended when confidence < 0.8 or liftover ambiguous.

Contributing & Next steps
- Add unit tests under tests/ for liftover, ai_resolver, embeddings, and semantic_context.
- Add a background worker (Celery/RQ) to offload large batch ingestion and RL training.
- To swap FAISS -> Milvus/Weaviate implement the VectorStore interface in app/vector_store.py.
```