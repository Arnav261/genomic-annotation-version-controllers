#!/usr/bin/env python3
"""
Flexible ingestion script:
- Accepts CSV/TSV/JSON/JSONL
- Auto-detects 'description', 'value', 'seq', 'source', 'id' columns
- Batches embeddings and pushes to FAISS vector store via the semantic_context.ingest_annotation_embeddings
Usage:
  python scripts/ingest.py --input data/annotations.csv --version v1 --model sbert --seq_type text --batch 64
"""
import argparse, os, json, csv, sys
from typing import List, Dict, Any
from pathlib import Path as Pathlib

ROOT = Pathlib(__file__).parent.parent
import app.semantic_context as sc

def read_input(path: Pathlib) -> List[Dict[str, Any]]:
    path = Pathlib(path)
    content = []
    if path.suffix.lower() in (".json", ".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line) if path.suffix.lower() == ".jsonl" else json.load(f)
                    if isinstance(obj, list):
                        content.extend(obj)
                        break
                    else:
                        content.append(obj)
                except json.JSONDecodeError:
                    # maybe JSONL line
                    try:
                        obj = json.loads(line)
                        content.append(obj)
                    except Exception:
                        continue
    elif path.suffix.lower() in (".csv", ".tsv", ".txt"):
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=sep)
            for row in reader:
                content.append({k: (v if v != "" else None) for k,v in row.items()})
    else:
        raise ValueError("Unsupported input format")
    return content

def normalize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        rec = {}
        rec["id"] = r.get("id") or r.get("uid") or r.get("ID")
        rec["value"] = r.get("value") or r.get("annotation") or r.get("label")
        rec["description"] = r.get("description") or r.get("desc") or r.get("note")
        rec["seq"] = r.get("seq") or r.get("sequence")
        rec["source"] = r.get("source") or r.get("db") or r.get("source_name")
        rec["metadata"] = r.get("metadata") or {}
        if r.get("evidence_score") is not None:
            try:
                rec["metadata"]["evidence_score"] = float(r.get("evidence_score"))
            except Exception:
                rec["metadata"]["evidence_score"] = r.get("evidence_score")
        out.append(rec)
    return out

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default="sbert")
    parser.add_argument("--seq_type", default="text", choices=["text","protein","dna"])
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    records = read_input(Pathlib(args.input))
    recs = normalize_records(records)
    total = len(recs)
    print(f"Loaded {total} records. Ingesting into version={args.version} seq_type={args.seq_type} model={args.model}")
    ingested = 0
    for chunk in chunkify(recs, args.batch):
        try:
            ids = sc.ingest_annotation_embeddings(chunk, args.version, model_key=args.model, seq_type=args.seq_type)
            ingested += len(ids)
            print(f"Ingested chunk: {len(ids)} total_ingested={ingested}/{total}")
        except Exception as e:
            print(f"Failed to ingest chunk: {e}", file=sys.stderr)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()