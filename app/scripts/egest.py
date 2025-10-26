#!/usr/bin/env python3
"""
Egestion script to export vectors and metadata from FAISS meta DB.
Usage:
  python scripts/egest.py --version v1 --out out.csv --embeddings out.npy
"""
import argparse, os, json, csv
from pathlib import Path as Pathlib
from app.vector_store import FaissVectorStore
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True, help="CSV file to write metadata")
    parser.add_argument("--embeddings", help="Optional .npy file path to save embeddings")
    args = parser.parse_args()
    store = FaissVectorStore()
    cur = store._conn.cursor()
    rows = cur.execute("SELECT id, metadata FROM vectors WHERE version=?", (args.version,)).fetchall()
    if not rows:
        print("No data for version", args.version)
        return
    ids = [r[0] for r in rows]
    metas = [json.loads(r[1]) if r[1] else {} for r in rows]
    out_rows = []
    embs = []
    for uid in ids:
        try:
            idx = store._id_map.index(uid)
            v = store._index.reconstruct(idx)
            embs.append(v)
            meta = store.get_by_id(uid)
            row = {"id": uid, "version": meta["version"], "metadata": json.dumps(meta["metadata"])}
            out_rows.append(row)
        except Exception as e:
            print("skip", uid, e)
    # write CSV
    with open(args.out, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id","version","metadata"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    if args.embeddings:
        np.save(args.embeddings, np.vstack(embs))
    print("Exported", len(out_rows), "rows")

if __name__ == "__main__":
    main()