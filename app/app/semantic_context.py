"""
Semantic contextualizer using embeddings + vector store.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import uuid

from .vector_store import FaissVectorStore
from .embeddings import SequenceEmbeddingBackend

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

logger = logging.getLogger(__name__)
EMBED_BACKEND = SequenceEmbeddingBackend()
VECTOR_STORE = FaissVectorStore(dim=768)

def ingest_annotation_embeddings(annotations: List[Dict[str, Any]], version: str, model_key: str = "sbert", seq_type: str = "text") -> List[str]:
    texts = []
    ids = []
    for ann in annotations:
        uid = ann.get("id") or ann.get("uid") or str(uuid.uuid4())
        ids.append(uid)
        if seq_type == "protein" and ann.get("seq"):
            texts.append(ann["seq"])
        elif seq_type == "dna" and ann.get("seq"):
            texts.append(ann["seq"])
        else:
            texts.append(str(ann.get("description") or ann.get("value") or ""))
    if seq_type == "protein":
        emb = EMBED_BACKEND.embed_proteins(texts, model_key=model_key)
    elif seq_type == "dna":
        emb = EMBED_BACKEND.embed_dna(texts, model_key=model_key)
    else:
        emb = EMBED_BACKEND.embed_texts_sbert(texts, model_key=model_key)
    for i, uid in enumerate(ids):
        metadata = annotations[i].get("metadata", {})
        metadata.update({"source": annotations[i].get("source"), "value": annotations[i].get("value")})
        VECTOR_STORE.add_vector(uid, emb[i], version, metadata)
    VECTOR_STORE.persist()
    return ids

def cluster_annotations(version: str, method: str = "hdbscan", min_cluster_size: int = 2) -> Dict[str, Any]:
    cur = VECTOR_STORE._conn.cursor()
    rows = cur.execute("SELECT id FROM vectors WHERE version=?", (version,)).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        return {"clusters": {}, "n": 0}
    idxs = [VECTOR_STORE._id_map.index(i) for i in ids if i in VECTOR_STORE._id_map]
    emb_list = []
    for idx in idxs:
        try:
            v = VECTOR_STORE._index.reconstruct(int(idx))
            emb_list.append(v)
        except Exception:
            pass
    if not emb_list:
        return {"clusters": {}, "n": 0}
    X = np.vstack(emb_list)
    if method == "hdbscan" and HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(X)
    else:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.15, min_samples=min_cluster_size, metric='cosine')
        labels = db.fit_predict(X)
    clusters = {}
    for lab, uid in zip(labels, ids):
        clusters.setdefault(int(lab), []).append(uid)
    return {"clusters": clusters, "n": len(ids)}

def flag_outliers(version: str, distance_threshold: float = 0.2) -> List[Dict[str, Any]]:
    cur = VECTOR_STORE._conn.cursor()
    rows = cur.execute("SELECT id FROM vectors WHERE version=?", (version,)).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        return []
    outliers = []
    for uid in ids:
        try:
            idx = VECTOR_STORE._id_map.index(uid)
            v = VECTOR_STORE._index.reconstruct(idx).astype('float32')
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            v = v / norm
            res = VECTOR_STORE.search(v, top_k=6, version=version)
            scores = [r["score"] for r in res if r["id"] != uid]
            if not scores:
                continue
            avg_sim = sum(scores) / len(scores)
            if avg_sim < (1.0 - distance_threshold):
                meta = VECTOR_STORE.get_by_id(uid)
                outliers.append({"id": uid, "avg_similarity": avg_sim, "metadata": meta})
        except Exception:
            continue
    return outliers

def suggest_merges(version: str, top_k: int = 5, sim_threshold: float = 0.85) -> List[Dict[str, Any]]:
    cur = VECTOR_STORE._conn.cursor()
    rows = cur.execute("SELECT id FROM vectors WHERE version=?", (version,)).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        return []
    suggestions = []
    for uid in ids:
        try:
            idx = VECTOR_STORE._id_map.index(uid)
            v = VECTOR_STORE._index.reconstruct(idx).astype('float32')
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            v = v / norm
            res = VECTOR_STORE.search(v, top_k=top_k, version=version)
            close = [r for r in res if r["id"] != uid and r["score"] >= sim_threshold]
            if close:
                suggestions.append({"id": uid, "candidates": close})
        except Exception:
            continue
    return suggestions