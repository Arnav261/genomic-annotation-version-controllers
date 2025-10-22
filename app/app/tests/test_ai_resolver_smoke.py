import numpy as np
from app.ai_resolver import AIResolver
import pytest

def test_ai_resolver_stub(monkeypatch):
    annotations = [
        {"source":"db1","value":"GENE1","description":"Gene one annotated in db1", "evidence_score":1.0},
        {"source":"db2","value":"GENE1","description":"Gene one annotation", "evidence_score":1.0},
        {"source":"db3","value":"GENE2","description":"Different gene annotation", "evidence_score":0.5},
    ]
    resolver = AIResolver()
    # stub embed_texts to return small 3x4 embeddings where first two are similar
    def fake_embed_texts(texts, model_key, prefer_sbert=False):
        e = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)
        # normalize rows
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / norms
    monkeypatch.setattr(resolver, "embed_texts", fake_embed_texts)
    out = resolver.resolve(annotations, model_name="sbert", sim_threshold=0.8, prefer_sbert=False)
    assert "clusters" in out
    assert out["num_annotations"] == 3
    # there should be at least 2 clusters (first two grouped)
    assert out["num_clusters"] >= 2