import pytest
from app.conflict_resolution import resolve_annotation_conflicts

def test_simple_majority():
    anns = [
        {"source":"db1","value":"GENE1","evidence_score":2.0},
        {"source":"db2","value":"GENE1","evidence_score":1.0},
        {"source":"db3","value":"GENE2","evidence_score":0.5},
    ]
    out = resolve_annotation_conflicts(anns)
    assert out["consensus_value"] == "GENE1"
    assert out["support_count"] == 2
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert "entropy" in out

def test_no_annotations():
    out = resolve_annotation_conflicts([])
    assert out["consensus_value"] is None
    assert out["support_count"] == 0