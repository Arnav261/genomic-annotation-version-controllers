from typing import List, Dict, Any
import math
import collections

def compute_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counts.values():
        p = v/total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def resolve_annotation_conflicts(annotations: List[Dict[str, Any]], source_priority: List[str]=None) -> Dict[str, Any]:
    """
    Deterministic weighted consensus resolver for categorical annotation conflicts.

    annotations: list of {source: str, value: str, evidence_score: float (optional)}
    Returns:
      - consensus_value
      - support_count
      - top_sources
      - entropy
      - confidence_score
      - support_counts (per-value)
    Strategy:
     - Weighted majority by evidence_score
     - Tie-breaker by source_priority if provided
     - Entropy and confidence metrics for interpretability
    """
    weighted = collections.defaultdict(float)
    per_value_sources = collections.defaultdict(list)
    total_weight = 0.0
    for ann in annotations:
        val = ann.get("value")
        if val is None:
            continue
        weight = float(ann.get("evidence_score", 1.0))
        src = ann.get("source", "unknown")
        weighted[val] += weight
        per_value_sources[val].append(src)
        total_weight += weight

    if not weighted:
        return {"consensus_value": None, "support_count": 0, "entropy": 0.0, "confidence_score": 0.0, "support_counts": {}}

    sorted_vals = sorted(weighted.items(), key=lambda x: (-x[1], x[0]))
    top_val, top_weight = sorted_vals[0]

    # tie-breaker by source priority when weights are equal or very close
    if len(sorted_vals) > 1 and abs(sorted_vals[0][1] - sorted_vals[1][1]) < 1e-9 and source_priority:
        for s in source_priority:
            for v, _ in sorted_vals[:3]:
                if s in per_value_sources.get(v, []):
                    top_val = v
                    break

    counts = {k: int(len(per_value_sources[k])) for k in per_value_sources}
    entropy = compute_entropy(counts)
    raw_confidence = (top_weight / total_weight) if total_weight > 0 else 0.0
    confidence_score = min(max(float(raw_confidence), 0.0), 1.0)

    return {
        "consensus_value": top_val,
        "support_count": counts.get(top_val, 0),
        "top_sources": per_value_sources.get(top_val, [])[:10],
        "entropy": entropy,
        "confidence_score": confidence_score,
        "support_counts": counts
    }