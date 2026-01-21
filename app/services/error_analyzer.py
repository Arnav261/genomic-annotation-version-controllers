"""
Simple error analysis utilities for liftover results.
Categorizes errors, computes distribution summaries, and prepares markdown reports.
"""
from typing import List, Dict
import statistics
import math


class ErrorAnalyzer:
    def __init__(self, records: List[Dict]):
        """
        records: list of validation records each with keys:
            - success (bool)
            - expected_pos (int)
            - actual_pos (int or None)
            - liftover_error (optional)
        """
        self.records = records

    def summarize(self) -> Dict:
        errors = []
        failed = 0
        ambiguous = 0
        for r in self.records:
            if not r.get("success"):
                failed += 1
                continue
            if r.get("ambiguous"):
                ambiguous += 1
            exp = r.get("expected_pos")
            act = r.get("actual_pos")
            if exp is None or act is None:
                continue
            errors.append(abs(exp - act))
        summary = {
            "total": len(self.records),
            "failed": failed,
            "ambiguous": ambiguous,
            "analyzed": len(errors),
        }
        if errors:
            summary.update({
                "mean_error_bp": statistics.mean(errors),
                "median_error_bp": statistics.median(errors),
                "p95_error_bp": sorted(errors)[max(0, int(len(errors) * 0.95) - 1)],
                "max_error_bp": max(errors),
            })
        else:
            summary.update({"mean_error_bp": None, "median_error_bp": None, "p95_error_bp": None, "max_error_bp": None})
        return summary

    def to_markdown(self) -> str:
        s = self.summarize()
        md = [
            "# Error Analysis Report",
            f"- Total records: {s['total']}",
            f"- Failed mappings: {s['failed']}",
            f"- Ambiguous mappings: {s['ambiguous']}",
            f"- Records analyzed for error: {s['analyzed']}",
        ]
        if s['analyzed']:
            md += [
                f"- Mean error (bp): {s['mean_error_bp']:.2f}",
                f"- Median error (bp): {s['median_error_bp']:.2f}",
                f"- 95th percentile error (bp): {s['p95_error_bp']}",
                f"- Max error (bp): {s['max_error_bp']}",
            ]
        return "\n".join(md)