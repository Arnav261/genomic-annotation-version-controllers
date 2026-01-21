"""
Benchmarking utilities: run comparisons between liftover tools and produce JSON + Markdown reports.
"""
from typing import List, Dict
from app.validation.statistical_tests import paired_t_test, bayesian_mean_estimate
from app.services.error_analyzer import ErrorAnalyzer
import statistics


class BenchmarkingSuite:
    def __init__(self):
        pass

    def compare_tool_results(self, results_a: List[Dict], results_b: List[Dict]) -> Dict:
        """
        results_a and results_b are lists of dicts with keys:
            - success (bool), error_bp (int)
        We compute paired t-test on error_bp where both succeeded.
        """
        paired_errors_a = []
        paired_errors_b = []
        records = []
        for a, b in zip(results_a, results_b):
            if a.get("success") and b.get("success"):
                paired_errors_a.append(a.get("error_bp", 0))
                paired_errors_b.append(b.get("error_bp", 0))
                records.append({"a": a, "b": b})
        stats = {}
        if paired_errors_a:
            stats["paired_t_test"] = paired_t_test(paired_errors_a, paired_errors_b)
            stats["bayesian_a"] = bayesian_mean_estimate(paired_errors_a)
            stats["bayesian_b"] = bayesian_mean_estimate(paired_errors_b)
            stats["summary"] = {
                "a_mean": statistics.mean(paired_errors_a),
                "b_mean": statistics.mean(paired_errors_b),
            }
        else:
            stats["note"] = "No paired successful records for statistical comparison."
        # Error distribution analysis
        ea = ErrorAnalyzer([r["a"] for r in records])
        eb = ErrorAnalyzer([r["b"] for r in records])
        stats["error_summary_a"] = ea.summarize()
        stats["error_summary_b"] = eb.summarize()
        return stats

    def to_markdown(self, comparison_result: Dict) -> str:
        md = ["# Benchmarking Report", ""]
        if "paired_t_test" in comparison_result:
            pt = comparison_result["paired_t_test"]
            md += [
                "## Paired t-test",
                f"- t-statistic: {pt['t_stat']:.4f}",
                f"- p-value: {pt['p_value']:.6f}",
                f"- mean difference (A - B): {pt['mean_diff']:.4f}",
                ""
            ]
        if "summary" in comparison_result:
            md += [
                "## Summary",
                f"- Mean error A: {comparison_result['summary']['a_mean']:.2f}",
                f"- Mean error B: {comparison_result['summary']['b_mean']:.2f}",
                ""
            ]
        md += ["## Error Summaries", "### Tool A", f"{comparison_result.get('error_summary_a')}","", "### Tool B", f"{comparison_result.get('error_summary_b')}"]
        return "\n".join(md)