"""
Bayesian utilities used by ML pipeline: lightweight credible intervals for error estimates.
"""
from typing import List, Dict
from app.validation.statistical_tests import bayesian_mean_estimate


def estimate_error_bayes(errors: List[float]) -> Dict:
    return bayesian_mean_estimate(errors)