"""
Statistical tests used by the benchmarking suite.
Provides paired t-test and a lightweight Bayesian mean estimator using PyTorch.
"""
from typing import List, Tuple, Dict
import math
import numpy as np
from scipy import stats

# PyTorch Bayesian estimator import delayed to avoid heavy import if not used
import torch


def paired_t_test(a: List[float], b: List[float]) -> Dict:
    """
    Perform paired t-test between arrays a and b.
    Returns t-statistic, p-value, mean_diff.
    """
    if len(a) != len(b):
        raise ValueError("Inputs must have same length")
    t_stat, p_val = stats.ttest_rel(a, b)
    mean_diff = float(np.mean(np.array(a) - np.array(b)))
    return {"t_stat": float(t_stat), "p_value": float(p_val), "mean_diff": mean_diff}


def bayesian_mean_estimate(values: List[float], prior_mu: float = 0.0, prior_sigma: float = 10.0, n_samples: int = 2000) -> Dict:
    """
    Simple Bayesian estimation of the mean assuming known Gaussian noise with unknown mean.
    Uses PyTorch to draw posterior samples with a conjugate normal-inverse-gamma style approximate sampler
    (implemented via a simple MCMC-free Normal posterior because of conjugacy when sigma known).
    For simplicity, we estimate the posterior of mean given observed data with assumed sigma = std(data).
    """
    if len(values) == 0:
        return {"posterior_mean": None}
    x = torch.tensor(values, dtype=torch.float32)
    n = x.numel()
    sample_mean = float(x.mean().item())
    sample_var = float(x.var(unbiased=False).item()) if n > 1 else 1.0

    # Using conjugate normal prior and known variance approximation:
    sigma2 = sample_var if sample_var > 0 else 1.0
    prior_var = prior_sigma ** 2
    post_var = 1.0 / (n / sigma2 + 1.0 / prior_var)
    post_mean = post_var * (n * sample_mean / sigma2 + prior_mu / prior_var)

    # approximate credible intervals
    post_sd = math.sqrt(post_var)
    ci_lower = post_mean - 1.96 * post_sd
    ci_upper = post_mean + 1.96 * post_sd

    return {
        "posterior_mean": float(post_mean),
        "posterior_sd": float(post_sd),
        "95ci": [float(ci_lower), float(ci_upper)],
        "n": n
    }