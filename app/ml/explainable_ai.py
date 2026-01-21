"""
Lightweight explainability functions:
- permutation importance for tabular features
- gradient-based saliency for pytorch models (input gradients)
"""
from typing import Callable, List
import numpy as np
import torch


def permutation_importance(predict_fn: Callable, X: np.ndarray, y: np.ndarray, n_repeats: int = 5) -> List[float]:
    """
    predict_fn: X -> predictions (probabilities or scores)
    Returns importance per feature (higher is more important)
    """
    baseline = predict_fn(X)
    if baseline.ndim > 1:
        baseline_score = np.mean(baseline[:, 1] if baseline.shape[1] > 1 else baseline)
    else:
        baseline_score = np.mean(baseline)
    importances = []
    for col in range(X.shape[1]):
        scores = []
        Xp = X.copy()
        for _ in range(n_repeats):
            np.random.shuffle(Xp[:, col])
            s = predict_fn(Xp)
            if s.ndim > 1:
                s = np.mean(s[:, 1] if s.shape[1] > 1 else s)
            else:
                s = np.mean(s)
            scores.append(baseline_score - s)
        importances.append(float(np.mean(scores)))
    return importances


def gradient_saliency(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of model output w.r.t input features for a single example.
    Returns absolute gradient per input feature.
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x.unsqueeze(0))
    # if binary prob, out is scalar; if vector, take max
    if out.dim() > 0:
        out = out.squeeze()
        if out.dim() > 0:
            out = out[0]
    out.backward()
    grad = x.grad.abs().squeeze(0)
    return grad.detach()