"""
Pool-based active learner using entropy (primary) and margin (secondary).
Stores pool and labels in SQLite via app/db.Label (simple functions).
"""
from typing import List, Dict, Tuple
import numpy as np
from scipy.special import softmax
from app.db import SessionLocal, Label
from sqlalchemy.exc import IntegrityError


def entropy_scores(probs: np.ndarray) -> np.ndarray:
    """
    probs: N x C numpy array of class probabilities
    returns entropy per row
    """
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return ent


def margin_scores(probs: np.ndarray) -> np.ndarray:
    """
    margin = difference between top two class probabilities (lower is more uncertain)
    """
    part = -np.diff(-np.sort(-probs, axis=1)[:, :2], axis=1).squeeze()
    return part


class ActiveLearner:
    def __init__(self, model_predict_fn):
        """
        model_predict_fn: callable(X_batch) -> np.ndarray of shape (N, C) probabilities
        """
        self.model_predict_fn = model_predict_fn

    def query(self, X_pool: np.ndarray, k: int = 10) -> List[int]:
        """
        Return indices of top-k samples to label.
        Primary: entropy, secondary filter margin.
        """
        probs = self.model_predict_fn(X_pool)  # N x C
        ent = entropy_scores(probs)
        margins = margin_scores(probs)
        # primary ordering by entropy desc, then margin asc
        order = np.lexsort((margins, -ent))
        return list(order[:k])

    def submit_label(self, sample_id: str, label: str, annotator: str = None):
        db = SessionLocal()
        try:
            rec = Label(sample_id=sample_id, label=label, annotator=annotator)
            db.add(rec)
            db.commit()
        except IntegrityError:
            db.rollback()
        finally:
            db.close()