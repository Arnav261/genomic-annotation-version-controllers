"""
Model training utilities with metadata save/load for robust startup.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, Any
from app.config import settings
import os
import numpy as np
from datetime import datetime

MODEL_DIR = settings.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X: np.ndarray, y: np.ndarray, feature_names: list = None, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, seed: int = 42) -> str:
    # determinism
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu")
    input_dim = X.shape[1]
    model = SimpleMLP(input_dim=input_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
    fname = MODEL_DIR / settings.MODEL_DEFAULT_FILENAME
    # save state_dict + metadata
    meta = {
        "input_dim": int(input_dim),
        "feature_names": feature_names or [],
        "trained_at": datetime.utcnow().isoformat(),
        "framework": "pytorch"
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, str(fname))
    return str(fname)


def load_model_with_meta(path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    if path is None:
        path = str(MODEL_DIR / settings.MODEL_DEFAULT_FILENAME)
    data = torch.load(path, map_location="cpu")
    meta = data.get("meta", {})
    input_dim = meta.get("input_dim")
    if input_dim is None:
        raise ValueError("model metadata missing input_dim; retrain or provide input_dim")
    model = SimpleMLP(input_dim=input_dim)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, meta