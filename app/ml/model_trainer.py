"""
Simple PyTorch model trainer for a tabular confidence predictor.
Saves and loads models to ./models (configurable).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from app.config import settings
import os
import joblib
import numpy as np

MODEL_DIR = settings.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3) -> str:
    device = torch.device("cpu")
    model = SimpleMLP(input_dim=X.shape[1])
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
    torch.save(model.state_dict(), str(fname))
    return str(fname)


def load_model(path: str = None, input_dim: int = None) -> SimpleMLP:
    if path is None:
        path = str(MODEL_DIR / settings.MODEL_DEFAULT_FILENAME)
    if input_dim is None:
        # when no shape, user should provide input_dim
        raise ValueError("input_dim required to construct model architecture")
    model = SimpleMLP(input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model