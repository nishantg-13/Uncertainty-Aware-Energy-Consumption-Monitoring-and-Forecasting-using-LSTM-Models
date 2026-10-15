from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """Direct multi-horizon LSTM; dropout is the approximate Bayesian posterior."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        representation = self.dropout(outputs[:, -1, :])
        return self.head(representation)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cpu",
):
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_inputs = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    history = {"train_loss": [], "validation_loss": []}
    best_state, best_loss = None, float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, observations = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            observations += len(inputs)

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_inputs), validation_targets).item()
        train_loss = total_loss / max(observations, 1)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"epoch {epoch:02d}/{epochs} | train={train_loss:.4f} | "
            f"validation={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_deterministic(model: nn.Module, inputs: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(inputs).to(device)).cpu().numpy()
    return predictions


def predict_with_mc_dropout(
    model: nn.Module,
    inputs: np.ndarray,
    samples: int = 100,
    device: str = "cpu",
) -> np.ndarray:
    """Sample the predictive distribution using Monte Carlo dropout."""
    model.train()  # intentionally active: dropout approximates posterior sampling
    tensor_inputs = torch.from_numpy(inputs).to(device)
    draws = []
    with torch.no_grad():
        for _ in range(samples):
            draws.append(model(tensor_inputs).cpu().numpy())
    model.eval()
    return np.stack(draws, axis=0)

