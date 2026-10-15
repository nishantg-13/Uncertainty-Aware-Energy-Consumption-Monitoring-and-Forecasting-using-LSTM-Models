from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceLSTM(nn.Module):
    """LSTM that reconstructs a sequence or predicts a future horizon."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        output_size: int = 1,
        layers: int = 2,
        dropout: float = 0.2,
        output_steps: int | None = None,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(inputs)
        if self.output_steps is None:
            return self.head(self.dropout(outputs)).squeeze(-1)
        return self.head(self.dropout(outputs[:, -1, :])).squeeze(-1)


def fit_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    loss_name: str = "mse",
) -> tuple[nn.Module, dict[str, list[float]]]:
    device_obj = torch.device(device)
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss() if loss_name == "smooth_l1" else nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_inputs = torch.from_numpy(validation_x).to(device_obj)
    validation_targets = torch.from_numpy(validation_y).to(device_obj)
    history = {"train_loss": [], "validation_loss": []}
    best_state, best_loss = None, float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            count += len(inputs)

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_inputs), validation_targets).item()
        train_loss = total_loss / max(1, count)
        history["train_loss"].append(float(train_loss))
        history["validation_loss"].append(float(validation_loss))
        print(
            f"epoch {epoch:02d}/{epochs} | train={train_loss:.4f} | "
            f"validation={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, inputs: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(inputs).to(device)).cpu().numpy()


def predict_mc(
    model: nn.Module,
    inputs: np.ndarray,
    samples: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """Return [samples, observations, horizon] MC-Dropout predictions."""
    model.train()
    tensor = torch.from_numpy(inputs).to(device)
    draws = []
    with torch.no_grad():
        for _ in range(samples):
            draws.append(model(tensor).cpu().numpy())
    model.eval()
    return np.stack(draws, axis=0)

