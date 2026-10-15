from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_hourly_load, make_split_sequences
from src.metrics import regression_metrics, uncertainty_metrics
from src.model import (
    LSTMForecaster,
    predict_deterministic,
    predict_with_mc_dropout,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an uncertainty-aware energy forecaster")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--max-hours", type=int, default=8760)
    parser.add_argument("--lookback", type=int, default=168, help="Past hours; 168 = one week")
    parser.add_argument("--horizon", type=int, default=24, help="Future hours to forecast")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


def save_forecast_plot(
    timestamps: np.ndarray,
    actual: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path,
) -> None:
    horizon = min(len(actual), 72)
    x = np.arange(horizon)
    plt.figure(figsize=(12, 5))
    plt.plot(x, actual[:horizon].reshape(-1), label="Actual", color="#1f2937", linewidth=2)
    plt.plot(x, mean[:horizon].reshape(-1), label="Bayesian LSTM mean", color="#2563eb")
    plt.fill_between(
        x,
        lower[:horizon].reshape(-1),
        upper[:horizon].reshape(-1),
        color="#93c5fd",
        alpha=0.45,
        label="95% predictive interval",
    )
    plt.title("Uncertainty-aware 24-hour household load forecast")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("Active power (kW)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def conformalize_intervals(
    validation_actual: np.ndarray,
    validation_lower: np.ndarray,
    validation_upper: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    coverage: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate interval width on held-out validation residuals.

    This split-conformal correction preserves the Bayesian mean forecast while
    widening intervals by a data-derived nonconformity quantile.
    """
    nonconformity = np.maximum.reduce(
        [
            validation_lower - validation_actual,
            validation_actual - validation_upper,
            np.zeros_like(validation_actual),
        ]
    ).reshape(-1)
    quantile_level = min(1.0, coverage * (len(nonconformity) + 1) / len(nonconformity))
    margin = float(np.quantile(nonconformity, quantile_level))
    return test_lower - margin, test_upper + margin, margin


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    frame = load_hourly_load(args.data_dir, max_hours=args.max_hours)
    splits, target_scaler, feature_columns = make_split_sequences(
        frame, lookback=args.lookback, horizon=args.horizon
    )
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    print(
        f"rows={len(frame)} | train={len(train['inputs'])} | "
        f"validation={len(validation['inputs'])} | test={len(test['inputs'])}"
    )

    model = LSTMForecaster(
        input_size=len(feature_columns),
        hidden_size=args.hidden_size,
        horizon=args.horizon,
        dropout=0.20,
    )
    model, history = train_model(
        model,
        train["inputs"],
        train["targets"],
        validation["inputs"],
        validation["targets"],
        epochs=args.epochs,
        device=args.device,
    )
    torch.save(model.state_dict(), artifacts / "bayesian_lstm.pt")

    validation_draws_scaled = predict_with_mc_dropout(
        model, validation["inputs"], samples=args.mc_samples, device=args.device
    )
    deterministic_scaled = predict_deterministic(model, test["inputs"], args.device)
    seasonal_naive_scaled = test["inputs"][:, -args.horizon :, 0]
    draws_scaled = predict_with_mc_dropout(
        model, test["inputs"], samples=args.mc_samples, device=args.device
    )
    actual = inverse_target(test["targets"], target_scaler)
    validation_actual = inverse_target(validation["targets"], target_scaler)
    deterministic = inverse_target(deterministic_scaled, target_scaler)
    seasonal_naive = inverse_target(seasonal_naive_scaled, target_scaler)
    validation_draws = inverse_target(validation_draws_scaled, target_scaler)
    draws = inverse_target(draws_scaled, target_scaler)
    validation_lower = np.quantile(validation_draws, 0.025, axis=0)
    validation_upper = np.quantile(validation_draws, 0.975, axis=0)
    mean = np.mean(draws, axis=0)
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    calibrated_lower, calibrated_upper, calibration_margin = conformalize_intervals(
        validation_actual,
        validation_lower,
        validation_upper,
        lower,
        upper,
    )

    metrics = {
        "dataset": "UCI Individual Household Electric Power Consumption",
        "configuration": vars(args),
        "seasonal_naive_baseline": regression_metrics(actual, seasonal_naive),
        "deterministic_baseline": regression_metrics(actual, deterministic),
        "bayesian_mc_dropout": {
            **regression_metrics(actual, mean),
            **uncertainty_metrics(actual, lower, upper),
        },
        "conformalized_bayesian_interval": {
            **regression_metrics(actual, mean),
            **uncertainty_metrics(actual, calibrated_lower, calibrated_upper),
            "calibration_margin_kw": calibration_margin,
        },
        "notes": [
            "MC dropout is an approximate Bayesian inference method, not an exact posterior.",
            "The interval coverage target is 95%; compare coverage and width together.",
        ],
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (artifacts / "training_history.json").write_text(json.dumps(history, indent=2))
    save_forecast_plot(
        test["timestamps"],
        actual[0],
        mean[0],
        calibrated_lower[0],
        calibrated_upper[0],
        artifacts / "forecast_intervals.png",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved artifacts to {artifacts.resolve()}")


if __name__ == "__main__":
    main()

