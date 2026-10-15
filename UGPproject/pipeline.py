from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.metrics import (
    binary_detection_metrics,
    masked_regression_metrics,
    regression_metrics,
    uncertainty_metrics,
)
from src.pipeline_data import (
    CALENDAR_COLUMNS,
    WEATHER_COLUMNS,
    add_load_features,
    fetch_open_meteo_weather,
    inject_synthetic_anomalies,
    load_15_minute_energy,
    make_windows,
    merge_weather,
)
from src.pipeline_models import SequenceLSTM, fit_model, predict, predict_mc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end energy intelligence pipeline")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--artifacts-dir", default="artifacts/pipeline")
    parser.add_argument("--max-days", type=int, default=60)
    parser.add_argument("--latitude", type=float, default=26.5123, help="IIT Kanpur latitude")
    parser.add_argument("--longitude", type=float, default=80.2329, help="IIT Kanpur longitude")
    parser.add_argument("--offline-weather", action="store_true")
    parser.add_argument("--lookback", type=int, default=96, help="Past 15-minute intervals; 96 = 24 hours")
    parser.add_argument("--horizon", type=int, default=96, help="Future 15-minute intervals; 96 = 24 hours")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_1d(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = np.asarray(values).shape
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(shape)


def make_reconstruction_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create same-length input/target windows for reconstruction models."""
    inputs, labels, starts = [], [], []
    for start in range(0, len(features) - lookback + 1, stride):
        end = start + lookback
        inputs.append(features[start:end])
        labels.append(target[start:end])
        starts.append(start)
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(starts, dtype=np.int64),
    )


def aggregate_sequence_predictions(
    predictions: np.ndarray,
    starts: np.ndarray,
    length: int,
    lookback: int,
) -> np.ndarray:
    totals = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for prediction, start in zip(predictions, starts):
        end = min(start + lookback, length)
        width = end - start
        totals[start:end] += prediction[:width]
        counts[start:end] += 1.0
    result = totals / np.maximum(counts, 1.0)
    if np.any(counts == 0):
        first_valid = result[counts > 0][0]
        result[counts == 0] = first_valid
    return result.astype(np.float32)


def rolling_rule_flags(observed: pd.Series) -> np.ndarray:
    filled = observed.interpolate(limit_direction="both").bfill().ffill()
    rolling_median = filled.rolling(9, center=True, min_periods=1).median()
    rolling_mad = (filled - rolling_median).abs().rolling(9, center=True, min_periods=1).median()
    robust_z = 0.6745 * (filled - rolling_median) / rolling_mad.clip(lower=1e-3)
    spike = robust_z.abs() > 6.0
    rolling_range = filled.rolling(8, min_periods=8).max() - filled.rolling(8, min_periods=8).min()
    flatline = rolling_range.fillna(np.inf) < 1e-5
    repeated = (
        filled.rolling(8, min_periods=8)
        .apply(lambda values: float(len(np.unique(np.round(values, 5))) <= 2), raw=True)
        .fillna(0.0)
        .astype(bool)
    )
    return (observed.isna() | spike | flatline | repeated).to_numpy(dtype=bool)


def train_reconstruction_detector(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    covariates = [*WEATHER_COLUMNS, *CALENDAR_COLUMNS]
    # Fit on the training period, then transform the complete frame.
    clean_values = frame[["clean_energy_kw", *covariates]].astype(float).to_numpy()
    feature_scaler = StandardScaler().fit(clean_values[:train_end])
    clean_features = feature_scaler.transform(clean_values).astype(np.float32)
    target_scaler = StandardScaler().fit(frame[["clean_energy_kw"]].iloc[:train_end])
    clean_target = target_scaler.transform(frame[["clean_energy_kw"]]).ravel().astype(np.float32)

    train_x, train_y, _ = make_reconstruction_windows(
        clean_features[:train_end], clean_target[:train_end], args.lookback, stride=4
    )
    split = max(2, int(len(train_x) * 0.85))
    detector = SequenceLSTM(
        input_size=train_x.shape[-1],
        hidden_size=args.hidden_size,
        output_size=1,
        output_steps=None,
        dropout=0.20,
    )
    detector, history = fit_model(
        detector,
        train_x[:split],
        train_y[:split],
        train_x[split:],
        train_y[split:],
        epochs=args.epochs,
        device=args.device,
    )

    filled_observed = (
        frame["observed_energy_kw"].interpolate(limit_direction="both").bfill().ffill().to_numpy()
    )
    observed_values = frame[["observed_energy_kw", *covariates]].astype(float).copy()
    observed_values["observed_energy_kw"] = filled_observed
    observed_scaled = feature_scaler.transform(observed_values.to_numpy()).astype(np.float32)
    all_x, _, all_starts = make_reconstruction_windows(
        observed_scaled, np.zeros(len(frame), dtype=np.float32), args.lookback, stride=4
    )
    reconstruction_scaled = aggregate_sequence_predictions(
        predict(detector, all_x, args.device), all_starts, len(frame), args.lookback
    )
    observed_scaled_load = observed_scaled[:, 0]
    error = np.abs(observed_scaled_load - reconstruction_scaled)

    clean_x, _, clean_starts = make_reconstruction_windows(
        clean_features[:train_end], clean_target[:train_end], args.lookback, stride=4
    )
    clean_reconstruction = aggregate_sequence_predictions(
        predict(detector, clean_x, args.device), clean_starts, train_end, args.lookback
    )
    threshold = float(np.quantile(np.abs(clean_target[:train_end] - clean_reconstruction), 0.995))
    rule_flags = rolling_rule_flags(frame["observed_energy_kw"])
    detected_flags = rule_flags | (error > threshold)
    details = {
        "threshold_scaled_error": threshold,
        "rule_flag_count": int(rule_flags.sum()),
        "reconstruction_flag_count": int((error > threshold).sum()),
        "history": history,
        "feature_columns": ["load"] + covariates,
    }
    return detected_flags, error, details


def build_imputation_training_data(
    frame: pd.DataFrame,
    train_end: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    covariates = [*WEATHER_COLUMNS, *CALENDAR_COLUMNS]
    values = frame[["clean_energy_kw", *covariates]].astype(float).to_numpy()
    feature_scaler = StandardScaler().fit(values[:train_end])
    target_scaler = StandardScaler().fit(frame[["clean_energy_kw"]].iloc[:train_end])
    scaled = feature_scaler.transform(values).astype(np.float32)
    target = target_scaler.transform(frame[["clean_energy_kw"]]).ravel().astype(np.float32)
    clean_x, clean_y, _ = make_reconstruction_windows(
        scaled[:train_end], target[:train_end], args.lookback, stride=4
    )
    mask = rng.random(clean_x.shape[:2]) < 0.15
    for row in range(len(mask)):
        if args.lookback >= 12:
            start = int(rng.integers(0, args.lookback - 8))
            mask[row, start : start + 8] = True
    masked_load = clean_x[:, :, 0].copy()
    masked_load[mask] = 0.0
    imputer_x = np.concatenate(
        [masked_load[..., None], mask.astype(np.float32)[..., None], clean_x[:, :, 1:]],
        axis=2,
    )
    split = max(2, int(len(imputer_x) * 0.85))
    return (
        imputer_x[:split],
        clean_y[:split],
        imputer_x[split:],
        clean_y[split:],
        feature_scaler,
        target_scaler,
    )


def train_and_apply_imputer(
    frame: pd.DataFrame,
    detected_flags: np.ndarray,
    args: argparse.Namespace,
    train_end: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(args.seed + 1)
    train_x, train_y, validation_x, validation_y, feature_scaler, target_scaler = build_imputation_training_data(
        frame, train_end, args, rng
    )
    model = SequenceLSTM(
        input_size=train_x.shape[-1],
        hidden_size=args.hidden_size,
        output_size=1,
        output_steps=None,
        dropout=0.20,
    )
    model, history = fit_model(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=args.epochs,
        device=args.device,
        loss_name="smooth_l1",
    )

    covariates = [*WEATHER_COLUMNS, *CALENDAR_COLUMNS]
    filled = frame["observed_energy_kw"].interpolate(limit_direction="both").bfill().ffill().to_numpy()
    observed_values = frame[["observed_energy_kw", *covariates]].astype(float).copy()
    observed_values["observed_energy_kw"] = filled
    observed_scaled = feature_scaler.transform(observed_values.to_numpy()).astype(np.float32)
    mask = detected_flags.astype(np.float32)
    imputer_inputs = np.concatenate(
        [observed_scaled[:, 0:1], mask[:, None], observed_scaled[:, 1:]], axis=1
    )
    input_windows, _, starts = make_reconstruction_windows(
        imputer_inputs, np.zeros(len(frame), dtype=np.float32), args.lookback, stride=4
    )
    predictions_scaled = aggregate_sequence_predictions(
        predict(model, input_windows, args.device), starts, len(frame), args.lookback
    )
    lstm_values = inverse_1d(predictions_scaled, target_scaler)
    linear_values = pd.Series(frame["observed_energy_kw"]).interpolate(limit_direction="both").to_numpy()
    ffill_values = pd.Series(frame["observed_energy_kw"]).ffill().bfill().to_numpy()
    repaired = filled.copy()
    repaired[detected_flags] = lstm_values[detected_flags]
    output = frame.copy()
    output["linear_imputed_kw"] = linear_values
    output["ffill_imputed_kw"] = ffill_values
    output["lstm_imputed_kw"] = repaired
    metrics = {
        "linear_interpolation": masked_regression_metrics(
            output["clean_energy_kw"], linear_values, output["synthetic_anomaly"]
        ),
        "forward_fill": masked_regression_metrics(
            output["clean_energy_kw"], ffill_values, output["synthetic_anomaly"]
        ),
        "lstm_imputation": masked_regression_metrics(
            output["clean_energy_kw"], repaired, output["synthetic_anomaly"]
        ),
        "history": history,
        "masked_points_by_detector": int(detected_flags.sum()),
    }
    return output, metrics


def make_forecast_split(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
    train_end: int,
    validation_end: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    partitions = {"train": [], "validation": [], "test": []}
    for start in range(0, len(features) - lookback - horizon + 1, 4):
        target_start = start + lookback
        target_end = target_start + horizon
        if target_end <= train_end:
            name = "train"
        elif target_start >= train_end and target_end <= validation_end:
            name = "validation"
        elif target_start >= validation_end:
            name = "test"
        else:
            continue
        partitions[name].append(
            (features[start : start + lookback], target[target_start:target_end], start)
        )
    result = {}
    for name, rows in partitions.items():
        result[name] = (
            np.asarray([row[0] for row in rows], dtype=np.float32),
            np.asarray([row[1] for row in rows], dtype=np.float32),
            np.asarray([row[2] for row in rows], dtype=np.int64),
        )
    return result


def train_forecaster(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    train_end: int,
    validation_end: int,
) -> tuple[dict[str, object], dict[str, object]]:
    forecast_frame = add_load_features(frame, "lstm_imputed_kw")
    feature_columns = [
        "lstm_imputed_kw",
        *WEATHER_COLUMNS,
        *CALENDAR_COLUMNS,
        "load_roll_mean_1h",
        "load_roll_std_1h",
        "load_roll_mean_4h",
        "load_roll_std_4h",
        "load_lag_15m",
        "load_lag_1h",
        "load_lag_1d",
    ]
    feature_values = forecast_frame[feature_columns].astype(float).to_numpy()
    feature_scaler = StandardScaler().fit(feature_values[:train_end])
    scaled_features = feature_scaler.transform(feature_values).astype(np.float32)
    target_scaler = StandardScaler().fit(frame[["clean_energy_kw"]].iloc[:train_end])
    target = target_scaler.transform(frame[["clean_energy_kw"]]).ravel().astype(np.float32)
    splits = make_forecast_split(
        scaled_features,
        target,
        args.lookback,
        args.horizon,
        train_end,
        validation_end,
    )
    train_x, train_y, _ = splits["train"]
    validation_x, validation_y, _ = splits["validation"]
    test_x, test_y, test_starts = splits["test"]
    model = SequenceLSTM(
        input_size=train_x.shape[-1],
        hidden_size=args.hidden_size,
        output_size=args.horizon,
        output_steps=args.horizon,
        dropout=0.20,
    )
    model, history = fit_model(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=args.epochs,
        device=args.device,
        loss_name="smooth_l1",
    )
    validation_draws = predict_mc(model, validation_x, args.mc_samples, args.device)
    test_draws = predict_mc(model, test_x, args.mc_samples, args.device)
    validation_actual = inverse_1d(validation_y, target_scaler)
    actual = inverse_1d(test_y, target_scaler)
    validation_draws = inverse_1d(validation_draws, target_scaler)
    draws = inverse_1d(test_draws, target_scaler)
    mean = draws.mean(axis=0)
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    validation_lower = np.quantile(validation_draws, 0.025, axis=0)
    validation_upper = np.quantile(validation_draws, 0.975, axis=0)
    nonconformity = np.maximum.reduce(
        [validation_lower - validation_actual, validation_actual - validation_upper, np.zeros_like(validation_actual)]
    ).reshape(-1)
    margin = float(np.quantile(nonconformity, min(1.0, 0.95 * (len(nonconformity) + 1) / len(nonconformity))))
    calibrated_lower, calibrated_upper = lower - margin, upper + margin
    seasonal_naive = inverse_1d(test_x[:, -args.horizon :, 0], target_scaler)
    metrics = {
        "seasonal_naive": regression_metrics(actual, seasonal_naive),
        "bayesian_mc_dropout": {
            **regression_metrics(actual, mean),
            **uncertainty_metrics(actual, lower, upper),
        },
        "conformalized_bayesian_interval": {
            **regression_metrics(actual, mean),
            **uncertainty_metrics(actual, calibrated_lower, calibrated_upper),
            "calibration_margin_kw": margin,
        },
        "history": history,
        "feature_columns": feature_columns,
    }
    predictions = {
        "actual": actual,
        "mean": mean,
        "lower": calibrated_lower,
        "upper": calibrated_upper,
        "test_starts": test_starts,
        "model": model,
    }
    return predictions, metrics


def tariff_for_timestamp(timestamp: pd.Timestamp) -> float:
    hour = timestamp.hour + timestamp.minute / 60.0
    if 17 <= hour < 22:
        return 0.22
    if 0 <= hour < 7:
        return 0.08
    return 0.14


def simulate_energy_optimization(
    forecast_kw: np.ndarray,
    start_timestamps: np.ndarray,
    horizon: int,
    flexible_fraction: float = 0.20,
) -> dict[str, float | str]:
    daily_savings = []
    baseline_costs = []
    optimized_costs = []
    for load, start in zip(forecast_kw[::horizon], start_timestamps[::horizon]):
        timestamps = pd.date_range(pd.Timestamp(start), periods=horizon, freq="15min")
        tariff = np.asarray([tariff_for_timestamp(timestamp) for timestamp in timestamps])
        energy_kwh = np.maximum(load, 0.0) * 0.25
        flexible_kwh = energy_kwh * flexible_fraction
        peak = tariff >= 0.20
        offpeak = tariff <= 0.08
        shift_kwh = float(flexible_kwh[peak].sum())
        if shift_kwh == 0 or not np.any(offpeak):
            saving = 0.0
        else:
            peak_rate = float(np.average(tariff[peak], weights=flexible_kwh[peak]))
            offpeak_rate = float(np.min(tariff[offpeak]))
            saving = max(0.0, shift_kwh * (peak_rate - offpeak_rate))
        baseline_costs.append(float(np.sum(energy_kwh * tariff)))
        optimized_costs.append(float(np.sum(energy_kwh * tariff) - saving))
        daily_savings.append(saving)
    average_daily_savings = float(np.mean(daily_savings)) if daily_savings else 0.0
    return {
        "average_daily_baseline_cost_usd": float(np.mean(baseline_costs)) if baseline_costs else 0.0,
        "average_daily_optimized_cost_usd": float(np.mean(optimized_costs)) if optimized_costs else 0.0,
        "average_daily_savings_usd": average_daily_savings,
        "estimated_annual_savings_usd": average_daily_savings * 365.0,
        "flexible_load_fraction": flexible_fraction,
        "tariff_assumption": "$0.08 off-peak (00:00-07:00), $0.14 standard, $0.22 peak (17:00-22:00)",
        "note": "Scenario estimate; replace tariff and flexible-load assumptions with local data.",
    }


def save_plots(
    frame: pd.DataFrame,
    forecast: dict[str, object],
    artifacts: Path,
    horizon: int,
) -> None:
    view = frame.head(min(len(frame), 7 * 96))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].plot(view["timestamp"], view["clean_energy_kw"], label="Clean reference", linewidth=1.5)
    axes[0].plot(view["timestamp"], view["observed_energy_kw"], label="Corrupted observations", alpha=0.7)
    axes[0].plot(view["timestamp"], view["lstm_imputed_kw"], label="LSTM repaired", linewidth=1.2)
    anomaly_view = view[view["synthetic_anomaly"]]
    axes[0].scatter(anomaly_view["timestamp"], anomaly_view["observed_energy_kw"], label="Injected anomaly", color="red", s=12)
    axes[0].set_title("15-minute energy quality repair")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    actual = np.asarray(forecast["actual"])[0]
    mean = np.asarray(forecast["mean"])[0]
    lower = np.asarray(forecast["lower"])[0]
    upper = np.asarray(forecast["upper"])[0]
    x = np.arange(min(horizon, len(actual)))
    axes[1].plot(x, actual[: len(x)], label="Actual", color="#111827", linewidth=2)
    axes[1].plot(x, mean[: len(x)], label="Bayesian LSTM mean", color="#2563eb")
    axes[1].fill_between(x, lower[: len(x)], upper[: len(x)], alpha=0.3, color="#60a5fa", label="95% calibrated interval")
    axes[1].set_title("24-hour probabilistic forecast")
    axes[1].set_xlabel("Future 15-minute intervals")
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)
    fig.savefig(artifacts / "pipeline_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    energy = load_15_minute_energy(args.data_dir, max_days=args.max_days)
    weather, weather_source = fetch_open_meteo_weather(
        energy["timestamp"],
        args.latitude,
        args.longitude,
        cache_path=Path(args.data_dir) / "open_meteo_weather.csv",
        offline=args.offline_weather,
    )
    frame = merge_weather(energy, weather)
    frame = inject_synthetic_anomalies(frame, seed=args.seed)
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)

    detected_flags, reconstruction_error, detection_details = train_reconstruction_detector(
        frame, args, train_end
    )
    frame["detected_anomaly"] = detected_flags
    frame["reconstruction_error"] = reconstruction_error
    detection_metrics = binary_detection_metrics(frame["synthetic_anomaly"], detected_flags)

    repaired_frame, imputation_metrics = train_and_apply_imputer(
        frame, detected_flags, args, train_end
    )
    forecast_predictions, forecast_metrics = train_forecaster(
        repaired_frame, args, train_end, validation_end
    )
    test_starts = np.asarray(forecast_predictions["test_starts"])
    start_timestamps = frame["timestamp"].iloc[test_starts + args.lookback].to_numpy()
    optimization = simulate_energy_optimization(
        np.asarray(forecast_predictions["mean"]),
        start_timestamps,
        args.horizon,
    )

    repaired_frame[
        [
            "timestamp",
            "clean_energy_kw",
            "observed_energy_kw",
            "detected_anomaly",
            "synthetic_anomaly",
            "synthetic_anomaly_type",
            "linear_imputed_kw",
            "ffill_imputed_kw",
            "lstm_imputed_kw",
            *WEATHER_COLUMNS,
        ]
    ].to_csv(artifacts / "energy_quality_results.csv", index=False)
    forecast_timestamps = pd.date_range(
        pd.Timestamp(start_timestamps[0]), periods=args.horizon, freq="15min"
    )
    pd.DataFrame(
        {
            "timestamp": forecast_timestamps,
            "actual_kw": np.asarray(forecast_predictions["actual"])[0],
            "forecast_mean_kw": np.asarray(forecast_predictions["mean"])[0],
            "interval_lower_kw": np.asarray(forecast_predictions["lower"])[0],
            "interval_upper_kw": np.asarray(forecast_predictions["upper"])[0],
        }
    ).to_csv(artifacts / "forecast_intervals.csv", index=False)
    (artifacts / "optimization_summary.json").write_text(json.dumps(optimization, indent=2))
    metrics = {
        "data": {
            "rows": len(frame),
            "frequency": "15 minutes",
            "weather_source": weather_source,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "synthetic_anomaly_rate": float(frame["synthetic_anomaly"].mean()),
        },
        "anomaly_detection": {**detection_metrics, **detection_details},
        "imputation": imputation_metrics,
        "forecasting": forecast_metrics,
        "optimization": optimization,
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (artifacts / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    save_plots(repaired_frame, forecast_predictions, artifacts, args.horizon)
    torch.save(forecast_predictions["model"].state_dict(), artifacts / "bayesian_forecaster.pt")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Saved end-to-end artifacts to {artifacts.resolve()}")


if __name__ == "__main__":
    main()

