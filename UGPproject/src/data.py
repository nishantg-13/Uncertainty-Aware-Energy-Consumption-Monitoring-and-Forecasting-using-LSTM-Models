from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
DATA_FILE = "household_power_consumption.txt"


def download_dataset(data_dir: str | Path) -> Path:
    """Download the UCI household electricity dataset if it is not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / DATA_FILE
    if target.exists():
        return target

    print("Downloading UCI household electricity data...")
    with urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open(DATA_FILE) as source, target.open("wb") as destination:
            destination.write(source.read())
    return target


def load_hourly_load(
    data_dir: str | Path,
    max_hours: int | None = 8760,
) -> pd.DataFrame:
    """Load and aggregate household active power to hourly kW observations."""
    path = download_dataset(data_dir)
    frame = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values="?",
        low_memory=False,
    )
    frame["timestamp"] = pd.to_datetime(
        frame["Date"] + " " + frame["Time"], dayfirst=True
    )
    frame["load_kw"] = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    frame = frame[["timestamp", "load_kw"]].dropna().set_index("timestamp")
    hourly = frame["load_kw"].resample("1h").mean().interpolate(limit_direction="both")
    hourly = hourly.to_frame().reset_index()

    # Retain a contiguous recent window so the demo trains quickly while
    # preserving daily and weekly seasonality.
    if max_hours is not None and len(hourly) > max_hours:
        hourly = hourly.tail(max_hours).reset_index(drop=True)
    return hourly


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical calendar features without leaking future load values."""
    result = frame.copy()
    timestamp = result["timestamp"]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_of_week = timestamp.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    return result


def make_split_sequences(
    frame: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 24,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    """Build chronological train/validation/test tensors as NumPy arrays.

    Scaling is fitted only on the training partition. Each partition creates
    sequences locally, preventing validation/test labels from influencing the
    training scaler or sequence boundaries.
    """
    feature_columns = ["load_kw", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    frame = add_time_features(frame)
    n_rows = len(frame)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    partitions = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }

    feature_scaler = StandardScaler().fit(partitions["train"][feature_columns])
    target_scaler = StandardScaler().fit(partitions["train"][["load_kw"]])
    result = {}
    for name, partition in partitions.items():
        features = feature_scaler.transform(partition[feature_columns]).astype(np.float32)
        targets = target_scaler.transform(partition[["load_kw"]]).astype(np.float32).ravel()
        inputs, labels, timestamps = [], [], []
        for end in range(lookback, len(partition) - horizon + 1):
            inputs.append(features[end - lookback : end])
            labels.append(targets[end : end + horizon])
            timestamps.append(partition["timestamp"].iloc[end])
        result[name] = {
            "inputs": np.asarray(inputs, dtype=np.float32),
            "targets": np.asarray(labels, dtype=np.float32),
            "timestamps": np.asarray(timestamps),
        }

    return result, target_scaler, feature_columns

