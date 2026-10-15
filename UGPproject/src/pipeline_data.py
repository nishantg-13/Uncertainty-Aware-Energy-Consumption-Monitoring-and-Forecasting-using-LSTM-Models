from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests


UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_FILE = "household_power_consumption.txt"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "radiation_wm2",
]
CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]


def download_energy_dataset(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / UCI_FILE
    if target.exists():
        return target

    print("Downloading UCI household electricity data...")
    with urlopen(UCI_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open(UCI_FILE) as source, target.open("wb") as destination:
            destination.write(source.read())
    return target


def load_15_minute_energy(
    data_dir: str | Path,
    max_days: int | None = 60,
) -> pd.DataFrame:
    """Load UCI minute-level power readings and aggregate them to 15 minutes."""
    path = download_energy_dataset(data_dir)
    frame = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values="?",
        low_memory=False,
    )
    frame["timestamp"] = pd.to_datetime(
        frame["Date"] + " " + frame["Time"], dayfirst=True, errors="coerce"
    )
    frame["energy_kw"] = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    frame = frame[["timestamp", "energy_kw"]].dropna().set_index("timestamp")
    quarter_hour = frame["energy_kw"].resample("15min").mean()
    quarter_hour = quarter_hour.interpolate(limit_direction="both")
    result = quarter_hour.to_frame().reset_index()
    if max_days is not None:
        result = result.tail(max_days * 96).reset_index(drop=True)
    return result


def _synthetic_weather(timestamps: pd.Series) -> pd.DataFrame:
    """Offline fallback with realistic daily/seasonal weather patterns."""
    hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
    day_number = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 86400.0
    daily = np.sin(2 * np.pi * (hours - 7) / 24.0)
    radiation = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12.0)) * 700
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": 27 + 8 * daily + 2 * np.sin(2 * np.pi * day_number / 30),
            "humidity_pct": np.clip(68 - 18 * daily, 20, 100),
            "precipitation_mm": np.zeros(len(timestamps)),
            "radiation_wm2": radiation,
        }
    )


def fetch_open_meteo_weather(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    cache_path: str | Path | None = None,
    offline: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Fetch historical hourly weather and interpolate it to 15-minute data.

    Open-Meteo's historical archive exposes the requested hourly variables and
    does not require an API key. A deterministic fallback keeps experiments
    reproducible when the network is unavailable.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return cached, "open-meteo-cache"
    if offline:
        return _synthetic_weather(timestamps), "synthetic-offline"

    start_date = pd.Timestamp(timestamps.iloc[0]).date().isoformat()
    end_date = pd.Timestamp(timestamps.iloc[-1]).date().isoformat()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,shortwave_radiation",
        "timezone": "UTC",
    }
    try:
        response = requests.get(
            f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}", timeout=90
        )
        response.raise_for_status()
        payload = response.json()["hourly"]
        hourly = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(payload["time"]),
                "temperature_c": payload["temperature_2m"],
                "humidity_pct": payload["relative_humidity_2m"],
                "precipitation_mm": payload["precipitation"],
                "radiation_wm2": payload["shortwave_radiation"],
            }
        )
        hourly = hourly.set_index("timestamp").sort_index()
        weather = hourly.resample("15min").interpolate(limit_direction="both").reset_index()
        weather = weather[(weather["timestamp"] >= timestamps.iloc[0]) & (weather["timestamp"] <= timestamps.iloc[-1])]
        weather = weather.reset_index(drop=True)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            weather.to_csv(cache_path, index=False)
        return weather, "open-meteo-archive"
    except Exception as exc:
        print(f"Weather API unavailable ({exc}); using deterministic fallback weather.")
        return _synthetic_weather(timestamps), "synthetic-fallback"


def merge_weather(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    energy = energy.sort_values("timestamp").reset_index(drop=True)
    weather = weather.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        energy,
        weather,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    for column in WEATHER_COLUMNS:
        merged[column] = merged[column].interpolate(limit_direction="both")
    return add_calendar_features(merged)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result["timestamp"]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day_of_week = timestamp.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    result["is_weekend"] = (day_of_week >= 5).astype(float)
    return result


def add_load_features(frame: pd.DataFrame, load_column: str) -> pd.DataFrame:
    """Add causal rolling and lag features using the selected load series."""
    result = frame.copy()
    load = result[load_column]
    result["load_roll_mean_1h"] = load.rolling(4, min_periods=1).mean()
    result["load_roll_std_1h"] = load.rolling(4, min_periods=1).std().fillna(0.0)
    result["load_roll_mean_4h"] = load.rolling(16, min_periods=1).mean()
    result["load_roll_std_4h"] = load.rolling(16, min_periods=1).std().fillna(0.0)
    result["load_lag_15m"] = load.shift(1).bfill()
    result["load_lag_1h"] = load.shift(4).bfill()
    result["load_lag_1d"] = load.shift(96).bfill()
    return result


def inject_synthetic_anomalies(
    frame: pd.DataFrame,
    anomaly_rate: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Corrupt clean readings in known locations for measurable evaluation."""
    result = frame.copy()
    clean = result["energy_kw"].to_numpy(dtype=float)
    observed = clean.copy()
    labels = np.zeros(len(result), dtype=bool)
    types = np.full(len(result), "normal", dtype=object)
    rng = np.random.default_rng(seed)
    spread = float(np.nanstd(clean))
    event_count = max(4, int(len(result) * anomaly_rate / 12))
    event_lengths = {"missing": 8, "flatline": 12, "repeated_random": 8, "spike": 1}
    event_types = list(event_lengths)
    used = np.zeros(len(result), dtype=bool)

    for event_number in range(event_count):
        event_type = event_types[event_number % len(event_types)]
        length = event_lengths[event_type]
        for _ in range(100):
            start = int(rng.integers(96, max(97, len(result) - length - 1)))
            if not used[start : start + length].any():
                break
        end = min(start + length, len(result))
        used[start:end] = True
        labels[start:end] = True
        types[start:end] = event_type
        if event_type == "missing":
            observed[start:end] = np.nan
        elif event_type == "flatline":
            observed[start:end] = clean[start]
        elif event_type == "repeated_random":
            observed[start:end] = rng.uniform(
                np.quantile(clean, 0.10), np.quantile(clean, 0.90)
            )
        else:
            observed[start] = max(0.0, clean[start] + rng.choice([-1, 1]) * 5 * spread)

    result["clean_energy_kw"] = clean
    result["observed_energy_kw"] = observed
    result["synthetic_anomaly"] = labels
    result["synthetic_anomaly_type"] = types
    return result


def make_windows(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, labels, starts = [], [], []
    for start in range(0, len(features) - lookback - horizon + 1, stride):
        end = start + lookback
        inputs.append(features[start:end])
        labels.append(targets[end : end + horizon])
        starts.append(start)
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(starts, dtype=np.int64),
    )

