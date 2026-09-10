"""
Data loading, validation, and synthetic track generation for iceberg drift prediction.
"""

from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
from .utils import validate_observation, ensure_directories
from .config import DATA_SAMPLE_DIR, DATA_RAW_DIR, FORECAST_INTERVAL_HOURS


def generate_synthetic_iceberg_tracks(
    num_icebergs: int = 3,
    points_per_iceberg: int = 50,
    start_date: str = "2026-09-01T00:00:00Z",
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Generates synthetic historical Antarctic iceberg trajectories for testing and training.
    """
    ensure_directories()
    np.random.seed(42)

    start_dt = pd.to_datetime(start_date)
    records: List[Dict[str, Any]] = []

    # Configured trajectories for different icebergs
    iceberg_configs = [
        {"id": "IB001", "start_lat": -64.10, "start_lon": 52.10, "drift_lat": 0.02, "drift_lon": 0.05},
        {"id": "IB002", "start_lat": -65.20, "start_lon": 50.50, "drift_lat": 0.015, "drift_lon": 0.035},
        {"id": "IB003", "start_lat": -63.80, "start_lon": 53.00, "drift_lat": 0.025, "drift_lon": 0.04},
    ]

    for i in range(num_icebergs):
        cfg = iceberg_configs[i % len(iceberg_configs)]
        iceberg_id = f"IB00{i+1}" if i < 9 else f"IB{i+1}"

        curr_lat = cfg["start_lat"]
        curr_lon = cfg["start_lon"]

        for step in range(points_per_iceberg):
            ts = start_dt + pd.Timedelta(hours=step * FORECAST_INTERVAL_HOURS)

            # Add realistic ocean current drift + minor random walk noise
            lat_noise = np.random.normal(0, 0.003)
            lon_noise = np.random.normal(0, 0.005)

            lat = round(curr_lat + cfg["drift_lat"] + lat_noise, 4)
            lon = round(curr_lon + cfg["drift_lon"] + lon_noise, 4)

            # Validate generated values
            validate_observation(iceberg_id, ts, lat, lon)

            records.append({
                "iceberg_id": iceberg_id,
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "latitude": lat,
                "longitude": lon,
                "data_source": "SYNTHETIC_SAMPLE"
            })

            curr_lat = lat
            curr_lon = lon

    df = pd.DataFrame(records)

    if output_path is None:
        output_path = DATA_SAMPLE_DIR / "sample_iceberg_tracks.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def load_iceberg_tracks(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads, validates, and sorts iceberg tracking CSV data.

    Raises:
        ValueError: If file is missing required columns or invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Historical tracks CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as err:
        raise ValueError(f"Failed to parse CSV file at {path}: {err}")

    required_cols = {"iceberg_id", "timestamp", "latitude", "longitude"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    # Parse timestamps
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as err:
        raise ValueError(f"Failed to parse ISO timestamps in {path}: {err}")

    # Validate coordinates
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            ice_id, lat, lon = validate_observation(
                row["iceberg_id"], row["timestamp"], row["latitude"], row["longitude"]
            )
            valid_rows.append(idx)
        except ValueError:
            continue  # Drop invalid coordinate row

    df = df.loc[valid_rows].copy()

    if len(df) == 0:
        raise ValueError(f"No valid observations found in {path}.")

    # Drop duplicates & sort chronologically
    df.drop_duplicates(subset=["iceberg_id", "timestamp"], keep="last", inplace=True)
    df.sort_values(by=["iceberg_id", "timestamp"], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
