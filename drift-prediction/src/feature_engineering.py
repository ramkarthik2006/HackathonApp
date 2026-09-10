"""
Feature engineering for time-series iceberg drift modeling.
Calculates delta positions and velocities across consecutive timesteps.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes delta_lat, delta_lon, velocity_lat, and velocity_lon for each iceberg track.

    Args:
        df: DataFrame sorted by iceberg_id and timestamp.

    Returns:
        DataFrame with engineered feature columns added.
    """
    df = df.copy()

    # Calculate differences per iceberg group
    df["delta_lat"] = df.groupby("iceberg_id")["latitude"].diff().fillna(0.0)
    df["delta_lon"] = df.groupby("iceberg_id")["longitude"].diff().fillna(0.0)

    # Time delta in hours
    time_diff_s = df.groupby("iceberg_id")["timestamp"].diff().dt.total_seconds()
    time_delta_h = (time_diff_s / 3600.0).fillna(6.0)

    # Replace 0 or negative time deltas with default 6.0 hours to avoid div by 0
    time_delta_h = time_delta_h.apply(lambda h: h if h > 0.01 else 6.0)

    df["velocity_lat"] = df["delta_lat"] / time_delta_h
    df["velocity_lon"] = df["delta_lon"] / time_delta_h

    # Ensure no NaN or infinite values remain
    for col in ["delta_lat", "delta_lon", "velocity_lat", "velocity_lon"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
