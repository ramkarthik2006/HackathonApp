"""
Utility functions for coordinate validation, Haversine distance, device configuration, and trajectory visualization.
"""

import math
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    TRAJECTORIES_DIR,
    METRICS_DIR,
)


def get_device() -> torch.device:
    """Returns PyTorch device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculates geographic distance in kilometers between two lat/lon points using Haversine formula.
    """
    R = 6371.0  # Earth's radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return float(R * c)


def validate_observation(
    iceberg_id: Any,
    timestamp: Any,
    latitude: float,
    longitude: float
) -> Tuple[str, float, float]:
    """
    Validates observation values.

    Raises:
        ValueError: If coordinates or parameters are invalid.
    """
    if not iceberg_id or not str(iceberg_id).strip():
        raise ValueError("Iceberg ID cannot be empty.")

    try:
        lat = float(latitude)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid latitude value: '{latitude}'")

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} is out of valid range [-90, 90].")

    try:
        lon = float(longitude)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid longitude value: '{longitude}'")

    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude {lon} is out of valid range [-180, 180].")

    return str(iceberg_id).strip(), lat, lon


def ensure_directories() -> None:
    """Creates all required directory structures if they do not exist."""
    for d in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SAMPLE_DIR,
        MODEL_DIR,
        PREDICTIONS_DIR,
        TRAJECTORIES_DIR,
        METRICS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def plot_iceberg_trajectory(
    history_df: pd.DataFrame,
    forecast_data: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generates trajectory visualization with historical track, current location, 24/48/72h predictions,
    and shaded uncertainty confidence circles.
    """
    ensure_directories()
    iceberg_id = forecast_data.get("iceberg_id", "IB001")

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Plot Historical Track
    hist_lats = history_df["latitude"].values
    hist_lons = history_df["longitude"].values

    ax.plot(
        hist_lons,
        hist_lats,
        "o--",
        color="royalblue",
        linewidth=2,
        markersize=5,
        label="Historical Track",
    )

    # 2. Plot Current Position (Last Historical Point)
    curr_lat, curr_lon = hist_lats[-1], hist_lons[-1]
    ax.plot(
        curr_lon,
        curr_lat,
        "*",
        color="crimson",
        markersize=14,
        label="Current Position",
    )

    # 3. Plot Forecast Points and Confidence Cones/Circles
    forecast_points = forecast_data.get("forecast", [])
    pred_lons = [p["longitude"] for p in forecast_points]
    pred_lats = [p["latitude"] for p in forecast_points]

    # Connect current position to forecast path
    all_pred_lons = [curr_lon] + pred_lons
    all_pred_lats = [curr_lat] + pred_lats

    ax.plot(
        all_pred_lons,
        all_pred_lats,
        "s-.",
        color="darkgreen",
        linewidth=2,
        markersize=8,
        label="Predicted Trajectory (24h/48h/72h)",
    )

    # Draw uncertainty radius circles (converted km to approx deg lat/lon)
    for p in forecast_points:
        lat, lon = p["latitude"], p["longitude"]
        unc_km = p["uncertainty_km"]
        hours = p["hours"]

        # Approximate 1 deg lat ≈ 111 km
        radius_deg = unc_km / 111.0

        circle = Circle(
            (lon, lat),
            radius_deg,
            facecolor="limegreen",
            alpha=0.18,
            linestyle="--",
            edgecolor="darkgreen",
        )
        ax.add_patch(circle)

        ax.annotate(
            f"{hours}h (+/-{unc_km:.1f}km)",
            (lon, lat),
            textcoords="offset points",
            xytext=(10, 10),
            ha="left",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
        )

    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(
        f"Iceberg Drift Trajectory & Confidence Cone - {iceberg_id}\n(SYNTHETIC SAMPLE DEMONSTRATION)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    if output_path is None:
        output_path = TRAJECTORIES_DIR / f"iceberg_trajectory_{iceberg_id}.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path
