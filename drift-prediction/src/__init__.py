"""
Iceberg Drift Prediction AI Package for Antarctic Navigation.
"""

from .train import train_model
from .inference import predict_drift
from .data_loader import load_iceberg_tracks, generate_synthetic_iceberg_tracks
from .model import IcebergLSTM
from .utils import haversine_distance_km, validate_observation, plot_iceberg_trajectory

__all__ = [
    "train_model",
    "predict_drift",
    "load_iceberg_tracks",
    "generate_synthetic_iceberg_tracks",
    "IcebergLSTM",
    "haversine_distance_km",
    "validate_observation",
    "plot_iceberg_trajectory",
]
