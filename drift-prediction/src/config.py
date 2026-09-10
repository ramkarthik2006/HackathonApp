"""
Configuration settings for Iceberg Drift Prediction AI.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "iceberg_lstm.pth"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TRAJECTORIES_DIR = OUTPUTS_DIR / "trajectories"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Model Parameters
SEQUENCE_LENGTH = 5
FORECAST_INTERVAL_HOURS = 6
FORECAST_HORIZONS = [24, 48, 72]  # in hours

# Features
INPUT_FEATURES = [
    "latitude",
    "longitude",
    "delta_lat",
    "delta_lon",
    "velocity_lat",
    "velocity_lon",
]
NUM_FEATURES = len(INPUT_FEATURES)
OUTPUT_DIM = 2  # latitude, longitude

# Training Hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 16
VAL_SPLIT_RATIO = 0.2
