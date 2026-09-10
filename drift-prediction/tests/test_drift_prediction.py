"""
Comprehensive test suite for Iceberg Drift Prediction AI (Member 3).
"""

import json
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Ensure src package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import validate_observation, haversine_distance_km, plot_iceberg_trajectory
from src.data_loader import generate_synthetic_iceberg_tracks, load_iceberg_tracks
from src.feature_engineering import add_engineered_features
from src.preprocessing import create_sequences
from src.model import IcebergLSTM
from src.train import train_model
from src.inference import predict_drift
from src.config import INPUT_FEATURES, SEQUENCE_LENGTH, DATA_SAMPLE_DIR, OUTPUTS_DIR


@pytest.fixture
def sample_csv_path(tmp_path):
    """Fixture providing temporary synthetic track dataset."""
    csv_file = tmp_path / "sample_tracks.csv"
    generate_synthetic_iceberg_tracks(num_icebergs=2, points_per_iceberg=20, output_path=csv_file)
    return csv_file


def test_1_historical_dataset_loads(sample_csv_path):
    """Test 1: Historical dataset loads properly."""
    df = load_iceberg_tracks(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_2_coordinates_validated():
    """Test 2: Valid coordinates pass, out-of-range coordinates fail."""
    id_val, lat, lon = validate_observation("IB001", "2026-09-10T12:00:00Z", -64.5, 52.3)
    assert id_val == "IB001"
    assert lat == -64.5
    assert lon == 52.3

    with pytest.raises(ValueError):
        validate_observation("IB001", "2026-09-10T12:00:00Z", -95.0, 52.3)

    with pytest.raises(ValueError):
        validate_observation("IB001", "2026-09-10T12:00:00Z", -64.5, 195.0)


def test_3_timestamps_parsed(sample_csv_path):
    """Test 3: Timestamps parsed as pandas datetime objects."""
    df = load_iceberg_tracks(sample_csv_path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_4_sorted_chronologically(sample_csv_path):
    """Test 4: Track data is sorted by iceberg_id and timestamp."""
    df = load_iceberg_tracks(sample_csv_path)
    for _, group in df.groupby("iceberg_id"):
        assert group["timestamp"].is_monotonic_increasing


def test_5_and_6_sequence_generation(sample_csv_path):
    """Test 5 & 6: Sequences generated with correct dimensions and lengths."""
    df = load_iceberg_tracks(sample_csv_path)
    df_feat = add_engineered_features(df)
    X, Y, meta = create_sequences(df_feat, feature_cols=INPUT_FEATURES, seq_length=SEQUENCE_LENGTH)

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.ndim == 3
    assert X.shape[1] == SEQUENCE_LENGTH
    assert X.shape[2] == len(INPUT_FEATURES)
    assert Y.ndim == 2
    assert Y.shape[1] == 2
    assert len(X) == len(Y) == len(meta)


def test_7_and_8_lstm_initialization_and_shape():
    """Test 7 & 8: PyTorch LSTM initializes and produces expected output tensor shape."""
    model = IcebergLSTM(input_size=6, hidden_size=32, num_layers=1, output_size=2)
    dummy_input = torch.randn(8, SEQUENCE_LENGTH, 6)  # batch_size = 8
    output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (8, 2)


def test_9_training_pipeline_runs(sample_csv_path, tmp_path):
    """Test 9: Training executes for a small number of epochs."""
    metrics = train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    assert isinstance(metrics, dict)
    assert "final_train_mse" in metrics
    assert "mean_geographic_error_km" in metrics


def test_10_through_13_inference_multi_horizon(sample_csv_path, tmp_path):
    """Test 10 - 13: Inference runs and produces 24h, 48h, 72h forecasts."""
    train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    df = load_iceberg_tracks(sample_csv_path)
    ib_hist = df[df["iceberg_id"] == "IB001"].copy()

    out_file = tmp_path / "drift_pred.json"
    pred_res = predict_drift(ib_hist, output_path=out_file)

    assert isinstance(pred_res, dict)
    assert pred_res["iceberg_id"] == "IB001"
    forecasts = pred_res["forecast"]
    assert len(forecasts) == 3

    horizons = [f["hours"] for f in forecasts]
    assert 24 in horizons
    assert 48 in horizons
    assert 72 in horizons


def test_14_and_15_output_coordinates_valid(sample_csv_path, tmp_path):
    """Test 14 & 15: Predicted latitudes and longitudes are within valid geographic bounds."""
    train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    df = load_iceberg_tracks(sample_csv_path)
    ib_hist = df[df["iceberg_id"] == "IB001"].copy()

    pred_res = predict_drift(ib_hist, output_path=tmp_path / "pred.json")
    for step in pred_res["forecast"]:
        assert -90.0 <= step["latitude"] <= 90.0
        assert -180.0 <= step["longitude"] <= 180.0


def test_16_json_output_fields(sample_csv_path, tmp_path):
    """Test 16: Saved JSON predictions file contains all required contract fields."""
    train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    df = load_iceberg_tracks(sample_csv_path)
    ib_hist = df[df["iceberg_id"] == "IB001"].copy()

    out_file = tmp_path / "contract_test.json"
    predict_drift(ib_hist, output_path=out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "iceberg_id" in data
    assert "prediction_timestamp" in data
    assert "forecast_interval_hours" in data
    assert "forecast" in data
    assert len(data["forecast"]) == 3


def test_17_uncertainty_non_negative(sample_csv_path, tmp_path):
    """Test 17: Estimated uncertainty values are non-negative and grow over horizon."""
    train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    df = load_iceberg_tracks(sample_csv_path)
    ib_hist = df[df["iceberg_id"] == "IB001"].copy()

    pred_res = predict_drift(ib_hist, output_path=tmp_path / "unc.json")
    forecasts = pred_res["forecast"]

    unc_values = [f["uncertainty_km"] for f in forecasts]
    for unc in unc_values:
        assert unc >= 0.0

    # Verify uncertainty increases over time
    assert unc_values[0] <= unc_values[1] <= unc_values[2]


def test_18_visualization_generation(sample_csv_path, tmp_path):
    """Test 18: Trajectory plot visualization image is successfully generated."""
    train_model(csv_path=sample_csv_path, epochs=2, batch_size=4)
    df = load_iceberg_tracks(sample_csv_path)
    ib_hist = df[df["iceberg_id"] == "IB001"].copy()

    pred_res = predict_drift(ib_hist, output_path=tmp_path / "pred.json")
    plot_file = tmp_path / "test_traj.png"

    res_path = plot_iceberg_trajectory(ib_hist, pred_res, output_path=plot_file)
    assert res_path.exists()
    assert res_path.stat().st_size > 0
