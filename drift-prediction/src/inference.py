"""
Inference pipeline and autoregressive trajectory forecasting for Iceberg Drift Prediction.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import torch

from .config import (
    MODEL_PATH,
    SCALER_PATH,
    PREDICTIONS_DIR,
    SEQUENCE_LENGTH,
    INPUT_FEATURES,
    NUM_FEATURES,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    FORECAST_INTERVAL_HOURS,
    FORECAST_HORIZONS,
)
from .utils import get_device, validate_observation, ensure_directories
from .feature_engineering import add_engineered_features
from .model import IcebergLSTM
from .uncertainty import EmpiricalUncertaintyEstimator


def predict_drift(
    iceberg_history: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None,
    forecast_horizons: List[int] = FORECAST_HORIZONS,
    forecast_interval_hours: int = FORECAST_INTERVAL_HOURS,
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generates 24-hour, 48-hour, and 72-hour drift trajectory forecasts.

    Args:
        iceberg_history: Historical observations list, DataFrame, or CSV path.
        model_path: Path to trained PyTorch model checkpoint.
        scaler_path: Path to fitted scalers file.
        forecast_horizons: Forecast horizons in hours (default [24, 48, 72]).
        forecast_interval_hours: Step interval in hours (default 6).
        output_path: Path to save predictions JSON.
        device: PyTorch device.

    Returns:
        Structured JSON dictionary containing forecast steps and uncertainty radii.
    """
    ensure_directories()

    if device is None:
        device = get_device()

    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)

    if scaler_path is None:
        scaler_path = SCALER_PATH
    scaler_path = Path(scaler_path)

    # 1. Load history
    if isinstance(iceberg_history, (str, Path)):
        hist_df = pd.read_csv(iceberg_history)
    elif isinstance(iceberg_history, list):
        hist_df = pd.DataFrame(iceberg_history)
    elif isinstance(iceberg_history, pd.DataFrame):
        hist_df = iceberg_history.copy()
    else:
        raise ValueError("Invalid iceberg_history type provided.")

    required_cols = {"iceberg_id", "timestamp", "latitude", "longitude"}
    if not required_cols.issubset(set(hist_df.columns)):
        missing = required_cols - set(hist_df.columns)
        raise ValueError(f"History data missing required columns: {missing}")

    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df.sort_values("timestamp", ascending=True, inplace=True)
    hist_df.reset_index(drop=True, inplace=True)

    iceberg_id = str(hist_df["iceberg_id"].iloc[0])
    n_obs = len(hist_df)

    if n_obs < SEQUENCE_LENGTH:
        raise ValueError(
            f"Insufficient historical observations for iceberg {iceberg_id}. "
            f"At least {SEQUENCE_LENGTH} observations are required, but got {n_obs}."
        )

    # 2. Check model and scaler files exist
    if not model_path.exists():
        raise ValueError(f"Model checkpoint not found at {model_path}. Train model first.")
    if not scaler_path.exists():
        raise ValueError(f"Scaler file not found at {scaler_path}. Train model first.")

    # 3. Load scalers
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]

    # 4. Load model
    model = IcebergLSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=2,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 5. Feature engineering on history
    hist_feat_df = add_engineered_features(hist_df)

    # Extract last SEQUENCE_LENGTH window
    curr_window_df = hist_feat_df.tail(SEQUENCE_LENGTH).copy().reset_index(drop=True)

    # 6. Autoregressive Multi-step forecasting
    max_horizon_hours = max(forecast_horizons)
    total_steps = int(np.ceil(max_horizon_hours / forecast_interval_hours))

    forecast_records = []
    current_ts = curr_window_df["timestamp"].iloc[-1]

    # Maintain sequence feature window
    window_features = curr_window_df[INPUT_FEATURES].values.copy()

    estimator = EmpiricalUncertaintyEstimator()

    with torch.no_grad():
        for step_idx in range(1, total_steps + 1):
            curr_step_hours = step_idx * forecast_interval_hours
            step_ts = current_ts + pd.Timedelta(hours=curr_step_hours)

            # Scale input window
            window_scaled = feature_scaler.transform(window_features).reshape(
                1, SEQUENCE_LENGTH, NUM_FEATURES
            )
            input_tensor = torch.tensor(window_scaled, dtype=torch.float32).to(device)

            # Predict next step
            pred_scaled = model(input_tensor).cpu().numpy()
            pred_unscaled = target_scaler.inverse_transform(pred_scaled)[0]

            pred_lat = round(float(pred_unscaled[0]), 4)
            pred_lon = round(float(pred_unscaled[1]), 4)

            # Check if this step matches a target horizon (24h, 48h, 72h)
            if curr_step_hours in forecast_horizons:
                unc_km = estimator.get_uncertainty_km(curr_step_hours)
                forecast_records.append({
                    "hours": curr_step_hours,
                    "timestamp": step_ts.isoformat().replace("+00:00", "Z"),
                    "latitude": pred_lat,
                    "longitude": pred_lon,
                    "uncertainty_km": unc_km
                })

            # Update feature window for autoregressive next step prediction
            prev_lat = window_features[-1, 0]
            prev_lon = window_features[-1, 1]

            delta_lat = pred_lat - prev_lat
            delta_lon = pred_lon - prev_lon
            vel_lat = delta_lat / forecast_interval_hours
            vel_lon = delta_lon / forecast_interval_hours

            next_feat_vector = np.array([
                pred_lat, pred_lon, delta_lat, delta_lon, vel_lat, vel_lon
            ], dtype=np.float32)

            # Slide window
            window_features = np.vstack([window_features[1:], next_feat_vector])

    last_obs_ts = current_ts.isoformat().replace("+00:00", "Z")

    result = {
        "iceberg_id": iceberg_id,
        "prediction_timestamp": last_obs_ts,
        "forecast_interval_hours": forecast_interval_hours,
        "forecast": forecast_records
    }

    # Save to outputs/predictions/drift_predictions.json
    if output_path is None:
        output_path = PREDICTIONS_DIR / "drift_predictions.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
