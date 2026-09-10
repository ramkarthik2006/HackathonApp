"""
Training pipeline for PyTorch LSTM Iceberg Drift Prediction model.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .config import (
    MODEL_PATH,
    SCALER_PATH,
    METRICS_DIR,
    DATA_SAMPLE_DIR,
    SEQUENCE_LENGTH,
    INPUT_FEATURES,
    NUM_FEATURES,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VAL_SPLIT_RATIO,
)
from .utils import get_device, haversine_distance_km, ensure_directories
from .data_loader import load_iceberg_tracks, generate_synthetic_iceberg_tracks
from .feature_engineering import add_engineered_features
from .preprocessing import create_sequences
from .dataset import IcebergDataset
from .model import IcebergLSTM


def train_model(
    csv_path: Optional[Union[str, Path]] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes full training pipeline:
      1. Loads & validates track data
      2. Computes features & sequences
      3. Performs time-aware train/val split
      4. Fits StandardScaler on training data only
      5. Trains IcebergLSTM model
      6. Evaluates MAE/RMSE metrics
      7. Saves model checkpoint and scaler
    """
    ensure_directories()
    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # 1. Load data
    if csv_path is None or not Path(csv_path).exists():
        csv_path = DATA_SAMPLE_DIR / "sample_iceberg_tracks.csv"
        if not csv_path.exists():
            print("No existing track data found. Generating synthetic track dataset...")
            generate_synthetic_iceberg_tracks(output_path=csv_path)

    print(f"Loading iceberg tracks from: {csv_path}")
    df_raw = load_iceberg_tracks(csv_path)

    # 2. Feature engineering
    df_feat = add_engineered_features(df_raw)

    # 3. Create sequences
    X_raw, Y_raw, meta = create_sequences(
        df_feat, feature_cols=INPUT_FEATURES, seq_length=SEQUENCE_LENGTH
    )
    n_samples = len(X_raw)

    # 4. Time-aware train/val split (80% early data for train, 20% late for val)
    split_idx = int(n_samples * (1.0 - VAL_SPLIT_RATIO))
    if split_idx < 1 or (n_samples - split_idx) < 1:
        # Fallback if dataset is very small
        split_idx = max(1, n_samples - 1)

    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    Y_train_raw, Y_val_raw = Y_raw[:split_idx], Y_raw[split_idx:]

    # 5. Fit Scalers on training set ONLY to prevent data leakage
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Reshape 3D X to 2D for feature_scaler fitting
    n_train, seq_len, n_feat = X_train_raw.shape
    X_train_reshaped = X_train_raw.reshape(-1, n_feat)
    feature_scaler.fit(X_train_reshaped)

    target_scaler.fit(Y_train_raw)

    # Transform features
    X_train_scaled = feature_scaler.transform(X_train_reshaped).reshape(
        n_train, seq_len, n_feat
    )

    n_val = len(X_val_raw)
    if n_val > 0:
        X_val_scaled = feature_scaler.transform(X_val_raw.reshape(-1, n_feat)).reshape(
            n_val, seq_len, n_feat
        )
        Y_val_scaled = target_scaler.transform(Y_val_raw)
    else:
        X_val_scaled = X_train_scaled
        Y_val_scaled = target_scaler.transform(Y_train_raw)

    Y_train_scaled = target_scaler.transform(Y_train_raw)

    # 6. Create PyTorch datasets and loaders
    train_dataset = IcebergDataset(X_train_scaled, Y_train_scaled)
    val_dataset = IcebergDataset(X_val_scaled, Y_val_scaled)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 7. Build Model, Criterion, Optimizer
    model = IcebergLSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=2,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    # 8. Training loop
    print(f"Starting LSTM training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save best checkpoint
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] - "
                f"Train Loss (MSE): {epoch_train_loss:.6f} | "
                f"Val Loss (MSE): {epoch_val_loss:.6f}"
            )

    # 9. Evaluate on validation set in physical units (degrees and km)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    val_preds_scaled_list = []
    val_targets_scaled_list = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            val_preds_scaled_list.append(preds.cpu().numpy())
            val_targets_scaled_list.append(batch_y.numpy())

    val_preds_scaled = np.vstack(val_preds_scaled_list)
    val_targets_scaled = np.vstack(val_targets_scaled_list)

    # Inverse transform to original lat/lon
    val_preds = target_scaler.inverse_transform(val_preds_scaled)
    val_targets = target_scaler.inverse_transform(val_targets_scaled)

    # Calculate geographic metrics
    lat_errors = np.abs(val_preds[:, 0] - val_targets[:, 0])
    lon_errors = np.abs(val_preds[:, 1] - val_targets[:, 1])

    mae_lat = float(np.mean(lat_errors))
    mae_lon = float(np.mean(lon_errors))
    rmse_lat = float(np.sqrt(np.mean(lat_errors ** 2)))
    rmse_lon = float(np.sqrt(np.mean(lon_errors ** 2)))

    geo_errors_km = [
        haversine_distance_km(p[0], p[1], t[0], t[1])
        for p, t in zip(val_preds, val_targets)
    ]
    mean_geo_error_km = float(np.mean(geo_errors_km))
    rmse_geo_error_km = float(np.sqrt(np.mean(np.array(geo_errors_km) ** 2)))

    metrics = {
        "model": "IcebergLSTM",
        "epochs": epochs,
        "sequence_length": SEQUENCE_LENGTH,
        "train_samples": n_train,
        "val_samples": n_val,
        "final_train_mse": train_losses[-1],
        "final_val_mse": val_losses[-1],
        "best_val_mse": best_val_loss,
        "mae_latitude_deg": round(mae_lat, 4),
        "mae_longitude_deg": round(mae_lon, 4),
        "rmse_latitude_deg": round(rmse_lat, 4),
        "rmse_longitude_deg": round(rmse_lon, 4),
        "mean_geographic_error_km": round(mean_geo_error_km, 2),
        "rmse_geographic_error_km": round(rmse_geo_error_km, 2),
    }

    # 10. Save Scalers
    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "input_features": INPUT_FEATURES,
    }
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scalers, f)

    # 11. Save Metrics JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scalers saved to: {SCALER_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Validation Mean Geographic Error: {mean_geo_error_km:.2f} km")

    return metrics
