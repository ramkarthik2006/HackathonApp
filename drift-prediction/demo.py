"""
Demo script for Iceberg Drift Prediction AI (Member 3).
Executes synthetic data loading, feature engineering, PyTorch LSTM training,
multi-step trajectory inference, uncertainty estimation, JSON export, and visualization.
"""

import sys
import json
from pathlib import Path

# Ensure src package is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import generate_synthetic_iceberg_tracks, load_iceberg_tracks
from src.train import train_model
from src.inference import predict_drift
from src.utils import plot_iceberg_trajectory, get_device
from src.config import (
    DATA_SAMPLE_DIR,
    PREDICTIONS_DIR,
    TRAJECTORIES_DIR,
    SEQUENCE_LENGTH,
)


def main():
    base_dir = Path(__file__).parent
    sample_csv = DATA_SAMPLE_DIR / "sample_iceberg_tracks.csv"
    predictions_json = PREDICTIONS_DIR / "drift_predictions.json"

    print("=" * 60)
    print("   AI-ENABLED ANTARCTIC ICEBERG DRIFT PREDICTION AI")
    print("=" * 60)

    # 1. Generate/Load Synthetic Historical Tracks
    print("\n1. Loading / Generating Synthetic Historical Iceberg Tracks...")
    df_raw = generate_synthetic_iceberg_tracks(
        num_icebergs=3, points_per_iceberg=50, output_path=sample_csv
    )
    df_tracks = load_iceberg_tracks(sample_csv)
    print(f"Loaded {len(df_tracks)} total observations for icebergs: {df_tracks['iceberg_id'].unique().tolist()}")

    # 2. Train LSTM Model
    print("\n2. Training PyTorch LSTM Sequence Model...")
    device = get_device()
    metrics = train_model(
        csv_path=sample_csv,
        epochs=15,
        batch_size=16,
        device=device,
    )

    print("\nTraining Metrics Summary:")
    print(f"  Best Validation Loss (MSE): {metrics['best_val_mse']:.6f}")
    print(f"  Validation Mean Geographic Error: {metrics['mean_geographic_error_km']:.2f} km")

    # 3. Run Inference on Iceberg IB001
    print("\n3. Running Multi-Step Trajectory Inference for Iceberg IB001...")
    ib001_history = df_tracks[df_tracks["iceberg_id"] == "IB001"].copy()

    forecast_result = predict_drift(
        iceberg_history=ib001_history,
        output_path=predictions_json,
        device=device,
    )

    # 4. Generate Trajectory Visualization
    print("\n4. Generating Trajectory Plot & Confidence Cone...")
    plot_path = TRAJECTORIES_DIR / "iceberg_trajectory_IB001.png"
    plot_iceberg_trajectory(ib001_history, forecast_result, output_path=plot_path)

    # 5. Summary Report
    last_obs = ib001_history.iloc[-1]
    forecast_steps = forecast_result["forecast"]

    print("\n" + "=" * 60)
    print("                 ICEBERG DRIFT PREDICTION AI SUMMARY")
    print("=" * 60)
    print(f"Model:                    PyTorch LSTM Sequence Model")
    print(f"Sequence length:          {SEQUENCE_LENGTH} observations")
    print(f"Forecast Horizons:        24h / 48h / 72h")
    print(f"Device:                   {device.type.upper()}")
    print(f"Target Iceberg:           IB001")
    print(f"Current Position:         Lat: {last_obs['latitude']:.4f}, Lon: {last_obs['longitude']:.4f}")
    print(f"Current Timestamp:        {last_obs['timestamp']}")
    print("-" * 60)

    for step in forecast_steps:
        print(f"  {step['hours']}h Forecast:              Lat: {step['latitude']:.4f}, Lon: {step['longitude']:.4f}  (Uncertainty: +/-{step['uncertainty_km']:.1f} km)")

    print("=" * 60)
    print("PREDICTION COMPLETE SUCCESSFUL")
    print(f"Output JSON Saved To:     {predictions_json}")
    print(f"Trajectory Map Saved To:   {plot_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
