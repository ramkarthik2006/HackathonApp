# Iceberg Drift Prediction AI

**Project:** AI-Enabled Antarctic Sea-Ice & Navigation Decision Support  
**Module:** Member 3 — Iceberg Drift Prediction AI  
**Git Branch:** `drift-prediction`

---

## Objective

The objective of this module is to forecast the future movement of detected icebergs across 24-hour, 48-hour, and 72-hour forecast horizons in Antarctic maritime regions. Predicting iceberg trajectories allows vessel operators and risk assessment systems to proactively evaluate future sea-ice hazards and avoid collision risks.

This module implements a **PyTorch LSTM sequence model** trained on historical time-series iceberg tracks to perform multi-step autoregressive trajectory forecasting and estimate empirical prediction uncertainty (confidence cones).

---

## Module Responsibility

Member 3 is strictly responsible for:
1. Ingesting historical iceberg tracking data and current iceberg detections from Member 2.
2. Validating, cleaning, and chronologically sorting time-series observations.
3. Feature engineering ($\Delta \text{Latitude}, \Delta \text{Longitude}, \text{Velocity}_{\text{lat}}, \text{Velocity}_{\text{lon}}$).
4. Generating sliding window time-series input sequences.
5. Training an LSTM neural network on CPU/CUDA.
6. Generating 24h, 48h, and 72h future position predictions ($\text{Latitude}, \text{Longitude}$).
7. Calculating empirical uncertainty radii ($\text{uncertainty\_km}$) to model prediction error confidence cones.
8. Exporting standardized JSON predictions for Member 4's Risk & Hazard Engine.
9. Generating visual trajectory plots with confidence cones.

> **Scope Isolation:** Member 3 does NOT implement satellite imagery processing, image segmentation, iceberg detection, risk hazard mapping, A* pathfinding, or web dashboard user interfaces.

---

## System Position

```
Member 2 (Iceberg Detection)
          │
          ▼
   Iceberg Observations (ID, Timestamp, Lat, Lon)
          │
          ▼
Member 3 (Iceberg Drift Prediction AI)  <-- YOU ARE HERE
          │
          ▼
   drift_predictions.json (24h/48h/72h Forecast + Uncertainty)
          │
          ▼
Member 4 (Risk & Hazard Engine)
```

---

## Input

Member 3 consumes detection outputs from Member 2 or historical tracking CSVs:

```json
{
  "image_id": "IMG001",
  "timestamp": "2026-09-10T12:00:00Z",
  "icebergs": [
    {
      "id": "IB001",
      "latitude": -64.21,
      "longitude": 52.71,
      "area_pixels": 2450,
      "confidence": 0.84
    }
  ]
}
```

---

## Historical Dataset Format

CSV dataset (`data/raw/iceberg_tracks.csv` or `data/sample/sample_iceberg_tracks.csv`):

```csv
iceberg_id,timestamp,latitude,longitude
IB001,2026-09-09T00:00:00Z,-64.10,52.10
IB001,2026-09-09T06:00:00Z,-64.12,52.15
IB001,2026-09-09T12:00:00Z,-64.14,52.20
IB001,2026-09-09T18:00:00Z,-64.17,52.27
IB001,2026-09-10T00:00:00Z,-64.20,52.34
```

---

## Preprocessing & Sequence Generation

1. **Validation:** Checks $-90 \le \text{Lat} \le 90$ and $-180 \le \text{Lon} \le 180$. Filters corrupt rows.
2. **Chronological Sorting:** Sorts by `[iceberg_id, timestamp]`.
3. **Sliding Window:** Constructs input sequences $X$ of size `(batch_size, sequence_length, num_features)` where `SEQUENCE_LENGTH = 5` past observations, and targets $Y$ of size `(batch_size, 2)` ($\text{Latitude}, \text{Longitude}$).

---

## Feature Engineering

For each consecutive observation step:
* $\Delta \text{Latitude} = \text{Lat}_t - \text{Lat}_{t-1}$
* $\Delta \text{Longitude} = \text{Lon}_t - \text{Lon}_{t-1}$
* $\text{Velocity}_{\text{lat}} = \frac{\Delta \text{Latitude}}{\Delta \text{Hours}}$
* $\text{Velocity}_{\text{lon}} = \frac{\Delta \text{Longitude}}{\Delta \text{Hours}}$

**Input Feature Vector (6 features):**
`["latitude", "longitude", "delta_lat", "delta_lon", "velocity_lat", "velocity_lon"]`

---

## LSTM Architecture

Built in PyTorch ([`src/model.py`](file:///C:/Users/Ram/OneDrive/Documents/HackathonApp/drift-prediction/src/model.py)):
* **Input Layer:** 6 features $\times$ 5 timesteps
* **LSTM Layers:** 2 stacked LSTM layers with `hidden_size=64`, `dropout=0.2`
* **Dense Layer 1:** Linear (64 $\to$ 32) + ReLU
* **Output Layer:** Linear (32 $\to$ 2) producing predicted $[\text{Latitude}, \text{Longitude}]$

---

## Training & Validation

* **Train/Val Split:** Time-aware split (first 80% chronologically for training, last 20% for validation per iceberg).
* **Normalization:** `StandardScaler` fitted **ONLY** on training set features to avoid data leakage. Saved to `models/scaler.pkl`.
* **Loss Function:** Mean Squared Error (MSE Loss).
* **Optimizer:** Adam ($\text{lr} = 0.001$).
* **Checkpointing:** Saves best model weights to `models/iceberg_lstm.pth`.

---

## Evaluation Metrics

Metrics calculated on validation set:
* **MAE (Latitude / Longitude):** Degrees
* **RMSE (Latitude / Longitude):** Degrees
* **Mean Geographic Error (km):** Calculated using **Haversine Distance Formula**
* Output written to `outputs/metrics/metrics.json`

---

## Multi-Step Forecasting & Horizons

Uses **Autoregressive Multi-Step Inference**:
To forecast $24\text{h}$ (4 steps of 6h), $48\text{h}$ (8 steps), $72\text{h}$ (12 steps):
1. Model predicts step $t+1$.
2. Prediction is unscaled and appended to input sequence.
3. Feature deltas/velocities are recomputed.
4. Window slides forward to predict step $t+2$, repeating up to 72 hours.

---

## Uncertainty Estimation & Confidence Cone

* Calculates empirical geographic uncertainty radius $\text{uncertainty\_km}$ for each forecast horizon ($24h, 48h, 72h$) based on validation set error variance.
* Uncertainty expands over time as forecast horizon grows.
* Formats predictions as confidence cones consumable by Member 6's dashboard visualization.

---

## Output JSON (Data Contract for Member 4)

Saved to `outputs/predictions/drift_predictions.json`:

```json
{
  "iceberg_id": "IB001",
  "prediction_timestamp": "2026-09-10T12:00:00Z",
  "forecast_interval_hours": 6,
  "forecast": [
    {
      "hours": 24,
      "timestamp": "2026-09-11T12:00:00Z",
      "latitude": -64.30,
      "longitude": 52.90,
      "uncertainty_km": 8.6
    },
    {
      "hours": 48,
      "timestamp": "2026-09-12T12:00:00Z",
      "latitude": -64.42,
      "longitude": 53.12,
      "uncertainty_km": 16.3
    },
    {
      "hours": 72,
      "timestamp": "2026-09-13T12:00:00Z",
      "latitude": -64.55,
      "longitude": 53.40,
      "uncertainty_km": 25.2
    }
  ]
}
```

---

## Visualization

Saved to `outputs/trajectories/iceberg_trajectory_IB001.png`:
* Historical iceberg track (royalblue line)
* Current position marker (crimson star)
* Predicted 24h, 48h, 72h path (darkgreen square line)
* Shaded confidence circles showing empirical uncertainty radii in km.

---

## Synthetic Dataset

Synthetic iceberg tracking dataset generated in `data/sample/sample_iceberg_tracks.csv` containing trajectories for icebergs `IB001`, `IB002`, `IB003`.

> **Disclaimer:** All sample data is synthetic demo data created for offline testing. It is not real satellite measurement data.

---

## Installation

```bash
cd drift-prediction
pip install -r requirements.txt
```

---

## Running the Demo

```bash
python demo.py
```

Or using full path:
```powershell
& "C:\Users\Ram\AppData\Local\Programs\Python\Python312\python.exe" drift-prediction/demo.py
```

---

## Training Command

To train the LSTM model on custom track CSV data:
```python
from drift_prediction.src import train_model

metrics = train_model(csv_path="data/raw/iceberg_tracks.csv", epochs=20)
```

---

## Inference Command

To run drift forecasting on an iceberg history:
```python
from drift_prediction.src import predict_drift

forecast_result = predict_drift("data/sample/sample_iceberg_tracks.csv")
```

---

## Testing

Run the full pytest suite:
```bash
pytest tests/ -v
```

---

## Integration with Member 2

Member 2 provides current iceberg detections. Member 3 appends these detections to historical track data (`iceberg_id`, `timestamp`, `latitude`, `longitude`) to maintain continuous tracking sequences.

---

## Integration with Member 4

Member 4 loads `outputs/predictions/drift_predictions.json` to extract predicted future iceberg coordinates and uncertainty radii, integrating them into the Risk & Hazard Grid.

---

## Limitations

* **Simplified Physics:** Does not explicitly solve hydrodynamic Navier-Stokes equations for ocean current or wind shear force vectors.
* **Interval Regularity:** Assumes relatively uniform sampling intervals (6 hours).

---

## Future Improvements

* Incorporate external ocean current velocity ($u, v$) and wind velocity vectors from atmospheric reanalysis models.
* Implement Transformer / Temporal Fusion Transformer (TFT) architectures for longer-term multi-horizon forecasting.
