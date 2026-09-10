"""
Time-series sequence creation and sliding window dataset generation.
"""

from typing import List, Tuple, Dict, Any
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 5,
    max_allowed_gap_hours: float = 18.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Creates sliding window input sequences X and target positions Y for LSTM training.

    Args:
        df: Feature-engineered DataFrame sorted by iceberg_id and timestamp.
        feature_cols: List of column names used as input features.
        seq_length: Number of past timesteps in input sequence.
        max_allowed_gap_hours: Maximum allowable time gap between consecutive points.

    Returns:
        X: Array of shape (N, seq_length, num_features)
        Y: Array of shape (N, 2) containing target [latitude, longitude]
        metadata: List of dicts with iceberg_id and target timestamp for each sequence.
    """
    X_list = []
    Y_list = []
    metadata = []

    for ice_id, group in df.groupby("iceberg_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_points = len(group)

        if n_points <= seq_length:
            logger.warning(
                f"Iceberg '{ice_id}' has {n_points} observations, "
                f"which is <= seq_length {seq_length}. Skipping sequence generation."
            )
            continue

        feature_matrix = group[feature_cols].values
        target_matrix = group[["latitude", "longitude"]].values
        timestamps = group["timestamp"].values

        for i in range(n_points - seq_length):
            # Check for large time gaps within the sequence
            window_ts = timestamps[i : i + seq_length + 1]
            time_diffs_h = (
                pd.to_datetime(window_ts[1:]) - pd.to_datetime(window_ts[:-1])
            ).total_seconds() / 3600.0

            if np.any(time_diffs_h > max_allowed_gap_hours):
                # Skip window if there is an irregular large time gap
                continue

            x_seq = feature_matrix[i : i + seq_length]
            y_target = target_matrix[i + seq_length]
            target_ts = timestamps[i + seq_length]

            X_list.append(x_seq)
            Y_list.append(y_target)
            metadata.append({
                "iceberg_id": ice_id,
                "target_timestamp": target_ts
            })

    if len(X_list) == 0:
        raise ValueError(
            f"No valid sequences could be created with sequence_length={seq_length}. "
            "Check if dataset has enough observations per iceberg."
        )

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    return X, Y, metadata
