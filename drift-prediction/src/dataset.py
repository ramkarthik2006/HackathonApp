"""
PyTorch Dataset wrapper for time-series iceberg drift sequences.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class IcebergDataset(Dataset):
    """
    PyTorch Dataset wrapping sequence features X and target positions Y.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X: Array of shape (N, seq_length, num_features)
            Y: Array of shape (N, 2)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]
