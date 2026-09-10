"""
PyTorch LSTM Sequence Model for Iceberg Drift Prediction.
"""

import torch
import torch.nn as nn


class IcebergLSTM(nn.Module):
    """
    LSTM-based sequence model for forecasting future iceberg positions (latitude, longitude).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2,
    ):
        super(IcebergLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Tensor of shape (batch_size, output_size)
        """
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
