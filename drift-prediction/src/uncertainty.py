"""
Empirical uncertainty estimation for multi-horizon iceberg drift forecasts.
"""

from typing import Dict, List, Any
import numpy as np


class EmpiricalUncertaintyEstimator:
    """
    Estimates geographic uncertainty radii (in kilometers) for forecast horizons
    based on empirical validation set prediction error statistics.
    """

    def __init__(self, validation_error_stats: Dict[int, float] = None):
        """
        Args:
            validation_error_stats: Dict mapping horizon_hours (24, 48, 72)
                                    to mean geographic error in km.
        """
        self.stats = validation_error_stats or {}

    def get_uncertainty_km(self, horizon_hours: int) -> float:
        """
        Returns estimated uncertainty radius in kilometers for a given forecast horizon.
        """
        if horizon_hours in self.stats:
            return round(float(self.stats[horizon_hours]), 2)

        # Baseline empirical uncertainty bounds scaling with horizon hours
        # Uncertainty grows over time due to compounding drift forces
        base_error = 2.0  # initial error floor in km
        growth_factor = 0.25  # km per hour forecast horizon
        horizon_penalty = 0.001 * (horizon_hours ** 2)

        uncertainty = base_error + (horizon_hours * growth_factor) + horizon_penalty
        return round(float(uncertainty), 2)

    def calculate_confidence_cone(
        self, forecasts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attaches uncertainty radii to forecast points to construct a confidence cone.
        """
        cone = []
        for point in forecasts:
            hours = int(point["hours"])
            uncertainty_km = self.get_uncertainty_km(hours)

            entry = dict(point)
            entry["uncertainty_km"] = uncertainty_km
            cone.append(entry)

        return cone
