"""
GridGraph representation and neighbor resolution for hazard-aware navigation.
"""

import math
from typing import List, Tuple, Dict, Any


class GridGraph:
    """
    Represents a spatial hazard grid and provides graph traversal methods
    with 8-directional movement and hazard-weighted cost calculations.
    """

    # 8-direction movement vectors: (d_row, d_col, distance_cost)
    DIRECTIONS: List[Tuple[int, int, float]] = [
        (-1, 0, 1.0),                  # Up
        (1, 0, 1.0),                   # Down
        (0, -1, 1.0),                  # Left
        (0, 1, 1.0),                   # Right
        (-1, -1, math.sqrt(2.0)),      # Upper-left
        (-1, 1, math.sqrt(2.0)),       # Upper-right
        (1, -1, math.sqrt(2.0)),       # Lower-left
        (1, 1, math.sqrt(2.0)),        # Lower-right
    ]

    def __init__(
        self,
        grid: List[List[float]],
        risk_weight: float = 10.0,
        critical_threshold: float = 0.95
    ):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.risk_weight = max(0.0, float(risk_weight))
        self.critical_threshold = float(critical_threshold)

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Checks if cell (row, col) is within grid dimensions."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_blocked(self, row: int, col: int) -> bool:
        """Determines whether cell is blocked due to critical hazard score."""
        if not self.is_in_bounds(row, col):
            return True
        return self.grid[row][col] >= self.critical_threshold

    def get_cell_risk(self, row: int, col: int) -> float:
        """Returns risk score for given cell."""
        return self.grid[row][col]

    def get_neighbors(
        self, row: int, col: int
    ) -> List[Tuple[Tuple[int, int], float, float]]:
        """
        Returns valid non-blocked neighboring cells.

        Returns:
            List of tuples: [((n_row, n_col), total_step_cost, distance_cost), ...]
        """
        neighbors = []
        for dr, dc, dist_cost in self.DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self.is_in_bounds(nr, nc) and not self.is_blocked(nr, nc):
                dest_risk = self.grid[nr][nc]
                step_cost = dist_cost + (self.risk_weight * dest_risk)
                neighbors.append(((nr, nc), step_cost, dist_cost))
        return neighbors

    def heuristic(self, node_a: Tuple[int, int], node_b: Tuple[int, int]) -> float:
        """
        Admissible heuristic for 8-direction grid (Octile distance).
        Since minimum possible edge movement cost is equal to distance_cost (when risk=0),
        Octile distance is strictly admissible and consistent.
        """
        dr = abs(node_a[0] - node_b[0])
        dc = abs(node_a[1] - node_b[1])
        # Octile distance formula:
        # max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)
        return float(max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc))
