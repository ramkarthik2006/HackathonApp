"""
Utility functions for grid validation, coordinate transformation, and formatting.
"""

import math
from typing import List, Tuple, Dict, Any, Union


def validate_grid_and_points(
    grid: Any,
    start: Any,
    goal: Any,
    critical_threshold: float = 0.95
) -> Tuple[List[List[float]], Tuple[int, int], Tuple[int, int]]:
    """
    Validates input grid, start point, and goal point.
    Supports standard Python lists/tuples and NumPy arrays.

    Raises:
        ValueError: If grid or coordinates fail validation.
    """
    # Convert NumPy array if provided
    if hasattr(grid, "tolist") and callable(getattr(grid, "tolist")):
        grid = grid.tolist()

    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        raise ValueError("Grid must be a non-empty 2D array or list of lists.")

    rows = len(grid)
    if not isinstance(grid[0], (list, tuple)) or len(grid[0]) == 0:
        raise ValueError("Grid rows must be non-empty lists.")

    cols = len(grid[0])
    validated_grid: List[List[float]] = []

    for r_idx, row in enumerate(grid):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"Row {r_idx} is not a valid list or array.")
        if len(row) != cols:
            raise ValueError(
                f"Grid is non-rectangular: Row {r_idx} has length {len(row)}, expected {cols}."
            )
        validated_row: List[float] = []
        for c_idx, val in enumerate(row):
            try:
                risk_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid non-numeric risk score '{val}' at cell ({r_idx}, {c_idx})."
                )
            if not (0.0 <= risk_val <= 1.0):
                raise ValueError(
                    f"Risk score {risk_val} at cell ({r_idx}, {c_idx}) is outside valid range [0.0, 1.0]."
                )
            validated_row.append(risk_val)
        validated_grid.append(validated_row)

    # Convert start/goal if NumPy array or sequence
    if hasattr(start, "tolist") and callable(getattr(start, "tolist")):
        start = start.tolist()
    if hasattr(goal, "tolist") and callable(getattr(goal, "tolist")):
        goal = goal.tolist()

    # Validate start point
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError(f"Start coordinate must be a list/tuple of [row, col], got {start}.")
    try:
        start_row, start_col = int(start[0]), int(start[1])
    except (ValueError, TypeError):
        raise ValueError(f"Start coordinates must be integers, got {start}.")

    if not (0 <= start_row < rows and 0 <= start_col < cols):
        raise ValueError(
            f"Start coordinate [{start_row}, {start_col}] is out of grid bounds ({rows}x{cols})."
        )

    # Validate goal point
    if not isinstance(goal, (list, tuple)) or len(goal) != 2:
        raise ValueError(f"Goal coordinate must be a list/tuple of [row, col], got {goal}.")
    try:
        goal_row, goal_col = int(goal[0]), int(goal[1])
    except (ValueError, TypeError):
        raise ValueError(f"Goal coordinates must be integers, got {goal}.")

    if not (0 <= goal_row < rows and 0 <= goal_col < cols):
        raise ValueError(
            f"Goal coordinate [{goal_row}, {goal_col}] is out of grid bounds ({rows}x{cols})."
        )

    # Check if start or goal are blocked
    start_risk = validated_grid[start_row][start_col]
    if start_risk >= critical_threshold:
        raise ValueError(
            f"Start location [{start_row}, {start_col}] is a critical/blocked cell "
            f"(risk {start_risk:.2f} >= threshold {critical_threshold:.2f})."
        )

    goal_risk = validated_grid[goal_row][goal_col]
    if goal_risk >= critical_threshold:
        raise ValueError(
            f"Goal location [{goal_row}, {goal_col}] is a critical/blocked cell "
            f"(risk {goal_risk:.2f} >= threshold {critical_threshold:.2f})."
        )

    return validated_grid, (start_row, start_col), (goal_row, goal_col)


def grid_to_geographic(
    row: int,
    col: int,
    origin_lat: float,
    origin_lon: float,
    cell_size_km: float
) -> Dict[str, float]:
    """
    Maps grid cell [row, col] to approximate Antarctic geographic coordinates (lat, lon).

    Assumptions:
    - Row 0 corresponds to northern edge (origin_lat).
    - Col 0 corresponds to western edge (origin_lon).
    - 1 degree of latitude ≈ 111.0 km.
    - 1 degree of longitude ≈ 111.0 * cos(lat_radians) km.
    """
    lat_delta_deg = (row * cell_size_km) / 111.0
    lat = origin_lat - lat_delta_deg

    cos_lat = math.cos(math.radians(origin_lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    lon_delta_deg = (col * cell_size_km) / (111.0 * cos_lat)
    lon = origin_lon + lon_delta_deg

    return {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4)
    }


def convert_route_to_geo(
    waypoints: List[Tuple[int, int]],
    geo_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Converts list of grid waypoints [(r, c), ...] to geographic format.
    """
    origin = geo_config.get("origin", {})
    origin_lat = float(origin.get("latitude", -64.0))
    origin_lon = float(origin.get("longitude", 52.0))
    cell_size_km = float(geo_config.get("cell_size_km", 5.0))

    geo_route = [
        grid_to_geographic(r, c, origin_lat, origin_lon, cell_size_km)
        for (r, c) in waypoints
    ]

    start_geo = geo_route[0] if geo_route else {}
    dest_geo = geo_route[-1] if geo_route else {}

    return {
        "start": start_geo,
        "destination": dest_geo,
        "route": geo_route
    }
