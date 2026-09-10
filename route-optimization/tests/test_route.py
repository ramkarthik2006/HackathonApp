"""
Comprehensive test suite for Hazard-Aware A* Route Optimization module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.route_engine import calculate_route
from src.grid_graph import GridGraph
from src.utils import validate_grid_and_points, grid_to_geographic


def test_basic_route_exists():
    """Test 1 & 2 & 3: Basic route exists, start and goal are correct."""
    grid = [
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.1],
        [0.0, 0.1, 0.0]
    ]
    start = [0, 0]
    goal = [2, 2]

    result = calculate_route(grid, start, goal)

    assert result["success"] is True
    assert len(result["route"]) > 0
    assert result["route"][0] == {"row": 0, "col": 0}
    assert result["route"][-1] == {"row": 2, "col": 2}


def test_critical_cells_avoided():
    """Test 4: Critical cells (risk >= threshold) are strictly avoided."""
    # Place a critical barrier in middle row
    grid = [
        [0.0, 0.0, 0.0],
        [0.98, 0.98, 0.0],
        [0.0, 0.0, 0.0]
    ]
    start = [0, 0]
    goal = [2, 0]

    result = calculate_route(grid, start, goal, critical_threshold=0.95)

    assert result["success"] is True
    route = result["route"]

    # Verify no cell in route hits a critical cell
    for point in route:
        r, c = point["row"], point["col"]
        assert grid[r][c] < 0.95, f"Route passed through critical cell at [{r}, {c}]"


def test_risk_aware_routing_preference():
    """
    Test 5 (IMPORTANT): Verifies risk-aware path selection.
    Grid offers two pathways from [0,0] to [2,2]:
      Pathway A (Direct middle row/col): short distance but high risk (0.80).
      Pathway B (Outer path): longer distance but low risk (0.01).
    Verify that:
      - With risk_weight = 10.0, A* chooses the safe detour (Pathway B).
      - With risk_weight = 0.0, A* chooses the shortest direct path (Pathway A).
    """
    grid = [
        [0.01, 0.01, 0.01],
        [0.80, 0.80, 0.01],
        [0.01, 0.01, 0.01]
    ]
    start = [0, 0]
    goal = [2, 0]

    # With high risk_weight, should prefer safe outer path around [1,0] & [1,1]
    safe_result = calculate_route(grid, start, goal, risk_weight=10.0)
    assert safe_result["success"] is True
    safe_waypoints = [(p["row"], p["col"]) for p in safe_result["route"]]
    assert (1, 0) not in safe_waypoints, "High risk_weight route should avoid high-risk cell [1, 0]"

    # With zero risk_weight, should choose shortest path straight down through [1,0]
    shortest_result = calculate_route(grid, start, goal, risk_weight=0.0)
    assert shortest_result["success"] is True
    shortest_waypoints = [(p["row"], p["col"]) for p in shortest_result["route"]]
    assert (1, 0) in shortest_waypoints, "Zero risk_weight should take shortest path through [1, 0]"


def test_invalid_grid_rejected():
    """Test 6: Rejects non-rectangular, non-2D, or invalid risk value grids."""
    # Non-rectangular
    grid_non_rect = [[0.1, 0.2], [0.1]]
    res1 = calculate_route(grid_non_rect, [0, 0], [0, 1])
    assert res1["success"] is False
    assert "non-rectangular" in res1["error"].lower()

    # Out of range risk (> 1.0)
    grid_invalid_risk = [[0.1, 1.5], [0.2, 0.1]]
    res2 = calculate_route(grid_invalid_risk, [0, 0], [1, 1])
    assert res2["success"] is False
    assert "outside valid range" in res2["error"].lower()

    # Non-numeric risk
    grid_str = [["safe", 0.1], [0.2, 0.1]]
    res3 = calculate_route(grid_str, [0, 0], [1, 1])
    assert res3["success"] is False
    assert "invalid non-numeric" in res3["error"].lower()


def test_start_outside_grid_rejected():
    """Test 7: Start coordinate outside grid bounds is rejected."""
    grid = [[0.1, 0.1], [0.1, 0.1]]
    result = calculate_route(grid, [5, 0], [1, 1])
    assert result["success"] is False
    assert "out of grid bounds" in result["error"].lower()


def test_goal_outside_grid_rejected():
    """Test 8: Goal coordinate outside grid bounds is rejected."""
    grid = [[0.1, 0.1], [0.1, 0.1]]
    result = calculate_route(grid, [0, 0], [0, 9])
    assert result["success"] is False
    assert "out of grid bounds" in result["error"].lower()


def test_blocked_start_rejected():
    """Test 9: Starting cell blocked by critical hazard is rejected."""
    grid = [[0.98, 0.1], [0.1, 0.1]]
    result = calculate_route(grid, [0, 0], [1, 1], critical_threshold=0.95)
    assert result["success"] is False
    assert "start location" in result["error"].lower() and "critical" in result["error"].lower()


def test_blocked_goal_rejected():
    """Test 10: Goal cell blocked by critical hazard is rejected."""
    grid = [[0.1, 0.1], [0.1, 0.99]]
    result = calculate_route(grid, [0, 0], [1, 1], critical_threshold=0.95)
    assert result["success"] is False
    assert "goal location" in result["error"].lower() and "critical" in result["error"].lower()


def test_no_route_scenario():
    """Test 11: Returns clear failure result when goal is completely walled off."""
    grid = [
        [0.0, 0.0, 0.0],
        [0.98, 0.98, 0.98],
        [0.0, 0.0, 0.0]
    ]
    start = [0, 0]
    goal = [2, 2]

    result = calculate_route(grid, start, goal, critical_threshold=0.95)

    assert result["success"] is False
    assert "no valid route exists" in result["error"].lower()


def test_route_engine_output_schema():
    """Test 12: Route engine returns all required output fields."""
    grid = [
        [0.1, 0.2],
        [0.2, 0.1]
    ]
    result = calculate_route(grid, [0, 0], [1, 1], risk_weight=5.0)

    assert result["success"] is True
    assert "route" in result
    assert "number_of_waypoints" in result
    assert "total_distance" in result
    assert "average_risk" in result
    assert "total_risk" in result
    assert result["risk_weight"] == 5.0
    assert result["critical_threshold"] == 0.95


def test_geographic_coordinate_integration():
    """Test 13: Converts grid route to geographic coordinates when geo_config is supplied."""
    grid = [
        [0.1, 0.2],
        [0.2, 0.1]
    ]
    geo_config = {
        "origin": {"latitude": -64.0, "longitude": 52.0},
        "cell_size_km": 10.0
    }

    result = calculate_route(grid, [0, 0], [1, 1], geo_config=geo_config)

    assert result["success"] is True
    assert "start" in result
    assert "destination" in result
    assert "geo_route" in result
    assert "distance_km" in result
    assert "risk_score" in result
    assert result["start"]["latitude"] == -64.0
    assert result["start"]["longitude"] == 52.0
    assert len(result["geo_route"]) == result["number_of_waypoints"]


def test_numpy_array_input_support():
    """Test 14: Verifies numpy ndarray input support for grid, start, and goal."""
    grid_np = np.array([
        [0.05, 0.10, 0.15],
        [0.10, 0.20, 0.10],
        [0.15, 0.10, 0.05]
    ], dtype=np.float64)
    start_np = np.array([0, 0])
    goal_np = np.array([2, 2])

    result = calculate_route(grid_np, start_np, goal_np)

    assert result["success"] is True
    assert len(result["route"]) > 0
    assert result["route"][0] == {"row": 0, "col": 0}
    assert result["route"][-1] == {"row": 2, "col": 2}
