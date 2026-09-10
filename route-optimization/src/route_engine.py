"""
Main high-level Route Engine interface for hazard-aware path planning.
"""

from typing import Dict, Any, Optional
from .utils import validate_grid_and_points, convert_route_to_geo
from .grid_graph import GridGraph
from .astar import AStarPlanner


def calculate_route(
    grid: Any,
    start: Any,
    goal: Any,
    risk_weight: float = 10.0,
    critical_threshold: float = 0.95,
    geo_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for hazard-aware safe route calculation.

    Args:
        grid: 2D list/array of hazard scores (0.0 safe - 1.0 critical).
        start: [row, col] start coordinates.
        goal: [row, col] destination coordinates.
        risk_weight: Multiplier weight for hazard cost contribution.
        critical_threshold: Risk threshold at or above which cells are blocked.
        geo_config: Optional dict containing origin latitude/longitude and cell_size_km.

    Returns:
        Structured dictionary containing route details, statistics, and success status.
    """
    # 1. Validate inputs
    try:
        validated_grid, start_pos, goal_pos = validate_grid_and_points(
            grid, start, goal, critical_threshold
        )
    except ValueError as err:
        return {
            "success": False,
            "error": str(err),
            "route": [],
            "number_of_waypoints": 0,
            "total_distance": 0.0,
            "average_risk": 0.0,
            "total_risk": 0.0,
            "risk_weight": risk_weight,
            "critical_threshold": critical_threshold,
        }

    # 2. Build grid graph
    graph = GridGraph(
        grid=validated_grid,
        risk_weight=risk_weight,
        critical_threshold=critical_threshold,
    )

    # 3. Run A* planner
    planner = AStarPlanner(graph)
    plan_result = planner.plan(start_pos, goal_pos)

    # 4. Handle no route case
    if plan_result is None:
        return {
            "success": False,
            "error": "No valid route exists between start and goal (all paths blocked or disconnected).",
            "route": [],
            "number_of_waypoints": 0,
            "total_distance": 0.0,
            "average_risk": 0.0,
            "total_risk": 0.0,
            "risk_weight": risk_weight,
            "critical_threshold": critical_threshold,
        }

    # 5. Format grid route waypoints
    grid_waypoints = plan_result["route"]
    formatted_route = [{"row": r, "col": c} for (r, c) in grid_waypoints]

    output: Dict[str, Any] = {
        "success": True,
        "route": formatted_route,
        "number_of_waypoints": len(formatted_route),
        "total_distance": plan_result["total_distance"],
        "average_risk": plan_result["average_risk"],
        "total_risk": plan_result["total_risk"],
        "risk_weight": risk_weight,
        "critical_threshold": critical_threshold,
    }

    # 6. Apply geographic conversion if geo_config is supplied
    if geo_config and isinstance(geo_config, dict):
        geo_data = convert_route_to_geo(grid_waypoints, geo_config)
        output["start"] = geo_data["start"]
        output["destination"] = geo_data["destination"]
        output["geo_route"] = geo_data["route"]
        output["distance_km"] = round(
            plan_result["total_distance"] * float(geo_config.get("cell_size_km", 1.0)),
            2,
        )
        output["risk_score"] = plan_result["average_risk"]

    return output
