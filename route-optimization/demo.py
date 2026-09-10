"""
Demo script for Hazard-Aware A* Route Optimization.
Loads sample hazard grid, computes safe route, displays ASCII map, and exports JSON result.
"""

import json
import sys
from pathlib import Path

# Ensure src module is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.route_engine import calculate_route


def print_ascii_grid(
    grid: list,
    start: list,
    goal: list,
    route_waypoints: list,
    critical_threshold: float = 0.95
):
    """Prints ASCII representation of hazard grid and calculated route."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    route_set = {(p["row"], p["col"]) for p in route_waypoints}
    start_pos = (start[0], start[1])
    goal_pos = (goal[0], goal[1])

    print("\n--- VISUAL HAZARD GRID & ROUTE MAP ---")
    print("Legend: S = Start, G = Goal, X = Critical/Blocked, * = Selected Route, . = Safe, ~ = High Risk\n")

    # Header column indices
    col_header = "    " + "".join([f" {c} " for c in range(cols)])
    print(col_header)
    print("   +" + "---" * cols + "+")

    for r in range(rows):
        row_str = f"{r:2d} |"
        for c in range(cols):
            cell_pos = (r, c)
            cell_risk = grid[r][c]

            if cell_pos == start_pos:
                symbol = " S "
            elif cell_pos == goal_pos:
                symbol = " G "
            elif cell_pos in route_set:
                symbol = " * "
            elif cell_risk >= critical_threshold:
                symbol = " X "
            elif cell_risk >= 0.5:
                symbol = " ~ "
            else:
                symbol = " . "
            row_str += symbol
        row_str += "|"
        print(row_str)

    print("   +" + "---" * cols + "+")


def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / "data" / "sample_hazard_grid.json"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sample_route.json"

    print("=" * 60)
    print("   AI-ENABLED ANTARCTIC SEA-ICE SAFE ROUTE OPTIMIZATION")
    print("=" * 60)

    if not input_file.exists():
        print(f"Error: Sample grid file not found at {input_file}")
        sys.exit(1)

    print(f"Loading input hazard grid from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = data["grid"]
    start = data["start"]
    goal = data["goal"]
    risk_weight = data.get("risk_weight", 10.0)
    critical_threshold = data.get("critical_threshold", 0.95)
    geo_config = data.get("geo_config", None)

    print(f"Grid size: {len(grid)}x{len(grid[0])}")
    print(f"START position: [Row {start[0]}, Col {start[1]}]")
    print(f"GOAL position:  [Row {goal[0]}, Col {goal[1]}]")
    print(f"Risk weight factor: {risk_weight}")
    print(f"Critical hazard threshold: {critical_threshold}")

    print("\nCalculating optimal safe route using Hazard-Aware A*...")
    result = calculate_route(
        grid=grid,
        start=start,
        goal=goal,
        risk_weight=risk_weight,
        critical_threshold=critical_threshold,
        geo_config=geo_config
    )

    print("\n" + "=" * 60)
    print("                     ROUTE RESULTS")
    print("=" * 60)

    if result["success"]:
        print("STATUS:               ROUTE FOUND SUCCESSFUL")
        print(f"NUMBER OF WAYPOINTS:  {result['number_of_waypoints']}")
        print(f"TOTAL DISTANCE:       {result['total_distance']:.2f} grid units")
        print(f"AVERAGE ROUTE RISK:   {result['average_risk']:.4f}")
        print(f"TOTAL ROUTE RISK:     {result['total_risk']:.4f}")

        if "distance_km" in result:
            print(f"TOTAL DISTANCE (KM):  {result['distance_km']:.2f} km")
            print(f"GEOGRAPHIC START:     Lat {result['start']['latitude']}, Lon {result['start']['longitude']}")
            print(f"GEOGRAPHIC GOAL:      Lat {result['destination']['latitude']}, Lon {result['destination']['longitude']}")

        print("\nGRID ROUTE WAYPOINTS:")
        for idx, wp in enumerate(result["route"]):
            print(f"  Step {idx:2d}: Row {wp['row']}, Col {wp['col']}")

        # Display ASCII Visualization
        print_ascii_grid(
            grid=grid,
            start=start,
            goal=goal,
            route_waypoints=result["route"],
            critical_threshold=critical_threshold
        )
    else:
        print("STATUS:               NO ROUTE FOUND")
        print(f"ERROR REASON:         {result.get('error', 'Unknown failure')}")

    # Export result to outputs/sample_route.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to: {output_file}\n")


if __name__ == "__main__":
    main()
