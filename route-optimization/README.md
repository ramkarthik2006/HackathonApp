# Safe Route Optimization using Hazard-Aware A*

**Project:** AI-Enabled Antarctic Sea-Ice & Navigation Decision Support  
**Module:** Member 5 - Safe Route Optimization  
**Branch:** `route-optimization`

---

## Objective

The objective of this module is to calculate safe, cost-effective navigation routes through Antarctic sea-ice environments. Traditional shortest-path routing algorithms (such as standard Dijkstra or unweighted A*) select the shortest spatial distance, which often forces vessels through hazardous ice formations, icebergs, or dense pressure ridges.

This module implements a **Hazard-Aware A\*** pathfinding algorithm that dynamically balances **travel distance** against **spatial hazard/risk scores** provided by satellite processing models, steering vessels along safer routes while avoiding critical hazards entirely.

---

## Member 5 Responsibility

Member 5 is responsible for:
1. Receiving a spatial hazard/risk grid provided by Member 4's Risk Engine.
2. Executing 8-directional Hazard-Aware A* path planning from a designated START location to a GOAL location.
3. Filtering out critical hazard cells (risk $\ge 0.95$) as impenetrable obstacles.
4. Exporting structured JSON routes and statistics ready for consumption by Member 6's React/Leaflet dashboard.

> **Note:** Member 5 does NOT perform satellite image preprocessing, iceberg detection, drift forecasting, or frontend dashboard rendering. This module focuses strictly on route optimization.

---

## System Position

```
Member 4 (Risk/Hazard Engine)
          │
          ▼
   Risk/Hazard Grid
          │
          ▼
Member 5 (Hazard-Aware A* Planner)
          │
          ▼
   Optimal Safe Route
          │
          ▼
Member 6 (React/Leaflet Dashboard)
```

---

## Input Format

Member 4 provides a standardized JSON payload containing a 2D matrix of risk values, START coordinates, GOAL coordinates, and optional geographic overlay metadata:

```json
{
  "grid": [
    [0.05, 0.05, 0.10],
    [0.05, 0.80, 0.10],
    [0.05, 0.05, 0.05]
  ],
  "start": [0, 0],
  "goal": [2, 2],
  "risk_weight": 10.0,
  "critical_threshold": 0.95,
  "geo_config": {
    "origin": {
      "latitude": -64.0,
      "longitude": 52.0
    },
    "cell_size_km": 5.0
  }
}
```

* **Grid:** 2D rectangular matrix where each cell represents a localized sea-ice hazard score from `0.0` (completely safe open water) to `1.0` (extremely dangerous / solid ice).
* **Start / Goal:** `[row, column]` 0-indexed integer coordinates.

---

## Risk Model

* `0.00 – 0.10`: Open water / very thin ice (Safe)
* `0.10 – 0.30`: Light sea ice (Low Risk)
* `0.30 – 0.60`: Moderate pack ice (Medium Risk)
* `0.60 – 0.94`: Heavy ice / pressure ridges (High Risk)
* `0.95 – 1.00`: Icebergs / thick ice sheet (Critical / Blocked)

---

## A* Algorithm & Cost Function

The module utilizes an 8-directional A* algorithm with a custom hazard-weighted step cost:

$$f(n) = g(n) + h(n)$$

Where:
* $g(n)$ is the accumulated path cost from the start node to node $n$.
* $h(n)$ is an admissible 8-directional **Octile Distance** heuristic to the goal:
  $$h(n) = \max(\Delta r, \Delta c) + (\sqrt{2} - 1) \cdot \min(\Delta r, \Delta c)$$
* **Step Cost Formula:**
  $$\text{movement\_cost} = \text{distance\_cost} + (\text{risk\_weight} \times \text{destination\_cell\_risk})$$

Where:
* $\text{distance\_cost} = 1.0$ for straight movements (Up, Down, Left, Right).
* $\text{distance\_cost} = \sqrt{2} \approx 1.4142$ for diagonal movements.
* `risk_weight` default is `10.0` (configurable).

### Why Risk-Aware Routing Works
When `risk_weight > 0`, traversing a high-risk cell (e.g. risk `0.80`) adds $10.0 \times 0.80 = 8.0$ penalty units to the step cost. A* will naturally choose a longer spatial detour through safe cells (risk `0.05`, penalty `0.5`) because the total cost $g(n)$ along the detour is lower than pushing through high-risk shortcuts.

---

## Critical Hazard Handling

Cells with a risk score at or above `critical_threshold` (default `0.95`) are classified as blocked obstacles.
* Neighbors with risk $\ge 0.95$ are excluded during neighbor expansion in `GridGraph`.
* The route engine verifies that `start` and `goal` coordinates are not blocked before pathfinding starts.
* If all routes are blocked, the engine returns a clean failure result (`"success": false`) without throwing unhandled exceptions.

---

## Output Format

The engine outputs a standardized JSON payload designed for Member 6's dashboard integration:

```json
{
  "success": true,
  "route": [
    {"row": 0, "col": 0},
    {"row": 1, "col": 0},
    {"row": 2, "col": 0}
  ],
  "number_of_waypoints": 3,
  "total_distance": 2.0,
  "average_risk": 0.05,
  "total_risk": 0.15,
  "risk_weight": 10.0,
  "critical_threshold": 0.95,
  "start": {
    "latitude": -64.0,
    "longitude": 52.0
  },
  "destination": {
    "latitude": -64.09,
    "longitude": 52.12
  },
  "geo_route": [
    {"latitude": -64.0, "longitude": 52.0},
    {"latitude": -64.045, "longitude": 52.0},
    {"latitude": -64.09, "longitude": 52.12}
  ],
  "distance_km": 10.0,
  "risk_score": 0.05
}
```

---

## Geographic Coordinate Integration

To convert local grid coordinates $[r, c]$ to Antarctic geographic coordinates $(\text{Latitude}, \text{Longitude})$:
* $\text{Latitude} = \text{origin\_lat} - \frac{r \times \text{cell\_size\_km}}{111.0}$
* $\text{Longitude} = \text{origin\_lon} + \frac{c \times \text{cell\_size\_km}}{111.0 \times \cos(\text{radians}(\text{origin\_lat}))}$

Grid pathfinding is strictly decoupled from coordinate conversion so that pathfinding remains fast and grid-native.

---

## Installation

```bash
cd route-optimization
pip install -r requirements.txt
```

---

## Running Demo

Execute the demo script to run route optimization on the sample dataset and display terminal visualizations:

```bash
python demo.py
```

Output is saved to `outputs/sample_route.json`.

---

## Running Tests

Execute the complete pytest test suite:

```bash
pytest tests/ -v
```

---

## Member 4 Integration Interface

Member 4 can invoke the route engine directly in Python:

```python
from route_optimization.src import calculate_route

result = calculate_route(
    grid=hazard_matrix,
    start=[start_row, start_col],
    goal=[goal_row, goal_col],
    risk_weight=10.0,
    critical_threshold=0.95,
    geo_config={"origin": {"latitude": -64.0, "longitude": 52.0}, "cell_size_km": 5.0}
)
```

---

## Member 6 Integration Contract

Member 6 can consume `outputs/sample_route.json` or call `calculate_route(...)` directly via API endpoints. The fields `geo_route`, `distance_km`, and `risk_score` map directly to Leaflet polyline overlays and dashboard summary cards.

---

## Limitations

* Grid resolution: Processing scales with $O(V \log V)$ where $V = \text{rows} \times \text{cols}$.
* Static snapshot: Route assumes hazard grid is stationary for the duration of the voyage step.

---

## Future Improvements

* Dynamic time-dependent A* incorporating iceberg drift velocity vectors from Member 3.
* Smooth path interpolation (e.g., Catmull-Rom or B-splines) for smoother vessel turn radii.

---

> **Disclaimer:** All sample grid data in `data/sample_hazard_grid.json` is synthetic demo data created for testing and verification purposes.
