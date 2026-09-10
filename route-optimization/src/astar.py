"""
Hazard-Aware A* Pathfinding Algorithm Implementation.
"""

import heapq
from typing import List, Tuple, Dict, Any, Optional
from .grid_graph import GridGraph


class AStarPlanner:
    """
    Executes A* search on GridGraph taking into account distance and cell risk.
    """

    def __init__(self, graph: GridGraph):
        self.graph = graph

    def plan(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates optimal path from start to goal minimizing distance and hazard risk.

        Returns:
            Dict containing route waypoints and statistics, or None if no path exists.
        """
        if start == goal:
            start_risk = self.graph.get_cell_risk(start[0], start[1])
            return {
                "route": [start],
                "total_distance": 0.0,
                "total_risk": start_risk,
                "average_risk": start_risk,
                "total_cost": 0.0,
            }

        # Open set stored as binary heap: (f_score, tie_breaker_counter, (row, col))
        counter = 0
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {
            start: self.graph.heuristic(start, goal)
        }
        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], float, float]] = {}

        heapq.heappush(open_heap, (f_score[start], counter, start))
        visited_nodes = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in visited_nodes:
                continue
            visited_nodes.add(current)

            if current == goal:
                # Reconstruct path
                return self._reconstruct_path(start, goal, came_from)

            for neighbor, step_cost, step_dist in self.graph.get_neighbors(
                current[0], current[1]
            ):
                if neighbor in visited_nodes:
                    continue

                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = (current, step_cost, step_dist)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.graph.heuristic(
                        neighbor, goal
                    )
                    counter += 1
                    heapq.heappush(
                        open_heap, (f_score[neighbor], counter, neighbor)
                    )

        # Path not found
        return None

    def _reconstruct_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        came_from: Dict[Tuple[int, int], Tuple[Tuple[int, int], float, float]],
    ) -> Dict[str, Any]:
        """Reconstructs ordered path and calculates distance, risk, and cost metrics."""
        path: List[Tuple[int, int]] = [goal]
        current = goal

        total_distance = 0.0
        total_cost = 0.0

        while current != start:
            prev, step_cost, step_dist = came_from[current]
            total_distance += step_dist
            total_cost += step_cost
            current = prev
            path.append(current)

        path.reverse()

        # Calculate risk metrics along path
        cell_risks = [self.graph.get_cell_risk(r, c) for r, c in path]
        total_risk = float(sum(cell_risks))
        average_risk = float(total_risk / len(path))

        return {
            "route": path,
            "total_distance": round(total_distance, 4),
            "total_risk": round(total_risk, 4),
            "average_risk": round(average_risk, 4),
            "total_cost": round(total_cost, 4),
        }
