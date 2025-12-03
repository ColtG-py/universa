"""
Pathfinding Module
A* pathfinding implementation for agent navigation
"""

from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass
import heapq
import math


@dataclass
class PathNode:
    """Node in the pathfinding graph"""
    x: int
    y: int
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to end
    parent: Optional['PathNode'] = None

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)"""
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'PathNode') -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class AStarPathfinder:
    """
    A* pathfinding for agent navigation.
    Takes terrain costs into account and can prefer roads.
    """

    # Movement costs for different terrains (multipliers)
    TERRAIN_COSTS = {
        "ocean_trench": float('inf'),
        "ocean_deep": float('inf'),
        "ocean_shallow": float('inf'),
        "ocean_shelf": float('inf'),
        "ocean_coral_reef": float('inf'),
        "ice": 3.0,
        "tundra": 1.5,
        "cold_desert": 1.3,
        "boreal_forest": 1.4,
        "temperate_rainforest": 1.5,
        "temperate_deciduous_forest": 1.3,
        "temperate_grassland": 1.0,
        "mediterranean": 1.1,
        "hot_desert": 1.8,
        "savanna": 1.1,
        "tropical_seasonal_forest": 1.4,
        "tropical_rainforest": 2.0,
        "alpine": 2.5,
        "mangrove": 2.5,
    }

    # Road speed multipliers (lower = faster)
    ROAD_COSTS = {
        "imperial_highway": 0.3,
        "main_road": 0.4,
        "rural_road": 0.5,
        "path": 0.7,
        "trail": 0.8,
    }

    def __init__(self, world_interface):
        """
        Initialize pathfinder.

        Args:
            world_interface: WorldInterface for querying terrain
        """
        self.world_interface = world_interface

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        world_state=None,
        prefer_roads: bool = True,
        max_iterations: int = 10000
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to end using A*.

        Args:
            start: Starting (x, y) coordinates
            end: Destination (x, y) coordinates
            world_state: World state for terrain queries
            prefer_roads: Whether to prefer roads
            max_iterations: Maximum iterations before giving up

        Returns:
            List of (x, y) coordinates forming path, or None if no path found
        """
        start_node = PathNode(
            x=start[0],
            y=start[1],
            g_cost=0,
            h_cost=self._heuristic(start, end)
        )

        open_set: List[PathNode] = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        node_map: Dict[Tuple[int, int], PathNode] = {start: start_node}

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)

            # Check if we've reached the goal
            if current.x == end[0] and current.y == end[1]:
                return self._reconstruct_path(current)

            closed_set.add((current.x, current.y))

            # Check all neighbors
            for neighbor_pos in self._get_neighbors(current.x, current.y):
                if neighbor_pos in closed_set:
                    continue

                # Calculate movement cost
                move_cost = self._get_movement_cost(
                    (current.x, current.y),
                    neighbor_pos,
                    world_state,
                    prefer_roads
                )

                if move_cost == float('inf'):
                    continue  # Impassable terrain

                tentative_g = current.g_cost + move_cost

                if neighbor_pos in node_map:
                    neighbor = node_map[neighbor_pos]
                    if tentative_g >= neighbor.g_cost:
                        continue
                    # Found better path
                    neighbor.g_cost = tentative_g
                    neighbor.parent = current
                else:
                    neighbor = PathNode(
                        x=neighbor_pos[0],
                        y=neighbor_pos[1],
                        g_cost=tentative_g,
                        h_cost=self._heuristic(neighbor_pos, end),
                        parent=current
                    )
                    node_map[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)

        return None  # No path found

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbor positions (8-directional movement)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((x + dx, y + dy))
        return neighbors

    def _get_movement_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        world_state,
        prefer_roads: bool
    ) -> float:
        """
        Calculate movement cost between adjacent tiles.

        Args:
            from_pos: Current position
            to_pos: Target position
            world_state: World state for terrain queries
            prefer_roads: Whether to apply road bonus

        Returns:
            Movement cost (float, inf for impassable)
        """
        # Query destination location
        loc = self.world_interface.query_location(to_pos[0], to_pos[1], world_state)

        if loc is None:
            return float('inf')

        # Base cost is diagonal or orthogonal distance
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        base_cost = 1.414 if (dx + dy == 2) else 1.0

        # Apply terrain modifier
        terrain_cost = self.TERRAIN_COSTS.get(loc.biome_type, 1.0)
        if terrain_cost == float('inf'):
            return float('inf')

        cost = base_cost * terrain_cost

        # Elevation change penalty
        if loc.elevation is not None:
            # Query source elevation
            src_loc = self.world_interface.query_location(from_pos[0], from_pos[1], world_state)
            if src_loc and src_loc.elevation is not None:
                elevation_diff = abs(loc.elevation - src_loc.elevation)
                # Penalty for steep terrain (>10m per tile = 10% grade at 100m tiles)
                if elevation_diff > 10:
                    cost *= 1 + (elevation_diff - 10) * 0.05

        # Apply road bonus
        if prefer_roads and loc.has_road and loc.road_type:
            road_multiplier = self.ROAD_COSTS.get(loc.road_type, 0.8)
            cost *= road_multiplier

        # Water crossing penalty (if not on a road/bridge)
        if loc.has_water and not loc.has_road:
            cost *= 5.0  # Significant penalty for wading

        return cost

    def _reconstruct_path(self, end_node: PathNode) -> List[Tuple[int, int]]:
        """Reconstruct path from end node back to start"""
        path = []
        current = end_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path


def calculate_travel_time(
    path: List[Tuple[int, int]],
    base_speed_kmh: float = 4.0,
    tile_size_m: float = 100.0
) -> float:
    """
    Calculate estimated travel time for a path.

    Args:
        path: List of (x, y) coordinates
        base_speed_kmh: Base walking speed in km/h
        tile_size_m: Size of one tile in meters

    Returns:
        Estimated travel time in hours
    """
    if len(path) < 2:
        return 0.0

    total_distance = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        distance = math.sqrt(dx*dx + dy*dy) * tile_size_m
        total_distance += distance

    # Convert to km and calculate time
    distance_km = total_distance / 1000.0
    travel_time = distance_km / base_speed_kmh

    return travel_time
