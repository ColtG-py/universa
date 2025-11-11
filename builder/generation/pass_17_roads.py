"""
World Builder - Pass 17: Road Networks
Generates realistic road networks connecting settlements using graph algorithms.

APPROACH:
1. Primary Highway Network: Use Prim's MST to identify major city connections,
   then A* pathfinding to create terrain-aware roads between them
2. Secondary Roads: Connect smaller settlements to highways or other settlements
   using cost-based decision making
3. Terrain Costs: Roads avoid difficult terrain but can cross water when beneficial

ALGORITHM:
- Prim's MST for connectivity graph
- A* pathfinding for actual road placement
- Hierarchical road types (Imperial Highway -> Rural Road)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import heapq

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType
from models.world import WorldState, RoadSegment, Bridge
from utils.spatial import calculate_slope


@dataclass
class RoadNode:
    """A node in the road network graph."""
    settlement_id: int
    x: int
    y: int
    settlement_type: int  # Used for determining road priority
    
    
@dataclass
class PathResult:
    """Result of A* pathfinding."""
    path: List[Tuple[int, int]]  # List of (x, y) coordinates
    cost: float
    crosses_water: bool
    water_crossings: List[Tuple[int, int]]  # Bridge locations


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate road network connecting all settlements.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating road networks (MST + A* pathfinding)...")
    
    size = world_state.size
    seed = params.seed
    rng = np.random.default_rng(seed + 17000)
    
    # STEP 1: Collect settlements and environmental data
    print(f"    - Collecting settlements and terrain data...")
    
    settlements = collect_settlements(world_state)
    
    if not settlements or len(settlements) < 2:
        print(f"    - WARNING: Not enough settlements to generate roads")
        return
    
    print(f"      Found {len(settlements)} settlements")
    
    # Collect terrain data for pathfinding
    terrain_data = collect_terrain_data(world_state, size)
    
    # STEP 2: Build primary highway network (major cities)
    print(f"    - Building primary highway network (major cities/metropolises)...")
    
    major_cities = [s for s in settlements if s.settlement_type >= 3]  # Cities and Metropolises
    
    if len(major_cities) < 2:
        print(f"      WARNING: Only {len(major_cities)} major cities, expanding to include towns...")
        major_cities = [s for s in settlements if s.settlement_type >= 2]  # Include towns
    
    print(f"      Connecting {len(major_cities)} major settlements...")
    
    highway_network = build_highway_network(
        major_cities,
        terrain_data,
        size,
        world_state
    )
    
    print(f"      Created {len(highway_network)} highway segments")
    
    # STEP 3: Connect secondary settlements (towns, villages, hamlets)
    print(f"    - Connecting secondary settlements to network...")
    
    secondary_settlements = [s for s in settlements if s.settlement_type < 3]
    
    if secondary_settlements:
        secondary_roads = connect_secondary_settlements(
            secondary_settlements,
            highway_network,
            major_cities,
            terrain_data,
            size,
            world_state
        )
        
        print(f"      Created {len(secondary_roads)} secondary roads")
        
        # Combine networks
        all_roads = highway_network + secondary_roads
    else:
        all_roads = highway_network
    
    print(f"    - Total road segments: {len(all_roads)}")
    
    # STEP 4: Rasterize roads for visualization
    print(f"    - Rasterizing roads for visualization...")
    
    road_map = rasterize_roads(all_roads, size)
    road_type_map = rasterize_road_types(all_roads, size)
    
    # STEP 5: Identify bridge locations
    print(f"    - Identifying bridge locations...")
    
    bridges = identify_bridges(all_roads, terrain_data['elevation'])
    
    print(f"      Found {len(bridges)} bridges")
    
    # STEP 6: Store in chunks
    print(f"    - Storing road data in chunks...")
    
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = x_start + CHUNK_SIZE
            y_end = y_start + CHUNK_SIZE
            
            # Extract chunk roads
            chunk_roads = [
                road for road in all_roads
                if any(x_start <= x < x_end and y_start <= y < y_end
                       for x, y in road.path)
            ]
            
            # Extract chunk bridges
            chunk_bridges = [
                bridge for bridge in bridges
                if x_start <= bridge.x < x_end and y_start <= bridge.y < y_end
            ]
            
            chunk.roads = chunk_roads
            chunk.bridges = chunk_bridges
            
            # Store rasterized maps
            chunk.road_presence = road_map[x_start:x_end, y_start:y_end].copy()
            chunk.road_type = road_type_map[x_start:x_end, y_start:y_end].copy()
    
    # Store global road network
    world_state.road_network = all_roads
    world_state.bridges = bridges
    
    # STEP 7: Calculate statistics
    total_length = sum(len(road.path) for road in all_roads)
    highway_count = sum(1 for road in all_roads if road.road_type <= 1)
    bridge_count = len(bridges)
    
    print(f"  - Road network statistics:")
    print(f"    Total road segments: {len(all_roads)}")
    print(f"    Total road length: {total_length:,} cells")
    print(f"    Highway segments: {highway_count}")
    print(f"    Secondary roads: {len(all_roads) - highway_count}")
    print(f"    Bridges: {bridge_count}")
    
    # Calculate average travel time to nearest settlement
    avg_time = calculate_average_travel_time(settlements, all_roads, size)
    print(f"    Average travel time to nearest city: {avg_time:.1f} hours")


def collect_settlements(world_state: WorldState) -> List:
    """Collect all non-ruin settlements from chunks."""
    settlements = []
    
    for chunk in world_state.chunks.values():
        if hasattr(chunk, 'settlements') and chunk.settlements:
            for settlement in chunk.settlements:
                if not settlement.is_ruin:
                    settlements.append(settlement)
    
    return settlements


def collect_terrain_data(world_state: WorldState, size: int) -> Dict[str, np.ndarray]:
    """
    Collect all terrain data needed for pathfinding cost calculations.
    
    Returns dict with:
        - elevation: float32[size, size]
        - biome: uint8[size, size]
        - slope: float32[size, size]
        - water_mask: bool[size, size]
    """
    elevation = np.zeros((size, size), dtype=np.float32)
    biome = np.zeros((size, size), dtype=np.uint8)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.biome_type is not None:
                biome[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
    
    # Calculate slope
    slope = calculate_slope(elevation)
    
    # Water mask
    water_mask = elevation <= 0
    
    return {
        'elevation': elevation,
        'biome': biome,
        'slope': slope,
        'water_mask': water_mask,
    }


def build_highway_network(
    major_cities: List,
    terrain_data: Dict,
    size: int,
    world_state: WorldState
) -> List[RoadSegment]:
    """
    Build primary highway network connecting major cities using Prim's MST + A*.
    
    Algorithm:
    1. Use Prim's algorithm to determine connectivity
    2. For each MST edge, use A* to find terrain-aware path
    3. Classify roads as Imperial Highway or Main Road
    
    Returns:
        List of RoadSegment objects
    """
    if len(major_cities) < 2:
        return []
    
    # Convert settlements to nodes
    nodes = [
        RoadNode(
            settlement_id=s.settlement_id,
            x=s.x,
            y=s.y,
            settlement_type=s.settlement_type
        )
        for s in major_cities
    ]
    
    # PHASE 1: Build MST to determine connectivity
    print(f"        Phase 1: Building MST of major cities...")
    
    mst_edges = build_mst(nodes, terrain_data, size)
    
    print(f"        MST has {len(mst_edges)} edges")
    
    # PHASE 2: For each MST edge, find A* path
    print(f"        Phase 2: Finding terrain-aware paths with A*...")
    
    road_segments = []
    
    for i, (node_a, node_b) in enumerate(mst_edges):
        print(f"          Path {i+1}/{len(mst_edges)}: City {node_a.settlement_id} -> {node_b.settlement_id}")
        
        path_result = find_path_astar(
            (node_a.x, node_a.y),
            (node_b.x, node_b.y),
            terrain_data,
            size,
            allow_water_crossing=True
        )
        
        if path_result is None:
            print(f"            WARNING: No path found!")
            continue
        
        # Determine road type based on settlement importance
        min_type = min(node_a.settlement_type, node_b.settlement_type)
        
        if min_type >= 4:  # Both are metropolises
            road_type = 0  # Imperial Highway
        elif min_type >= 3:  # At least one city
            road_type = 1  # Main Road
        else:
            road_type = 2  # Rural Road
        
        # Create road segment
        segment = RoadSegment(
            start_settlement_id=node_a.settlement_id,
            end_settlement_id=node_b.settlement_id,
            path=path_result.path,
            road_type=road_type,
            length=len(path_result.path),
            cost=path_result.cost,
            crosses_water=path_result.crosses_water,
        )
        
        road_segments.append(segment)
        
        print(f"            ✓ Length: {len(path_result.path)} cells, Cost: {path_result.cost:.1f}, Water: {path_result.crosses_water}")
    
    return road_segments


def build_mst(
    nodes: List[RoadNode],
    terrain_data: Dict,
    size: int
) -> List[Tuple[RoadNode, RoadNode]]:
    """
    Build Minimum Spanning Tree using Prim's algorithm.
    
    Edge weights are Euclidean distances (we'll do terrain-aware routing later).
    
    Returns:
        List of (node_a, node_b) tuples representing MST edges
    """
    if len(nodes) < 2:
        return []
    
    # Start with arbitrary node
    visited = {0}
    edges = []
    
    # Priority queue: (distance, node_idx_a, node_idx_b)
    pq = []
    
    # Add all edges from node 0
    for j in range(1, len(nodes)):
        dist = euclidean_distance(nodes[0], nodes[j])
        heapq.heappush(pq, (dist, 0, j))
    
    # Prim's algorithm
    while pq and len(visited) < len(nodes):
        dist, idx_a, idx_b = heapq.heappop(pq)
        
        if idx_b in visited:
            continue
        
        # Add edge
        edges.append((nodes[idx_a], nodes[idx_b]))
        visited.add(idx_b)
        
        # Add edges from newly visited node
        for j in range(len(nodes)):
            if j not in visited:
                new_dist = euclidean_distance(nodes[idx_b], nodes[j])
                heapq.heappush(pq, (new_dist, idx_b, j))
    
    return edges


def euclidean_distance(node_a: RoadNode, node_b: RoadNode) -> float:
    """Calculate Euclidean distance between two nodes."""
    return np.sqrt((node_a.x - node_b.x)**2 + (node_a.y - node_b.y)**2)


def find_path_astar(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    terrain_data: Dict,
    size: int,
    allow_water_crossing: bool = True
) -> Optional[PathResult]:
    """
    Find path from start to goal using A* with terrain-aware costs.
    
    Cost function considers:
    - Distance (base cost)
    - Slope (steeper = more expensive)
    - Biome difficulty (desert/mountain = hard)
    - Water crossing (expensive but not infinite if allow_water_crossing)
    
    Args:
        start: (x, y) starting position
        goal: (x, y) goal position
        terrain_data: Dict with elevation, biome, slope, water_mask
        size: World size
        allow_water_crossing: If True, allow crossing water at high cost
    
    Returns:
        PathResult or None if no path found
    """
    elevation = terrain_data['elevation']
    biome = terrain_data['biome']
    slope = terrain_data['slope']
    water_mask = terrain_data['water_mask']
    
    # Priority queue: (f_score, g_score, x, y)
    pq = [(0, 0, start[0], start[1])]
    
    # Tracking
    came_from = {}
    g_score = {start: 0}
    
    # Closed set
    closed = set()
    
    # 8-directional movement
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    while pq:
        f, g, x, y = heapq.heappop(pq)
        
        current = (x, y)
        
        if current in closed:
            continue
        
        closed.add(current)
        
        # Goal check
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            
            # Check water crossings
            water_crossings = [pos for pos in path if water_mask[pos[0], pos[1]]]
            crosses_water = len(water_crossings) > 0
            
            return PathResult(
                path=path,
                cost=g,
                crosses_water=crosses_water,
                water_crossings=water_crossings
            )
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Bounds check
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            
            neighbor = (nx, ny)
            
            if neighbor in closed:
                continue
            
            # Calculate movement cost
            move_cost = calculate_terrain_cost(
                x, y, nx, ny,
                elevation, biome, slope, water_mask,
                allow_water_crossing
            )
            
            if move_cost < 0:  # Impassable
                continue
            
            tentative_g = g + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Heuristic: Euclidean distance
                h = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                f_score = tentative_g + h
                
                heapq.heappush(pq, (f_score, tentative_g, nx, ny))
    
    # No path found
    return None


def calculate_terrain_cost(
    x: int, y: int,
    nx: int, ny: int,
    elevation: np.ndarray,
    biome: np.ndarray,
    slope: np.ndarray,
    water_mask: np.ndarray,
    allow_water_crossing: bool
) -> float:
    """
    Calculate movement cost from (x, y) to (nx, ny) based on terrain.
    
    Returns:
        Cost (positive float) or -1 if impassable
    """
    # Base cost (diagonal = sqrt(2), cardinal = 1)
    if abs(nx - x) + abs(ny - y) == 2:
        base_cost = 1.414
    else:
        base_cost = 1.0
    
    # Water crossing
    if water_mask[nx, ny]:
        if not allow_water_crossing:
            return -1  # Impassable
        else:
            # High cost for water crossing (bridge)
            return base_cost * 50.0
    
    # Slope cost (exponential penalty for steep slopes)
    slope_val = slope[nx, ny]
    slope_multiplier = 1.0 + (slope_val * 10.0)  # 0.1 slope = 2x cost
    
    if slope_val > 0.5:  # Very steep (>26 degrees)
        slope_multiplier *= 3.0
    
    # Biome difficulty
    biome_val = BiomeType(biome[nx, ny])
    
    biome_costs = {
        BiomeType.HOT_DESERT: 2.0,
        BiomeType.COLD_DESERT: 2.5,
        BiomeType.TUNDRA: 2.0,
        BiomeType.ICE: 5.0,
        BiomeType.ALPINE: 3.0,
        BiomeType.BOREAL_FOREST: 1.5,
        BiomeType.TROPICAL_RAINFOREST: 2.0,
        BiomeType.SAVANNA: 1.2,
        BiomeType.TEMPERATE_GRASSLAND: 1.0,
        BiomeType.TEMPERATE_DECIDUOUS_FOREST: 1.3,
        BiomeType.TEMPERATE_RAINFOREST: 1.5,
        BiomeType.MEDITERRANEAN: 1.1,
    }
    
    biome_multiplier = biome_costs.get(biome_val, 1.0)
    
    # Elevation change cost
    elev_diff = abs(elevation[nx, ny] - elevation[x, y])
    elev_multiplier = 1.0 + (elev_diff / 100.0)  # 100m = 2x cost
    
    # Total cost
    total_cost = base_cost * slope_multiplier * biome_multiplier * elev_multiplier
    
    return total_cost


def connect_secondary_settlements(
    secondary_settlements: List,
    highway_network: List[RoadSegment],
    major_cities: List,
    terrain_data: Dict,
    size: int,
    world_state: WorldState
) -> List[RoadSegment]:
    """
    Connect towns, villages, and hamlets to the road network.
    
    For each settlement, choose cheaper option:
    1. Connect to nearest highway segment
    2. Connect directly to another settlement
    
    Returns:
        List of RoadSegment objects
    """
    road_segments = []
    
    # Rasterize highway network for distance calculations
    highway_map = rasterize_roads(highway_network, size)
    
    print(f"        Connecting {len(secondary_settlements)} secondary settlements...")
    
    for i, settlement in enumerate(secondary_settlements):
        if (i + 1) % 50 == 0 or i == len(secondary_settlements) - 1:
            print(f"          Progress: {i+1}/{len(secondary_settlements)}")
        
        start_pos = (settlement.x, settlement.y)
        
        # OPTION 1: Find nearest point on highway
        highway_cost = float('inf')
        highway_target = None
        
        if np.any(highway_map):
            distance_to_highway = distance_transform_edt(~highway_map)
            dist_to_highway = distance_to_highway[settlement.x, settlement.y]
            
            # Find nearest highway cell
            if dist_to_highway < size // 4:  # Only consider if reasonably close
                # Find nearest highway point
                highway_points = np.argwhere(highway_map)
                distances = np.sqrt(
                    (highway_points[:, 0] - settlement.x)**2 +
                    (highway_points[:, 1] - settlement.y)**2
                )
                nearest_idx = distances.argmin()
                highway_target = tuple(highway_points[nearest_idx])
                
                # Estimate cost (A* is expensive, so estimate first)
                highway_cost = distances[nearest_idx] * 2.0  # Rough estimate
        
        # OPTION 2: Find nearest other settlement
        settlement_cost = float('inf')
        settlement_target = None
        
        all_other_settlements = [s for s in secondary_settlements if s.settlement_id != settlement.settlement_id] + list(major_cities)
        
        if all_other_settlements:
            # Find nearest settlement
            distances = [
                np.sqrt((s.x - settlement.x)**2 + (s.y - settlement.y)**2)
                for s in all_other_settlements
            ]
            nearest_idx = np.argmin(distances)
            nearest_settlement = all_other_settlements[nearest_idx]
            settlement_target = (nearest_settlement.x, nearest_settlement.y)
            settlement_cost = distances[nearest_idx] * 2.0  # Rough estimate
        
        # Choose cheaper option
        if highway_cost < settlement_cost and highway_target is not None:
            target = highway_target
            target_type = 'highway'
        elif settlement_target is not None:
            target = settlement_target
            target_type = 'settlement'
        else:
            continue
        
        # Run A* to find actual path
        path_result = find_path_astar(
            start_pos,
            target,
            terrain_data,
            size,
            allow_water_crossing=True
        )
        
        if path_result is None:
            continue
        
        # Determine road type
        if settlement.settlement_type >= 2:  # Town
            road_type = 2  # Rural Road
        elif settlement.settlement_type >= 1:  # Village
            road_type = 3  # Path
        else:  # Hamlet
            road_type = 4  # Trail
        
        # Create road segment
        segment = RoadSegment(
            start_settlement_id=settlement.settlement_id,
            end_settlement_id=-1 if target_type == 'highway' else nearest_settlement.settlement_id,
            path=path_result.path,
            road_type=road_type,
            length=len(path_result.path),
            cost=path_result.cost,
            crosses_water=path_result.crosses_water,
        )
        
        road_segments.append(segment)
        
        # Update highway map with new road (so future settlements can connect to it)
        for x, y in path_result.path:
            highway_map[x, y] = True
    
    return road_segments


def rasterize_roads(roads: List[RoadSegment], size: int) -> np.ndarray:
    """
    Convert road segments to boolean raster map.
    
    Returns:
        bool[size, size] - True where roads exist
    """
    road_map = np.zeros((size, size), dtype=bool)
    
    for road in roads:
        for x, y in road.path:
            if 0 <= x < size and 0 <= y < size:
                road_map[x, y] = True
    
    return road_map


def rasterize_road_types(roads: List[RoadSegment], size: int) -> np.ndarray:
    """
    Convert road segments to road type map.
    
    Road types:
    0 = Imperial Highway (metropolis connections)
    1 = Main Road (city connections)
    2 = Rural Road (town connections)
    3 = Path (village connections)
    4 = Trail (hamlet connections)
    
    Returns:
        uint8[size, size] - Road type (0 = no road, 1-5 = road types)
    """
    road_type_map = np.zeros((size, size), dtype=np.uint8)
    
    # Sort roads by type (lower type = higher priority for overlaps)
    sorted_roads = sorted(roads, key=lambda r: r.road_type, reverse=True)
    
    for road in sorted_roads:
        for x, y in road.path:
            if 0 <= x < size and 0 <= y < size:
                # Only overwrite if current value is lower priority
                if road_type_map[x, y] == 0 or road.road_type < road_type_map[x, y] - 1:
                    road_type_map[x, y] = road.road_type + 1  # +1 so 0 = no road
    
    return road_type_map


def identify_bridges(roads: List[RoadSegment], elevation: np.ndarray) -> List[Bridge]:
    """
    Identify all bridge locations in the road network.
    
    A bridge is a road segment that crosses water.
    
    Returns:
        List of Bridge objects
    """
    bridges = []
    bridge_id = 0
    
    for road in roads:
        if not road.crosses_water:
            continue
        
        # Find continuous water crossings
        in_water = False
        bridge_start = None
        bridge_points = []
        
        for x, y in road.path:
            is_water = elevation[x, y] <= 0
            
            if is_water and not in_water:
                # Start of bridge
                in_water = True
                bridge_start = (x, y)
                bridge_points = [(x, y)]
            elif is_water and in_water:
                # Continue bridge
                bridge_points.append((x, y))
            elif not is_water and in_water:
                # End of bridge
                in_water = False
                
                # Create bridge
                if len(bridge_points) > 0:
                    # Use midpoint as bridge location
                    mid_idx = len(bridge_points) // 2
                    bridge_x, bridge_y = bridge_points[mid_idx]
                    
                    bridge = Bridge(
                        bridge_id=bridge_id,
                        x=bridge_x,
                        y=bridge_y,
                        length=len(bridge_points),
                        road_type=road.road_type,
                    )
                    bridges.append(bridge)
                    bridge_id += 1
                
                bridge_points = []
    
    return bridges


def calculate_average_travel_time(
    settlements: List,
    roads: List[RoadSegment],
    size: int
) -> float:
    """
    Calculate average travel time from any settlement to nearest major city.
    
    Uses simplified model: assume 25 km/h average speed on roads.
    
    Returns:
        Average time in hours
    """
    # Rasterize roads
    road_map = rasterize_roads(roads, size)
    
    # Distance transform to nearest road
    distance_to_road = distance_transform_edt(~road_map)
    
    # Calculate travel times
    travel_times = []
    
    for settlement in settlements:
        # Distance to nearest road (in cells)
        dist_to_road = distance_to_road[settlement.x, settlement.y]
        
        # Assume 1 cell = ~100m, walking speed = 5 km/h
        walk_time = (dist_to_road * 0.1) / 5.0  # hours
        
        # Assume average road travel of 50km to major city at 25 km/h
        road_time = 50.0 / 25.0  # 2 hours
        
        total_time = walk_time + road_time
        travel_times.append(total_time)
    
    return np.mean(travel_times) if travel_times else 0.0