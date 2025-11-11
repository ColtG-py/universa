"""
World Builder - Pass 15: Ley Lines & Magic
Generates magical energy networks, mana concentrations, and enchanted locations.

APPROACH:
- Use tectonic stress as base for ley line probability
- Water sources channel and amplify magic
- Generate ley line network connecting high-energy points
- Calculate mana concentration with distance-based falloff
- Identify intersection nodes as power nexuses
- Mark corrupted zones and enchanted locations
- Assign elemental affinities based on terrain
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import List, Tuple, Dict, Set
import heapq

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState, LeyLineSegment, EnchantedLocation


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate magical energy network and enchanted locations.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating magical energy network...")
    
    size = world_state.size
    seed = params.seed
    rng = np.random.default_rng(seed + 15000)
    
    # STEP 1: Calculate base magical potential from tectonic stress and water
    print(f"    - Calculating magical potential field...")
    
    magic_potential = calculate_magic_potential(world_state, size)
    
    # STEP 2: Identify high-energy anchor points for ley line network
    print(f"    - Identifying magical anchor points...")
    
    anchor_points = identify_anchor_points(magic_potential, size, rng)
    print(f"      Found {len(anchor_points)} anchor points")
    
    # STEP 3: Generate ley line network connecting anchor points
    print(f"    - Generating ley line network...")
    
    ley_lines, ley_line_map = generate_ley_line_network(
        anchor_points,
        magic_potential,
        size,
        rng
    )
    print(f"      Generated {len(ley_lines)} ley line segments")
    
    # STEP 4: Calculate mana concentration based on proximity to ley lines
    print(f"    - Calculating mana concentration field...")
    
    mana_concentration = calculate_mana_concentration(
        ley_line_map,
        magic_potential,
        size
    )
    
    # STEP 5: Identify ley line nodes (intersections)
    print(f"    - Identifying ley line intersection nodes...")
    
    ley_line_nodes = identify_ley_line_nodes(ley_line_map, mana_concentration, size)
    print(f"      Found {len(ley_line_nodes)} ley line nodes")
    
    # STEP 6: Mark corrupted zones (0.5% of land area)
    print(f"    - Marking corrupted magic zones...")
    
    corrupted_zones = generate_corrupted_zones(world_state, size, rng)
    num_corrupted = corrupted_zones.sum()
    print(f"      Marked {num_corrupted} corrupted cells")
    
    # STEP 7: Identify enchanted locations
    print(f"    - Identifying enchanted locations...")
    
    enchanted_locations = identify_enchanted_locations(
        world_state,
        mana_concentration,
        ley_line_nodes,
        corrupted_zones,
        size,
        rng
    )
    print(f"      Identified {len(enchanted_locations)} enchanted locations")
    
    # STEP 8: Determine elemental affinities based on terrain
    print(f"    - Calculating elemental affinities...")
    
    elemental_affinity = calculate_elemental_affinity(world_state, size)
    
    # STEP 9: Store results in chunks
    print(f"    - Storing magic data in chunks...")
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.mana_concentration = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.ley_line_presence = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            chunk.ley_line_node = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            chunk.corrupted_zone = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            chunk.elemental_affinity = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.mana_concentration[local_x, local_y] = mana_concentration[global_x, global_y]
                        chunk.ley_line_presence[local_x, local_y] = ley_line_map[global_x, global_y]
                        chunk.corrupted_zone[local_x, local_y] = corrupted_zones[global_x, global_y]
                        chunk.elemental_affinity[local_x, local_y] = elemental_affinity[global_x, global_y]
                        
                        # Mark nodes
                        if (global_x, global_y) in ley_line_nodes:
                            chunk.ley_line_node[local_x, local_y] = True
            
            # Store enchanted locations for this chunk
            chunk.enchanted_locations = [
                loc for loc in enchanted_locations
                if loc.chunk_x == chunk_x and loc.chunk_y == chunk_y
            ]
    
    # Store ley line network at world level
    world_state.ley_line_network = ley_lines
    
    # STEP 10: Calculate and display statistics
    land_mask = calculate_land_mask(world_state, size)
    land_cells = land_mask.sum()
    
    mana_on_land = mana_concentration[land_mask]
    
    print(f"  - Magic system statistics:")
    print(f"    Mana concentration mean: {mana_on_land.mean():.3f}")
    print(f"    Mana concentration median: {np.median(mana_on_land):.3f}")
    print(f"    High magic zones (>0.6): {(mana_on_land > 0.6).sum():,} ({(mana_on_land > 0.6).sum() / land_cells * 100:.1f}%)")
    print(f"    Very high magic zones (>0.8): {(mana_on_land > 0.8).sum():,} ({(mana_on_land > 0.8).sum() / land_cells * 100:.1f}%)")
    print(f"    Dead zones (<0.2): {(mana_on_land < 0.2).sum():,} ({(mana_on_land < 0.2).sum() / land_cells * 100:.1f}%)")
    print(f"    Ley line network length: {len(ley_lines)} segments")
    print(f"    Ley line nodes: {len(ley_line_nodes)}")
    print(f"    Corrupted zones: {num_corrupted:,} cells")
    print(f"    Enchanted locations: {len(enchanted_locations)}")


def calculate_magic_potential(world_state: WorldState, size: int) -> np.ndarray:
    """
    Calculate base magical potential from tectonic stress and water sources.
    
    Magic follows geological stress and is amplified by water.
    
    Args:
        world_state: World state with tectonic and hydrological data
        size: World size
    
    Returns:
        Magic potential map (0-1 scale)
    """
    potential = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    # Base potential from tectonic stress (primary source)
                    stress = chunk.tectonic_stress[local_x, local_y] if chunk.tectonic_stress is not None else 0.0
                    base_potential = stress * 0.7
                    
                    # Water amplifies magic
                    water_bonus = 0.0
                    if chunk.river_presence is not None and chunk.river_presence[local_x, local_y]:
                        water_bonus = 0.2
                    elif chunk.water_table_depth is not None:
                        water_depth = chunk.water_table_depth[local_x, local_y]
                        if water_depth < 10.0:  # Shallow water table
                            water_bonus = 0.1 * (1.0 - water_depth / 10.0)
                    
                    # Elevation influences magic (peaks and valleys)
                    elevation_bonus = 0.0
                    if chunk.elevation is not None:
                        elev = chunk.elevation[local_x, local_y]
                        if elev > 3000:  # High peaks
                            elevation_bonus = 0.15 * min(1.0, (elev - 3000) / 3000)
                        elif elev < -2000:  # Deep ocean trenches
                            elevation_bonus = 0.1 * min(1.0, abs(elev + 2000) / 2000)
                    
                    # Combine sources
                    total_potential = min(1.0, base_potential + water_bonus + elevation_bonus)
                    potential[global_x, global_y] = total_potential
    
    # Smooth potential field slightly
    potential = gaussian_filter(potential, sigma=2.0)
    
    return potential


def identify_anchor_points(
    magic_potential: np.ndarray,
    size: int,
    rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """
    Identify high-energy points to anchor the ley line network.
    
    Args:
        magic_potential: Magic potential map
        size: World size
        rng: Random generator
    
    Returns:
        List of (x, y) anchor point coordinates
    """
    # Use high percentile threshold
    threshold = np.percentile(magic_potential, 90)
    
    # Find all high-potential cells
    candidates = np.argwhere(magic_potential > threshold)
    
    # Select anchor points with minimum spacing
    anchors = []
    min_distance = size // 20  # About 25-50 cells apart
    
    # Sort by potential (highest first)
    candidates = sorted(
        candidates,
        key=lambda p: magic_potential[p[0], p[1]],
        reverse=True
    )
    
    for x, y in candidates:
        # Check if far enough from existing anchors
        too_close = False
        for ax, ay in anchors:
            dist = np.sqrt((x - ax)**2 + (y - ay)**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            anchors.append((x, y))
    
    return anchors


def generate_ley_line_network(
    anchor_points: List[Tuple[int, int]],
    magic_potential: np.ndarray,
    size: int,
    rng: np.random.Generator
) -> Tuple[List[LeyLineSegment], np.ndarray]:
    """
    Generate ley line network connecting anchor points.
    
    Uses A* pathfinding with magic potential as edge weights.
    
    Args:
        anchor_points: List of anchor point coordinates
        magic_potential: Magic potential map
        size: World size
        rng: Random generator
    
    Returns:
        Tuple of (ley line segments, ley line presence map)
    """
    if len(anchor_points) < 2:
        return [], np.zeros((size, size), dtype=bool)
    
    ley_lines = []
    ley_line_map = np.zeros((size, size), dtype=bool)
    
    # Build minimum spanning tree with some extra connections
    # This creates a network rather than just a tree
    
    # First, create MST
    connected = {anchor_points[0]}
    unconnected = set(anchor_points[1:])
    
    while unconnected:
        # Find nearest unconnected point to any connected point
        best_distance = float('inf')
        best_pair = None
        
        for connected_point in connected:
            for unconnected_point in unconnected:
                dist = np.sqrt(
                    (connected_point[0] - unconnected_point[0])**2 +
                    (connected_point[1] - unconnected_point[1])**2
                )
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (connected_point, unconnected_point)
        
        if best_pair:
            start, end = best_pair
            
            # Generate path using A* with magic potential
            path = astar_path(start, end, magic_potential, size)
            
            if path:
                # Create ley line segment
                segment = LeyLineSegment(
                    start_x=start[0],
                    start_y=start[1],
                    end_x=end[0],
                    end_y=end[1],
                    path_points=path,
                    strength=magic_potential[end[0], end[1]]
                )
                ley_lines.append(segment)
                
                # Mark cells in ley line map
                for x, y in path:
                    ley_line_map[x, y] = True
            
            connected.add(end)
            unconnected.remove(end)
    
    # Add some extra connections for network redundancy (20% more edges)
    num_extra = max(1, len(anchor_points) // 5)
    
    for _ in range(num_extra):
        # Pick two random connected points
        if len(connected) < 2:
            break
        
        points = list(connected)
        rng.shuffle(points)
        start, end = points[0], points[1]
        
        # Only connect if not too close
        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        if dist > size / 10:
            path = astar_path(start, end, magic_potential, size)
            
            if path:
                segment = LeyLineSegment(
                    start_x=start[0],
                    start_y=start[1],
                    end_x=end[0],
                    end_y=end[1],
                    path_points=path,
                    strength=min(magic_potential[start[0], start[1]], magic_potential[end[0], end[1]])
                )
                ley_lines.append(segment)
                
                for x, y in path:
                    ley_line_map[x, y] = True
    
    return ley_lines, ley_line_map


def astar_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    magic_potential: np.ndarray,
    size: int
) -> List[Tuple[int, int]]:
    """
    A* pathfinding for ley lines.
    
    Prefers paths through high magic potential areas.
    
    Args:
        start: Start coordinates
        end: End coordinates
        magic_potential: Magic potential map (higher = better path)
        size: World size
    
    Returns:
        List of path coordinates
    """
    def heuristic(pos):
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    
    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbors.append((nx, ny))
        return neighbors
    
    # Priority queue: (f_score, g_score, position)
    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        
        for neighbor in get_neighbors(current):
            # Cost is inversely related to magic potential (prefer high magic)
            # Lower cost = preferred path
            edge_cost = 1.0 - magic_potential[neighbor[0], neighbor[1]] * 0.5
            
            tentative_g = current_g + edge_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    # No path found - use straight line
    return [start, end]


def calculate_mana_concentration(
    ley_line_map: np.ndarray,
    magic_potential: np.ndarray,
    size: int
) -> np.ndarray:
    """
    Calculate mana concentration based on proximity to ley lines.
    
    Mana is highest on ley lines and falls off with distance.
    
    Args:
        ley_line_map: Boolean map of ley line locations
        magic_potential: Base magic potential
        size: World size
    
    Returns:
        Mana concentration map (0-1 scale)
    """
    # Calculate distance from each cell to nearest ley line
    distance_to_ley_line = distance_transform_edt(~ley_line_map)
    
    # Mana falls off exponentially with distance
    # Max influence distance: ~50 cells
    max_distance = 50.0
    distance_factor = np.exp(-distance_to_ley_line / max_distance)
    
    # Combine base potential with ley line proximity
    mana_concentration = magic_potential * 0.4 + distance_factor * 0.6
    
    # Boost on actual ley lines
    mana_concentration[ley_line_map] = np.maximum(
        mana_concentration[ley_line_map],
        0.7
    )
    
    # Smooth for natural gradients
    mana_concentration = gaussian_filter(mana_concentration, sigma=3.0)
    
    # Normalize to [0, 1]
    if mana_concentration.max() > 0:
        mana_concentration = mana_concentration / mana_concentration.max()
    
    return mana_concentration


def identify_ley_line_nodes(
    ley_line_map: np.ndarray,
    mana_concentration: np.ndarray,
    size: int
) -> Set[Tuple[int, int]]:
    """
    Identify ley line intersection nodes (high power nexuses).
    
    Args:
        ley_line_map: Ley line presence map
        mana_concentration: Mana concentration map
        size: World size
    
    Returns:
        Set of node coordinates
    """
    nodes = set()
    
    # Find cells where ley lines intersect (8-connected neighborhood)
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            if not ley_line_map[x, y]:
                continue
            
            # Count ley line neighbors
            neighbors = [
                ley_line_map[x-1, y], ley_line_map[x+1, y],
                ley_line_map[x, y-1], ley_line_map[x, y+1],
                ley_line_map[x-1, y-1], ley_line_map[x-1, y+1],
                ley_line_map[x+1, y-1], ley_line_map[x+1, y+1]
            ]
            
            num_neighbors = sum(neighbors)
            
            # Node if 3+ ley line neighbors (intersection point)
            if num_neighbors >= 3:
                # Also require high mana concentration
                if mana_concentration[x, y] > 0.7:
                    nodes.add((x, y))
    
    return nodes


def generate_corrupted_zones(
    world_state: WorldState,
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate corrupted magic zones (0.5% of land area).
    
    Corrupted zones are areas warped by excessive or misused magic.
    
    Args:
        world_state: World state
        size: World size
        rng: Random generator
    
    Returns:
        Boolean array of corrupted zones
    """
    corrupted = np.zeros((size, size), dtype=bool)
    
    land_mask = calculate_land_mask(world_state, size)
    land_cells = land_mask.sum()
    
    # Target 0.5% of land
    num_corrupted_centers = max(3, int(land_cells * 0.005 / 100))
    
    # Find random land cells for corruption centers
    land_coords = np.argwhere(land_mask)
    
    if len(land_coords) == 0:
        return corrupted
    
    # Select random centers
    center_indices = rng.choice(len(land_coords), size=num_corrupted_centers, replace=False)
    centers = land_coords[center_indices]
    
    # Spread corruption in irregular patches
    for cx, cy in centers:
        # Random corruption radius
        radius = rng.integers(10, 30)
        
        for x in range(max(0, cx - radius), min(size, cx + radius)):
            for y in range(max(0, cy - radius), min(size, cy + radius)):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Probability decreases with distance
                if dist < radius:
                    prob = 1.0 - (dist / radius)
                    if rng.random() < prob:
                        corrupted[x, y] = True
    
    return corrupted


def identify_enchanted_locations(
    world_state: WorldState,
    mana_concentration: np.ndarray,
    ley_line_nodes: Set[Tuple[int, int]],
    corrupted_zones: np.ndarray,
    size: int,
    rng: np.random.Generator
) -> List[EnchantedLocation]:
    """
    Identify special enchanted locations.
    
    Args:
        world_state: World state
        mana_concentration: Mana concentration map
        ley_line_nodes: Ley line intersection nodes
        corrupted_zones: Corrupted zone map
        size: World size
        rng: Random generator
    
    Returns:
        List of enchanted locations
    """
    locations = []
    
    num_chunks = size // CHUNK_SIZE
    
    # Collect biome data
    biome_map = np.zeros((size, size), dtype=np.uint8)
    elevation_map = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.biome_type is not None:
                biome_map[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
            if chunk.elevation is not None:
                elevation_map[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
    
    # 1. Mana Wells at ley line nodes
    for x, y in ley_line_nodes:
        if mana_concentration[x, y] > 0.8:  # Very high mana
            chunk_x, chunk_y = x // CHUNK_SIZE, y // CHUNK_SIZE
            locations.append(EnchantedLocation(
                location_type="mana_well",
                location_x=x,
                location_y=y,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                power_level=mana_concentration[x, y],
                properties={"node": True}
            ))
    
    # 2. Fey Groves in ancient forests with high mana
    from config import BiomeType
    
    for _ in range(5):  # Place 5 fey groves
        # Find forest cells with high mana
        forest_mask = np.isin(biome_map, [
            BiomeType.TEMPERATE_DECIDUOUS_FOREST,
            BiomeType.TEMPERATE_RAINFOREST,
            BiomeType.TROPICAL_RAINFOREST,
            BiomeType.BOREAL_FOREST
        ])
        
        high_mana_mask = mana_concentration > 0.6
        valid_mask = forest_mask & high_mana_mask & (elevation_map > 0)
        
        candidates = np.argwhere(valid_mask)
        if len(candidates) == 0:
            continue
        
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        chunk_x, chunk_y = x // CHUNK_SIZE, y // CHUNK_SIZE
        
        locations.append(EnchantedLocation(
            location_type="fey_grove",
            location_x=x,
            location_y=y,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            power_level=mana_concentration[x, y],
            properties={"biome": int(biome_map[x, y])}
        ))
    
    # 3. Dragon Lairs in high mountains with high mana
    for _ in range(3):  # Place 3 dragon lairs
        mountain_mask = elevation_map > np.percentile(elevation_map[elevation_map > 0], 90)
        high_mana_mask = mana_concentration > 0.7
        valid_mask = mountain_mask & high_mana_mask
        
        candidates = np.argwhere(valid_mask)
        if len(candidates) == 0:
            continue
        
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        chunk_x, chunk_y = x // CHUNK_SIZE, y // CHUNK_SIZE
        
        locations.append(EnchantedLocation(
            location_type="dragon_lair",
            location_x=x,
            location_y=y,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            power_level=mana_concentration[x, y],
            properties={"elevation": float(elevation_map[x, y])}
        ))
    
    # 4. Corrupted Sites
    corrupted_coords = np.argwhere(corrupted_zones)
    if len(corrupted_coords) > 0:
        # Pick a few centers
        for _ in range(min(5, len(corrupted_coords) // 50)):
            idx = rng.integers(0, len(corrupted_coords))
            x, y = corrupted_coords[idx]
            chunk_x, chunk_y = x // CHUNK_SIZE, y // CHUNK_SIZE
            
            locations.append(EnchantedLocation(
                location_type="corrupted_site",
                location_x=x,
                location_y=y,
                chunk_x=chunk_x,
                chunk_y=chunk_y,
                power_level=0.1,  # Low/negative magic
                properties={"corrupted": True}
            ))
    
    return locations


def calculate_elemental_affinity(world_state: WorldState, size: int) -> np.ndarray:
    """
    Calculate elemental affinity based on terrain.
    
    Elemental types:
    0 = None
    1 = Fire (deserts, volcanoes, hot areas)
    2 = Water (coasts, rivers, wet areas)
    3 = Earth (mountains, caves, stable ground)
    4 = Air (peaks, windy areas, high elevation)
    5 = Arcane (mixed/neutral)
    
    Args:
        world_state: World state
        size: World size
    
    Returns:
        Elemental affinity map
    """
    affinity = np.zeros((size, size), dtype=np.uint8)
    
    num_chunks = size // CHUNK_SIZE
    
    from config import BiomeType
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    # Default to Arcane
                    element = 5
                    
                    # Determine by biome and terrain
                    if chunk.biome_type is not None:
                        biome = chunk.biome_type[local_x, local_y]
                        
                        # Fire: Hot, dry biomes
                        if biome in [BiomeType.HOT_DESERT, BiomeType.SAVANNA]:
                            element = 1
                        
                        # Water: Wet biomes and coasts
                        elif biome in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_RAINFOREST, BiomeType.MANGROVE]:
                            element = 2
                        
                        # Earth: Forests and mountains
                        elif biome in [BiomeType.BOREAL_FOREST, BiomeType.TEMPERATE_DECIDUOUS_FOREST]:
                            element = 3
                        
                        # Air: High, cold biomes
                        elif biome in [BiomeType.ALPINE, BiomeType.TUNDRA]:
                            element = 4
                    
                    # Override by elevation
                    if chunk.elevation is not None:
                        elev = chunk.elevation[local_x, local_y]
                        
                        if elev > 3500:  # Very high peaks - Air
                            element = 4
                        elif elev > 2500:  # Mountains - Earth
                            element = 3
                    
                    # Override by water presence
                    if chunk.river_presence is not None and chunk.river_presence[local_x, local_y]:
                        element = 2  # Water
                    
                    affinity[global_x, global_y] = element
    
    return affinity


def calculate_land_mask(world_state: WorldState, size: int) -> np.ndarray:
    """Helper to get land mask from elevation."""
    land_mask = np.zeros((size, size), dtype=bool)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None or chunk.elevation is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        if chunk.elevation[local_x, local_y] > 0:
                            land_mask[global_x, global_y] = True
    
    return land_mask