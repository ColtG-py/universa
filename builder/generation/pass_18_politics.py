"""
World Builder - Pass 18: Political Boundaries
Divides the world into kingdoms, territories, and regions based on geography and settlements.

SCIENTIFIC BASIS:
- Settlements naturally form political hierarchies
- Geographic features (rivers, mountains) create natural borders
- Resource-rich areas drive territorial competition
- City size determines political influence

POLITICAL HIERARCHY:
1. Kingdom: 5-15 cities, 50+ settlements, organized feudal state
2. Duchy: 2-5 cities, 15-30 settlements, semi-autonomous region
3. County: 0-1 city, 5-10 settlements, ruled by a noble
4. Free City: Independent city-state
5. Tribal Confederation: Nomadic/semi-nomadic peoples
6. Theocracy: Religious state controlled by clergy

TERRITORY GENERATION:
1. Capital Selection: Largest cities become faction capitals
2. Influence Expansion: Weighted Voronoi based on city size
3. Natural Borders: Snap to rivers, mountains, coastlines
4. Resource Competition: Mark contested zones
5. Vassal Assignment: Feudal hierarchy
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType
from models.world import WorldState, Faction, FactionType, BorderType


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate political boundaries and factions based on settlements.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating political boundaries and factions...")
    
    size = world_state.size
    seed = params.seed
    rng = np.random.default_rng(seed + 18000)
    
    # STEP 1: Collect all settlements
    print(f"    - Collecting settlements...")
    
    if not hasattr(world_state, 'settlements') or not world_state.settlements:
        print(f"    ⚠️  No settlements found! Run Pass 16 first.")
        return
    
    settlements = world_state.settlements
    
    # Filter out ruins for faction assignment
    active_settlements = [s for s in settlements if not s.is_ruin]
    
    print(f"      Total settlements: {len(settlements)}")
    print(f"      Active settlements: {len(active_settlements)}")
    
    # STEP 2: Collect geographic data
    print(f"    - Collecting geographic data for border calculation...")
    
    elevation_global, river_global, biome_global = collect_geographic_data(world_state, size)
    land_mask = elevation_global > 0
    
    # Calculate slope for mountain detection
    from utils.spatial import calculate_slope
    slope_global = calculate_slope(elevation_global)
    
    # STEP 3: Select faction capitals
    print(f"    - Selecting faction capitals...")
    
    factions = create_factions(active_settlements, size, rng)
    
    print(f"      Created {len(factions)} factions:")
    for faction in factions:
        faction_type_name = FactionType(faction.faction_type).name if isinstance(faction.faction_type, int) else faction.faction_type.name
        print(f"        {faction.name} ({faction_type_name}): {faction.num_settlements} settlements")
    
    # STEP 4: Assign territories using weighted Voronoi
    print(f"    - Calculating territorial influence (weighted Voronoi)...")
    
    faction_territory_map = calculate_weighted_voronoi(
        factions,
        active_settlements,
        size,
        land_mask
    )
    
    # STEP 5: Snap borders to natural features
    print(f"    - Snapping borders to natural features...")
    
    border_type_map = np.zeros((size, size), dtype=np.uint8)
    
    faction_territory_map, border_type_map = snap_to_natural_borders(
        faction_territory_map,
        elevation_global,
        slope_global,
        river_global,
        land_mask,
        size
    )
    
    # STEP 6: Identify contested zones
    print(f"    - Identifying contested resource zones...")
    
    contested_zones = identify_contested_zones(
        faction_territory_map,
        world_state,
        factions,
        size,
        rng
    )
    
    num_contested = contested_zones.sum()
    print(f"      Marked {num_contested:,} cells as contested ({num_contested / land_mask.sum() * 100:.2f}% of land)")
    
    # STEP 7: Assign settlements to factions and create vassalage
    print(f"    - Assigning settlements to factions and creating vassalage...")
    
    assign_settlements_to_factions(
        settlements,
        faction_territory_map,
        factions
    )
    
    create_vassalage_relationships(
        settlements,
        factions
    )
    
    # STEP 8: Update faction statistics
    print(f"    - Calculating faction statistics...")
    
    for faction in factions:
        faction_settlements = [s for s in active_settlements if s.faction_id == faction.faction_id]
        
        if faction_settlements:
            faction.num_settlements = len(faction_settlements)
            faction.total_population = sum(s.population for s in faction_settlements)
            
            # Count cities
            from generation.pass_16_settlements import SettlementType
            faction.num_cities = sum(1 for s in faction_settlements 
                                    if s.settlement_type >= SettlementType.CITY)
            
            # Calculate territory size
            faction_cells = (faction_territory_map == faction.faction_id).sum()
            faction.territory_size_km2 = faction_cells * 0.01  # Assuming 100m per cell = 0.01 km²
    
    # STEP 9: Store results in chunks
    print(f"    - Storing political data in chunks...")
    
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.faction_territory = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
            chunk.border_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.contested_zone = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    chunk.faction_territory[local_x, local_y] = faction_territory_map[global_x, global_y]
                    chunk.border_type[local_x, local_y] = border_type_map[global_x, global_y]
                    chunk.contested_zone[local_x, local_y] = contested_zones[global_x, global_y]
    
    # Store factions at world level
    world_state.factions = factions
    
    # STEP 10: Report statistics
    print(f"  - Political boundaries generated:")
    print(f"    Total factions: {len(factions)}")
    
    faction_type_counts = {}
    for faction in factions:
        faction_type_counts[faction.faction_type] = faction_type_counts.get(faction.faction_type, 0) + 1

    for faction_type, count in sorted(faction_type_counts.items(), key=lambda x: x[1], reverse=True):
        # Convert integer back to enum for display
        faction_type_name = FactionType(faction_type).name if isinstance(faction_type, int) else faction_type.name
        print(f"      {faction_type_name}s: {count}")
    
    # Calculate border statistics
    natural_borders = (border_type_map == BorderType.RIVER).sum() + (border_type_map == BorderType.MOUNTAIN).sum()
    political_borders = (border_type_map == BorderType.POLITICAL).sum()
    total_borders = natural_borders + political_borders
    
    if total_borders > 0:
        print(f"    Border composition:")
        print(f"      Natural borders: {natural_borders / total_borders * 100:.1f}%")
        print(f"      Political borders: {political_borders / total_borders * 100:.1f}%")
    
    # Vassalage statistics
    total_vassals = sum(len(faction.vassal_faction_ids) for faction in factions)
    print(f"    Vassalage relationships: {total_vassals}")


def collect_geographic_data(world_state: WorldState, size: int) -> tuple:
    """Collect elevation, rivers, and biome data."""
    elevation = np.zeros((size, size), dtype=np.float32)
    rivers = np.zeros((size, size), dtype=bool)
    biomes = np.zeros((size, size), dtype=np.uint8)
    
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
            if chunk.river_presence is not None:
                rivers[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
            if chunk.biome_type is not None:
                biomes[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
    
    return elevation, rivers, biomes


def create_factions(settlements: List, size: int, rng: np.random.Generator) -> List[Faction]:
    """
    Create factions by selecting capitals and grouping settlements.
    
    Strategy:
    1. Largest cities become kingdom capitals
    2. Medium cities become duchy/county capitals
    3. Some cities become independent (free cities, theocracies)
    4. Tribal areas for low-density regions
    """
    from generation.pass_16_settlements import SettlementType, SettlementSpecialization
    
    factions = []
    faction_id_counter = 1
    
    # Separate settlements by type
    metropolises = [s for s in settlements if s.settlement_type == SettlementType.METROPOLIS]
    cities = [s for s in settlements if s.settlement_type == SettlementType.CITY]
    towns = [s for s in settlements if s.settlement_type == SettlementType.TOWN]
    
    # Sort by population
    metropolises.sort(key=lambda s: s.population, reverse=True)
    cities.sort(key=lambda s: s.population, reverse=True)
    
    print(f"      Settlement distribution:")
    print(f"        Metropolises: {len(metropolises)}")
    print(f"        Cities: {len(cities)}")
    print(f"        Towns: {len(towns)}")
    
    # KINGDOMS - From metropolises and largest cities
    kingdom_candidates = metropolises + cities[:max(3, len(cities) // 3)]
    
    # Target: ~5-8 kingdoms for 512x512 world, scales with area
    scale_factor = (size / 512.0) ** 2
    num_kingdoms = max(3, min(10, int(6 * scale_factor)))
    
    # Ensure spatial distribution - avoid clustering
    kingdoms_created = []
    min_distance_between_kingdoms = size / (num_kingdoms ** 0.5) * 0.8
    
    for capital_candidate in kingdom_candidates:
        if len(kingdoms_created) >= num_kingdoms:
            break
        
        # Check distance to existing kingdoms
        too_close = False
        for existing_faction in kingdoms_created:
            dist = np.sqrt(
                (capital_candidate.x - existing_faction.capital_x)**2 +
                (capital_candidate.y - existing_faction.capital_y)**2
            )
            if dist < min_distance_between_kingdoms:
                too_close = True
                break
        
        if too_close:
            continue
        
        # Create kingdom
        faction = Faction(
            faction_id=faction_id_counter,
            name=generate_faction_name(faction_id_counter, FactionType.KINGDOM, rng),
            faction_type=FactionType.KINGDOM,
            capital_settlement_id=capital_candidate.settlement_id,
            capital_x=capital_candidate.x,
            capital_y=capital_candidate.y,
        )
        
        factions.append(faction)
        kingdoms_created.append(faction)
        capital_candidate.faction_id = faction_id_counter
        capital_candidate.is_capital = True
        faction_id_counter += 1
    
    print(f"      Created {len(kingdoms_created)} kingdoms")
    
    # DUCHIES - From remaining large cities
    remaining_cities = [c for c in cities if not hasattr(c, 'faction_id') or c.faction_id is None]
    num_duchies = min(len(remaining_cities), int(len(kingdoms_created) * 1.5))
    
    for capital_candidate in remaining_cities[:num_duchies]:
        faction = Faction(
            faction_id=faction_id_counter,
            name=generate_faction_name(faction_id_counter, FactionType.DUCHY, rng),
            faction_type=FactionType.DUCHY,
            capital_settlement_id=capital_candidate.settlement_id,
            capital_x=capital_candidate.x,
            capital_y=capital_candidate.y,
        )
        
        factions.append(faction)
        capital_candidate.faction_id = faction_id_counter
        capital_candidate.is_capital = True
        faction_id_counter += 1
    
    print(f"      Created {num_duchies} duchies")
    
    # FREE CITIES - Some independent cities
    independent_candidates = [c for c in cities if not hasattr(c, 'faction_id') or c.faction_id is None]
    num_free_cities = min(3, len(independent_candidates) // 4)
    
    for capital_candidate in independent_candidates[:num_free_cities]:
        faction = Faction(
            faction_id=faction_id_counter,
            name=f"{capital_candidate.name or 'Free City'} Republic",
            faction_type=FactionType.FREE_CITY,
            capital_settlement_id=capital_candidate.settlement_id,
            capital_x=capital_candidate.x,
            capital_y=capital_candidate.y,
        )
        
        factions.append(faction)
        capital_candidate.faction_id = faction_id_counter
        capital_candidate.is_capital = True
        faction_id_counter += 1
    
    print(f"      Created {num_free_cities} free cities")
    
    # THEOCRACIES - Religious settlements
    religious_settlements = [
        s for s in settlements
        if s.specialization == SettlementSpecialization.RELIGIOUS
        and s.settlement_type >= SettlementType.TOWN
        and (not hasattr(s, 'faction_id') or s.faction_id is None)
    ]
    
    num_theocracies = min(2, len(religious_settlements))
    
    for capital_candidate in religious_settlements[:num_theocracies]:
        faction = Faction(
            faction_id=faction_id_counter,
            name=generate_faction_name(faction_id_counter, FactionType.THEOCRACY, rng),
            faction_type=FactionType.THEOCRACY,
            capital_settlement_id=capital_candidate.settlement_id,
            capital_x=capital_candidate.x,
            capital_y=capital_candidate.y,
        )
        
        factions.append(faction)
        capital_candidate.faction_id = faction_id_counter
        capital_candidate.is_capital = True
        faction_id_counter += 1
    
    print(f"      Created {num_theocracies} theocracies")
    
    return factions


def calculate_weighted_voronoi(
    factions: List[Faction],
    settlements: List,
    size: int,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    Calculate territorial control using weighted Voronoi diagram.
    
    Larger factions (by population) have stronger influence.
    """
    territory_map = np.zeros((size, size), dtype=np.uint16)
    
    # For each land cell, calculate which faction has strongest influence
    print(f"      Calculating influence for {len(factions)} factions...")
    
    # Build influence map for each faction
    faction_influence_maps = {}
    
    for faction in factions:
        influence_map = np.zeros((size, size), dtype=np.float32)
        
        # Get settlements belonging to this faction
        faction_settlements = [s for s in settlements if hasattr(s, 'faction_id') and s.faction_id == faction.faction_id]
        
        if not faction_settlements:
            # Use capital only
            influence_map[faction.capital_x, faction.capital_y] = 100.0
        else:
            # Add influence from all faction settlements
            for settlement in faction_settlements:
                # Influence strength based on population (logarithmic)
                strength = np.log10(settlement.population + 1) * 10.0
                
                # Add Gaussian influence
                influence_map[settlement.x, settlement.y] += strength
        
        # Smooth influence
        influence_map = gaussian_filter(influence_map, sigma=size / 20.0)
        
        faction_influence_maps[faction.faction_id] = influence_map
    
    # Assign each cell to the faction with highest influence
    for x in range(size):
        for y in range(size):
            if not land_mask[x, y]:
                continue
            
            max_influence = -1
            best_faction = 0
            
            for faction_id, influence_map in faction_influence_maps.items():
                if influence_map[x, y] > max_influence:
                    max_influence = influence_map[x, y]
                    best_faction = faction_id
            
            territory_map[x, y] = best_faction
    
    return territory_map


def snap_to_natural_borders(
    territory_map: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    rivers: np.ndarray,
    land_mask: np.ndarray,
    size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap political borders to natural features (rivers, mountains).
    
    Returns:
        Tuple of (updated_territory_map, border_type_map)
    """
    from scipy.ndimage import sobel
    
    # Detect borders (where adjacent cells have different factions)
    edges_x = sobel(territory_map.astype(float), axis=0)
    edges_y = sobel(territory_map.astype(float), axis=1)
    border_mask = (np.abs(edges_x) + np.abs(edges_y)) > 0
    
    border_type_map = np.zeros((size, size), dtype=np.uint8)
    
    # Classify borders
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            if not border_mask[x, y]:
                continue
            
            if not land_mask[x, y]:
                continue
            
            # Check for natural features
            # Rivers are strong natural borders
            if rivers[x, y]:
                border_type_map[x, y] = BorderType.RIVER
                continue
            
            # Mountains (high elevation + high slope)
            if elevation[x, y] > 1500 and slope[x, y] > 0.3:
                border_type_map[x, y] = BorderType.MOUNTAIN
                continue
            
            # Check neighbors for rivers/mountains
            local_has_river = (
                rivers[max(0, x-1):min(size, x+2), max(0, y-1):min(size, y+2)].any()
            )
            
            local_has_mountain = (
                (elevation[max(0, x-1):min(size, x+2), max(0, y-1):min(size, y+2)] > 1500).any() and
                (slope[max(0, x-1):min(size, x+2), max(0, y-1):min(size, y+2)] > 0.3).any()
            )
            
            if local_has_river:
                border_type_map[x, y] = BorderType.RIVER
            elif local_has_mountain:
                border_type_map[x, y] = BorderType.MOUNTAIN
            else:
                border_type_map[x, y] = BorderType.POLITICAL
    
    return territory_map, border_type_map


def identify_contested_zones(
    territory_map: np.ndarray,
    world_state: WorldState,
    factions: List[Faction],
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Identify contested zones - resource-rich border areas.
    """
    from scipy.ndimage import sobel
    
    contested = np.zeros((size, size), dtype=bool)
    
    # Get border regions
    edges_x = sobel(territory_map.astype(float), axis=0)
    edges_y = sobel(territory_map.astype(float), axis=1)
    border_mask = (np.abs(edges_x) + np.abs(edges_y)) > 0
    
    # Expand borders slightly (contested zones are near borders)
    from scipy.ndimage import binary_dilation
    border_expanded = binary_dilation(border_mask, iterations=5)
    
    # Collect resource data
    mineral_value = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.mineral_deposits is not None:
                for mineral, deposits in chunk.mineral_deposits.items():
                    mineral_value[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] += deposits
    
    # Normalize
    if mineral_value.max() > 0:
        mineral_value /= mineral_value.max()
    
    # Mark contested zones: border areas with high resources
    resource_threshold = 0.6
    contested = border_expanded & (mineral_value > resource_threshold)
    
    # Randomly contest some border areas (representing historical claims)
    random_contest_chance = 0.05
    random_contest = rng.random((size, size)) < random_contest_chance
    contested = contested | (border_expanded & random_contest)
    
    return contested


def assign_settlements_to_factions(
    settlements: List,
    territory_map: np.ndarray,
    factions: List[Faction]
):
    """Assign settlements to factions based on territory map."""
    
    for settlement in settlements:
        if settlement.is_ruin:
            continue
        
        # If already assigned (capital), skip
        if hasattr(settlement, 'faction_id') and settlement.faction_id is not None:
            continue
        
        # Get faction from territory
        faction_id = territory_map[settlement.x, settlement.y]
        
        if faction_id > 0:
            settlement.faction_id = int(faction_id)


def create_vassalage_relationships(
    settlements: List,
    factions: List[Faction]
):
    """
    Create feudal vassalage relationships between factions.
    
    Smaller factions near larger ones become vassals.
    """
    from generation.pass_16_settlements import SettlementType
    
    # Sort factions by power (population + cities)
    faction_power = {}
    for faction in factions:
        faction_settlements = [s for s in settlements 
                              if hasattr(s, 'faction_id') and s.faction_id == faction.faction_id]
        
        num_cities = sum(1 for s in faction_settlements if s.settlement_type >= SettlementType.CITY)
        total_pop = sum(s.population for s in faction_settlements)
        
        power = num_cities * 10000 + total_pop
        faction_power[faction.faction_id] = power
    
    # Sort by power
    sorted_factions = sorted(factions, key=lambda f: faction_power.get(f.faction_id, 0), reverse=True)
    
    # Assign vassalage
    for i, small_faction in enumerate(sorted_factions):
        if small_faction.faction_type == FactionType.KINGDOM:
            continue  # Kingdoms don't become vassals
        
        # Find nearby stronger faction
        for large_faction in sorted_factions[:i]:
            if large_faction.faction_type not in [FactionType.KINGDOM, FactionType.DUCHY]:
                continue
            
            # Calculate distance
            dist = np.sqrt(
                (small_faction.capital_x - large_faction.capital_x)**2 +
                (small_faction.capital_y - large_faction.capital_y)**2
            )
            
            # Vassalage if close and significantly weaker
            power_ratio = faction_power.get(large_faction.faction_id, 1) / max(faction_power.get(small_faction.faction_id, 1), 1)
            
            if dist < 200 and power_ratio > 3.0:
                large_faction.vassal_faction_ids.append(small_faction.faction_id)
                small_faction.liege_faction_id = large_faction.faction_id
                break


def generate_faction_name(faction_id: int, faction_type: FactionType, rng: np.random.Generator) -> str:
    """Generate a faction name based on type."""
    
    # Simple name generation - in production, use a proper name generator
    prefixes = [
        "North", "South", "East", "West", "High", "Low", "Great", "Lesser",
        "New", "Old", "Upper", "Lower", "Grand", "Holy", "Royal", "Imperial"
    ]
    
    roots = [
        "mark", "land", "realm", "haven", "garde", "wick", "ton", "dale",
        "ford", "burg", "mont", "val", "crest", "shore", "bay", "march"
    ]
    
    suffixes = {
        FactionType.KINGDOM: ["Kingdom", "Empire", "Realm", "Dominion"],
        FactionType.DUCHY: ["Duchy", "Principality", "Margraviate"],
        FactionType.COUNTY: ["County", "Shire", "March"],
        FactionType.FREE_CITY: ["Republic", "Commonwealth", "League"],
        FactionType.TRIBAL: ["Clans", "Tribes", "Horde", "Confederation"],
        FactionType.THEOCRACY: ["Holy See", "Ecclesiarchy", "Temple State"],
    }
    
    prefix = rng.choice(prefixes)
    root = rng.choice(roots)
    suffix = rng.choice(suffixes.get(faction_type, ["State"]))
    
    if faction_type == FactionType.THEOCRACY:
        return f"The {suffix} of {prefix}{root}"
    else:
        return f"{suffix} of {prefix}{root}"