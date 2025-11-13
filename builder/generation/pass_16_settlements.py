"""
World Builder - Pass 16: Settlement Sites (REFACTORED)
Generates settlement locations with percentile-based scoring and hierarchical spacing.

KEY IMPROVEMENTS:
- Percentile-based scoring (adaptive to world conditions)
- Hierarchical spacing (villages can be near cities, but not near other villages)
- Kingdom-like clustering (settlements group around major cities)
- Debug layers for troubleshooting

SETTLEMENT HIERARCHY:
Metropolis → Cities → Towns → Villages → Hamlets
Fortresses (military) and Monasteries (religious) are special cases
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType, DrainageClass
from models.world import WorldState


# Settlement type enumeration
class SettlementType:
    HAMLET = 0
    VILLAGE = 1
    TOWN = 2
    CITY = 3
    METROPOLIS = 4
    FORTRESS = 5
    MONASTERY = 6
    RUIN = 7


# Specialization enumeration
class SettlementSpecialization:
    AGRICULTURAL = 0
    MINING = 1
    FISHING_PORT = 2
    TRADE_HUB = 3
    FORTRESS_MILITARY = 4
    RELIGIOUS = 5
    MAGICAL = 6
    MANUFACTURING = 7


@dataclass
class Settlement:
    """A settlement location with metadata."""
    settlement_id: int
    x: int
    y: int
    chunk_x: int
    chunk_y: int
    settlement_type: int
    population: int
    specialization: int
    age_years: int
    is_ruin: bool
    is_capital: bool
    name: str = ""
    
    # Site quality scores
    water_score: float = 0.0
    defense_score: float = 0.0
    resource_score: float = 0.0
    climate_score: float = 0.0
    access_score: float = 0.0
    total_score: float = 0.0

    # Political affiliation (assigned in Pass 18)
    faction_id: Optional[int] = None


@dataclass
class WorldStatistics:
    """Global statistics for percentile-based scoring."""
    # Temperature percentiles
    temp_p05: float
    temp_p25: float
    temp_p50: float
    temp_p75: float
    temp_p95: float
    
    # Precipitation percentiles
    precip_p05: float
    precip_p25: float
    precip_p50: float
    precip_p75: float
    precip_p95: float
    
    # Elevation percentiles
    elev_p05: float
    elev_p25: float
    elev_p50: float
    elev_p75: float
    elev_p95: float
    
    # Other ranges
    max_mineral: float
    max_mana: float


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate settlement sites across the world using improved placement algorithm.
    """
    print(f"  - Generating settlement sites (REFACTORED)...")
    
    size = world_state.size
    seed = params.seed
    rng = np.random.default_rng(seed + 16000)
    
    # STEP 1: Collect global data
    print(f"    - Collecting environmental data...")
    
    elevation_global, temp_global, precip_global = collect_climate_data(world_state, size)
    biome_global, agricultural_yield = collect_biome_data(world_state, size)
    mineral_deposits = collect_resource_data(world_state, size)
    mana_concentration = collect_magic_data(world_state, size)
    soil_quality = collect_soil_data(world_state, size)
    river_global = collect_hydrology_data(world_state, size)
    
    land_mask = elevation_global > 0
    
    # STEP 2: Calculate world statistics for percentile-based scoring
    print(f"    - Computing world statistics for adaptive scoring...")
    
    world_stats = calculate_world_statistics(
        elevation_global,
        temp_global,
        precip_global,
        mineral_deposits,
        mana_concentration,
        land_mask
    )
    
    # STEP 3: Calculate suitability scores using percentiles
    print(f"    - Calculating percentile-based site suitability scores...")
    
    suitability_components = calculate_suitability_scores(
        elevation_global,
        temp_global,
        precip_global,
        biome_global,
        agricultural_yield,
        mineral_deposits,
        mana_concentration,
        soil_quality,
        river_global,
        land_mask,
        world_stats,
        size
    )
    
    suitability_scores = suitability_components['total']
    
    # Store debug data for visualization
    world_state.settlement_debug_data = suitability_components
    
    # STEP 4: Hierarchical settlement placement with improved spacing
    print(f"    - Placing settlements with hierarchical spacing...")
    
    settlements = []
    occupied_map = np.zeros((size, size), dtype=np.uint8)  # 0 = empty, 1-7 = settlement type
    
    specialization_counts = {spec: 0 for spec in range(8)}
    
    # Define settlement targets (scales with world size)
    scale_factor = (size / 512.0) ** 2  # Area-based scaling
    
    settlement_targets = {
        SettlementType.METROPOLIS: max(1, int(2 * scale_factor)),
        SettlementType.CITY: max(5, int(10 * scale_factor)),
        SettlementType.TOWN: max(20, int(40 * scale_factor)),
        SettlementType.VILLAGE: max(60, int(120 * scale_factor)),
        SettlementType.HAMLET: max(150, int(250 * scale_factor)),
        SettlementType.FORTRESS: max(8, int(12 * scale_factor)),
        SettlementType.MONASTERY: max(8, int(12 * scale_factor)),
    }
    
    print(f"      Settlement targets for {size}x{size} world:")
    for stype, target in settlement_targets.items():
        print(f"        Type {stype}: {target}")
    
    # Place in hierarchical order
    placement_order = [
        SettlementType.METROPOLIS,
        SettlementType.CITY,
        SettlementType.TOWN,
        SettlementType.VILLAGE,
        SettlementType.HAMLET,
        SettlementType.FORTRESS,
        SettlementType.MONASTERY,
    ]
    
    for settlement_type in placement_order:
        target_count = settlement_targets[settlement_type]
        
        type_name = get_settlement_type_name(settlement_type)
        print(f"      Placing {target_count} {type_name}s...")
        
        new_settlements = place_settlements_hierarchical(
            settlement_type,
            target_count,
            suitability_scores,
            occupied_map,
            settlements,
            land_mask,
            size,
            rng,
            specialization_counts,
            mana_concentration,
            mineral_deposits,
            agricultural_yield,
            river_global,
            elevation_global
        )
        
        settlements.extend(new_settlements)
        print(f"        Placed {len(new_settlements)} {type_name}s")
    
    # STEP 5: Mark ruins
    print(f"    - Marking abandoned settlements as ruins...")
    
    num_ruins = int(len(settlements) * 0.12)
    ruin_candidates = [s for s in settlements if s.settlement_type in [
        SettlementType.HAMLET, SettlementType.VILLAGE, SettlementType.TOWN
    ]]
    
    if len(ruin_candidates) > num_ruins:
        ruin_indices = rng.choice(len(ruin_candidates), size=num_ruins, replace=False)
        for idx in ruin_indices:
            ruin_candidates[idx].is_ruin = True
            ruin_candidates[idx].population = 0
    
    num_actual_ruins = sum(1 for s in settlements if s.is_ruin)
    print(f"        Marked {num_actual_ruins} settlements as ruins")
    
    # STEP 6: Designate regional capitals
    print(f"    - Designating regional capitals...")
    
    region_size = size // 4
    for region_x in range(4):
        for region_y in range(4):
            region_settlements = [
                s for s in settlements
                if (region_x * region_size <= s.x < (region_x + 1) * region_size and
                    region_y * region_size <= s.y < (region_y + 1) * region_size and
                    s.settlement_type in [SettlementType.CITY, SettlementType.METROPOLIS] and
                    not s.is_ruin)
            ]
            
            if region_settlements:
                capital = max(region_settlements, key=lambda s: s.population)
                capital.is_capital = True
    
    num_capitals = sum(1 for s in settlements if s.is_capital)
    print(f"        Designated {num_capitals} regional capitals")
    
    # STEP 7: Store in chunks
    print(f"    - Storing settlement data in chunks...")
    
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            chunk_settlements = [
                s for s in settlements
                if s.chunk_x == chunk_x and s.chunk_y == chunk_y
            ]
            
            chunk.settlements = chunk_settlements
    
    # STEP 8: Create visualization map
    print(f"    - Creating settlement visualization map...")
    
    settlement_map = create_settlement_map(settlements, size)
    
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            chunk.settlement_presence = settlement_map[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE]
    
    # STEP 9: Report statistics
    print(f"  - Settlement generation statistics:")
    print(f"    Total settlements: {len(settlements)}")
    
    type_counts = {}
    for s in settlements:
        type_counts[s.settlement_type] = type_counts.get(s.settlement_type, 0) + 1
    
    for stype in placement_order:
        count = type_counts.get(stype, 0)
        target = settlement_targets[stype]
        type_name = get_settlement_type_name(stype)
        print(f"      {type_name}s: {count}/{target} ({count/target*100:.0f}%)")
    
    print(f"      Ruins: {num_actual_ruins}")
    print(f"      Capitals: {num_capitals}")
    
    # Store global settlement list
    world_state.settlements = settlements


def calculate_world_statistics(
    elevation,
    temperature,
    precipitation,
    mineral_deposits,
    mana_concentration,
    land_mask
) -> WorldStatistics:
    """Calculate percentile statistics from land areas only."""
    
    # Only use land cells for statistics
    land_temp = temperature[land_mask]
    land_precip = precipitation[land_mask]
    land_elev = elevation[land_mask]
    
    return WorldStatistics(
        # Temperature
        temp_p05=np.percentile(land_temp, 5),
        temp_p25=np.percentile(land_temp, 25),
        temp_p50=np.percentile(land_temp, 50),
        temp_p75=np.percentile(land_temp, 75),
        temp_p95=np.percentile(land_temp, 95),
        
        # Precipitation
        precip_p05=np.percentile(land_precip, 5),
        precip_p25=np.percentile(land_precip, 25),
        precip_p50=np.percentile(land_precip, 50),
        precip_p75=np.percentile(land_precip, 75),
        precip_p95=np.percentile(land_precip, 95),
        
        # Elevation
        elev_p05=np.percentile(land_elev, 5),
        elev_p25=np.percentile(land_elev, 25),
        elev_p50=np.percentile(land_elev, 50),
        elev_p75=np.percentile(land_elev, 75),
        elev_p95=np.percentile(land_elev, 95),
        
        # Other
        max_mineral=mineral_deposits.max(),
        max_mana=mana_concentration.max(),
    )


def calculate_suitability_scores(
    elevation,
    temperature,
    precipitation,
    biome,
    agricultural_yield,
    mineral_deposits,
    mana_concentration,
    soil_quality,
    river,
    land_mask,
    world_stats: WorldStatistics,
    size
) -> Dict[str, np.ndarray]:
    """
    Calculate site suitability using percentile-based scoring.
    Returns dict with component scores for debugging.
    """
    
    # Initialize score components
    water_score = np.zeros((size, size), dtype=np.float32)
    defense_score = np.zeros((size, size), dtype=np.float32)
    resource_score = np.zeros((size, size), dtype=np.float32)
    climate_score = np.zeros((size, size), dtype=np.float32)
    access_score = np.zeros((size, size), dtype=np.float32)
    
    # WATER ACCESS (30%) - Distance to rivers
    print(f"        Water access...")
    distance_to_water = distance_transform_edt(~river)
    water_score = np.exp(-distance_to_water / 50.0)
    
    # DEFENSIBILITY (20%) - Hills but not mountains, moderate slope
    print(f"        Defensibility...")
    from utils.spatial import calculate_slope
    slope = calculate_slope(elevation)
    
    # Ideal elevation: 25th-75th percentile (moderate hills)
    elev_defense = np.zeros_like(elevation)
    elev_defense[(elevation >= world_stats.elev_p25) & (elevation < world_stats.elev_p50)] = 0.7
    elev_defense[(elevation >= world_stats.elev_p50) & (elevation < world_stats.elev_p75)] = 1.0
    elev_defense[(elevation >= world_stats.elev_p75) & (elevation < world_stats.elev_p95)] = 0.6
    elev_defense[elevation < world_stats.elev_p25] = 0.3
    elev_defense[elevation >= world_stats.elev_p95] = 0.2
    
    # Moderate slope is good
    slope_defense = np.where(slope < 0.15, 1.0, np.exp(-slope / 0.2))
    
    defense_score = (elev_defense + slope_defense) / 2.0
    
    # RESOURCES (25%) - Agriculture, minerals, soil
    print(f"        Resources...")
    # Normalize mineral deposits
    mineral_norm = mineral_deposits / (world_stats.max_mineral + 1e-6)
    mineral_norm = np.clip(mineral_norm, 0, 1)
    
    resource_score = (
        agricultural_yield * 0.5 +
        mineral_norm * 0.3 +
        soil_quality * 0.2
    )
    
    # CLIMATE (15%) - Moderate temperature and precipitation (25th-75th percentile)
    print(f"        Climate...")
    
    # Temperature: ideal in 25th-75th percentile
    temp_score = np.zeros_like(temperature)
    temp_score[(temperature >= world_stats.temp_p25) & (temperature <= world_stats.temp_p75)] = 1.0
    temp_score[(temperature >= world_stats.temp_p05) & (temperature < world_stats.temp_p25)] = 0.6
    temp_score[(temperature > world_stats.temp_p75) & (temperature <= world_stats.temp_p95)] = 0.6
    temp_score[temperature < world_stats.temp_p05] = 0.2
    temp_score[temperature > world_stats.temp_p95] = 0.3
    
    # Precipitation: ideal in 25th-75th percentile
    precip_score = np.zeros_like(precipitation)
    precip_score[(precipitation >= world_stats.precip_p25) & (precipitation <= world_stats.precip_p75)] = 1.0
    precip_score[(precipitation >= world_stats.precip_p05) & (precipitation < world_stats.precip_p25)] = 0.5
    precip_score[(precipitation > world_stats.precip_p75) & (precipitation <= world_stats.precip_p95)] = 0.7
    precip_score[precipitation < world_stats.precip_p05] = 0.2
    precip_score[precipitation > world_stats.precip_p95] = 0.4
    
    climate_score = (temp_score + precip_score) / 2.0
    
    # ACCESSIBILITY (10%) - Flat terrain, water access
    print(f"        Accessibility...")
    flat_score = np.where(slope < 0.1, 1.0, np.exp(-slope / 0.2))
    river_access = np.exp(-distance_to_water / 100.0)
    
    access_score = (flat_score + river_access) / 2.0
    
    # COMBINE with weights
    total_score = (
        water_score * 0.30 +
        defense_score * 0.20 +
        resource_score * 0.25 +
        climate_score * 0.15 +
        access_score * 0.10
    )
    
    # Apply land mask
    total_score[~land_mask] = 0
    water_score[~land_mask] = 0
    defense_score[~land_mask] = 0
    resource_score[~land_mask] = 0
    climate_score[~land_mask] = 0
    access_score[~land_mask] = 0
    
    # Smooth slightly
    total_score = gaussian_filter(total_score, sigma=2.0)
    
    return {
        'total': total_score,
        'water': water_score,
        'defense': defense_score,
        'resource': resource_score,
        'climate': climate_score,
        'access': access_score,
    }


def place_settlements_hierarchical(
    settlement_type: int,
    target_count: int,
    suitability_scores: np.ndarray,
    occupied_map: np.ndarray,
    existing_settlements: List[Settlement],
    land_mask: np.ndarray,
    size: int,
    rng: np.random.Generator,
    specialization_counts: Dict[int, int],
    mana_concentration: np.ndarray,
    mineral_deposits: np.ndarray,
    agricultural_yield: np.ndarray,
    river: np.ndarray,
    elevation: np.ndarray
) -> List[Settlement]:
    """
    Place settlements with hierarchical spacing rules.
    
    Key insight: Different types can be closer together.
    - Cities must be far from other cities
    - Villages can be near cities, but not near other villages
    - Hamlets can be near villages/cities, but not near other hamlets
    """
    
    settlements = []
    
    # Spacing rules
    spacing_rules = {
        SettlementType.METROPOLIS: {
            'same_type': 200,    # Far from other metropolises
            'larger_type': 200,  # N/A (largest type)
        },
        SettlementType.CITY: {
            'same_type': 100,    # Far from other cities
            'larger_type': 120,  # Distance from metropolises
        },
        SettlementType.TOWN: {
            'same_type': 50,     # Moderate distance from towns
            'larger_type': 30,   # Can be near cities
        },
        SettlementType.VILLAGE: {
            'same_type': 25,     # Close spacing between villages
            'larger_type': 15,   # Can be close to towns/cities
        },
        SettlementType.HAMLET: {
            'same_type': 12,     # Very close spacing
            'larger_type': 8,    # Very close to villages
        },
        SettlementType.FORTRESS: {
            'same_type': 80,     # Spread out fortresses
            'larger_type': 30,   # Can be near cities
        },
        SettlementType.MONASTERY: {
            'same_type': 100,    # Isolated monasteries
            'larger_type': 50,   # Away from settlements
        },
    }
    
    rules = spacing_rules[settlement_type]
    same_type_spacing = rules['same_type']
    larger_type_spacing = rules['larger_type']
    
    # Population ranges
    pop_ranges = {
        SettlementType.METROPOLIS: (50000, 100000),
        SettlementType.CITY: (5000, 50000),
        SettlementType.TOWN: (500, 5000),
        SettlementType.VILLAGE: (100, 500),
        SettlementType.HAMLET: (20, 100),
        SettlementType.FORTRESS: (50, 500),
        SettlementType.MONASTERY: (20, 200),
    }
    
    pop_min, pop_max = pop_ranges[settlement_type]
    
    # Try to place settlements
    attempts = 0
    max_attempts = target_count * 20
    
    while len(settlements) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Find available high-quality sites
        available_scores = suitability_scores.copy()
        available_scores[occupied_map > 0] *= 0.5  # Penalize already occupied, but don't eliminate
        
        # Special logic for fortresses (prefer high ground)
        if settlement_type == SettlementType.FORTRESS:
            available_scores *= (elevation / elevation.max()) * 2.0
        
        # Special logic for monasteries (prefer isolation + high mana)
        if settlement_type == SettlementType.MONASTERY:
            distance_from_occupied = distance_transform_edt(occupied_map == 0)
            isolation = np.clip(distance_from_occupied / 100.0, 0, 1)
            available_scores *= isolation * 2.0
            available_scores *= (mana_concentration + 0.5)
        
        if available_scores.max() < 0.05:
            break
        
        # Pick from top candidates
        threshold = available_scores.max() * 0.6
        candidates = np.argwhere(available_scores > threshold)
        
        if len(candidates) == 0:
            break
        
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        
        # Check spacing constraints
        valid_placement = True
        
        # Check distance to same type
        for s in existing_settlements:
            if s.settlement_type == settlement_type:
                dist = np.sqrt((s.x - x)**2 + (s.y - y)**2)
                if dist < same_type_spacing:
                    valid_placement = False
                    break
        
        if not valid_placement:
            # Mark this area as checked
            occupied_map[max(0, x-same_type_spacing//2):min(size, x+same_type_spacing//2),
                        max(0, y-same_type_spacing//2):min(size, y+same_type_spacing//2)] = 255
            continue
        
        # Check distance to larger settlements (if applicable)
        if settlement_type != SettlementType.METROPOLIS:
            for s in existing_settlements:
                if s.settlement_type > settlement_type:  # Larger settlements
                    dist = np.sqrt((s.x - x)**2 + (s.y - y)**2)
                    if dist < larger_type_spacing:
                        valid_placement = False
                        break
        
        if not valid_placement:
            occupied_map[max(0, x-larger_type_spacing//2):min(size, x+larger_type_spacing//2),
                        max(0, y-larger_type_spacing//2):min(size, y+larger_type_spacing//2)] = 255
            continue
        
        # Valid placement - create settlement
        specialization = determine_specialization(
            x, y,
            mana_concentration,
            mineral_deposits,
            agricultural_yield,
            river,
            settlement_type,
            specialization_counts,
            rng
        )
        
        population = rng.integers(pop_min, pop_max)
        
        age_base = {
            SettlementType.METROPOLIS: 800,
            SettlementType.CITY: 500,
            SettlementType.TOWN: 300,
            SettlementType.VILLAGE: 150,
            SettlementType.HAMLET: 50,
            SettlementType.FORTRESS: 200,
            SettlementType.MONASTERY: 300,
        }
        
        age = int(rng.normal(age_base.get(settlement_type, 100), 50))
        age = max(10, age)
        
        chunk_x = x // CHUNK_SIZE
        chunk_y = y // CHUNK_SIZE
        
        settlement = Settlement(
            settlement_id=len(existing_settlements) + len(settlements),
            x=x,
            y=y,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            settlement_type=settlement_type,
            population=population,
            specialization=specialization,
            age_years=age,
            is_ruin=False,
            is_capital=False,
            total_score=suitability_scores[x, y],
        )
        
        settlements.append(settlement)
        
        # Mark as occupied (type-specific)
        occupied_map[x, y] = settlement_type + 1
        
        # Mark influence zone (smaller radius for smaller settlements)
        influence_radius = same_type_spacing // 3
        occupied_map[max(0, x-influence_radius):min(size, x+influence_radius),
                    max(0, y-influence_radius):min(size, y+influence_radius)] = settlement_type + 1
    
    return settlements


def determine_specialization(
    x, y,
    mana_concentration,
    mineral_deposits,
    agricultural_yield,
    river,
    settlement_type,
    specialization_counts,
    rng
):
    """Determine settlement specialization based on local resources."""
    
    mana = mana_concentration[x, y]
    minerals = mineral_deposits[x, y]
    agriculture = agricultural_yield[x, y]
    has_river = river[x, y]
    
    scores = {}
    
    scores[SettlementSpecialization.AGRICULTURAL] = agriculture * 2.0
    scores[SettlementSpecialization.MINING] = minerals * 3.0
    scores[SettlementSpecialization.FISHING_PORT] = 1.0 if has_river else 0.1
    scores[SettlementSpecialization.TRADE_HUB] = (
        (1.0 if has_river else 0.3) * (agriculture + minerals + mana) / 3.0
    )
    scores[SettlementSpecialization.FORTRESS_MILITARY] = 0.3
    scores[SettlementSpecialization.RELIGIOUS] = 0.5
    scores[SettlementSpecialization.MAGICAL] = mana * 3.0
    scores[SettlementSpecialization.MANUFACTURING] = (
        minerals * 1.5 + (1.0 if has_river else 0.3)
    )
    
    if settlement_type == SettlementType.FORTRESS:
        return SettlementSpecialization.FORTRESS_MILITARY
    
    if settlement_type == SettlementType.MONASTERY:
        return SettlementSpecialization.RELIGIOUS
    
    # Balance specializations
    total_placed = sum(specialization_counts.values())
    if total_placed > 0:
        for spec in scores:
            current_ratio = specialization_counts.get(spec, 0) / total_placed
            target_ratio = 1.0 / 8.0
            
            if current_ratio < target_ratio:
                scores[spec] *= 1.5
            elif current_ratio > target_ratio * 1.5:
                scores[spec] *= 0.7
    
    valid_specs = [s for s, score in scores.items() if score > 0.1]
    
    if not valid_specs:
        specialization = SettlementSpecialization.AGRICULTURAL
    else:
        spec_weights = np.array([scores[s] for s in valid_specs])
        spec_weights = spec_weights / spec_weights.sum()
        specialization = rng.choice(valid_specs, p=spec_weights)
    
    specialization_counts[specialization] = specialization_counts.get(specialization, 0) + 1
    
    return specialization


def get_settlement_type_name(settlement_type: int) -> str:
    """Get human-readable name for settlement type."""
    names = {
        SettlementType.HAMLET: "Hamlet",
        SettlementType.VILLAGE: "Village",
        SettlementType.TOWN: "Town",
        SettlementType.CITY: "City",
        SettlementType.METROPOLIS: "Metropolis",
        SettlementType.FORTRESS: "Fortress",
        SettlementType.MONASTERY: "Monastery",
    }
    return names.get(settlement_type, "Unknown")


def create_settlement_map(settlements: List[Settlement], size: int) -> np.ndarray:
    """Create visualization map of settlement types."""
    settlement_map = np.zeros((size, size), dtype=np.uint8)
    
    for s in settlements:
        if s.is_ruin:
            value = 8
        else:
            value = s.settlement_type + 1
        
        # Mark area based on settlement size
        radius = 1 if s.settlement_type <= SettlementType.VILLAGE else 2
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = s.x + dx, s.y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    settlement_map[nx, ny] = value
    
    return settlement_map


# Data collection functions (unchanged from original)
def collect_climate_data(world_state, size):
    """Collect temperature and precipitation data."""
    elevation = np.zeros((size, size), dtype=np.float32)
    temperature = np.zeros((size, size), dtype=np.float32)
    precipitation = np.zeros((size, size), dtype=np.float32)
    
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
            if chunk.temperature_c is not None:
                temperature[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.temperature_c
            if chunk.precipitation_mm is not None:
                precipitation[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
    
    return elevation, temperature, precipitation


def collect_biome_data(world_state, size):
    """Collect biome and agricultural data."""
    biome = np.zeros((size, size), dtype=np.uint8)
    agricultural = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.biome_type is not None:
                biome[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
            if chunk.agricultural_yield is not None:
                agricultural[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.agricultural_yield
    
    return biome, agricultural


def collect_resource_data(world_state, size):
    """Collect mineral deposit data."""
    minerals = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.mineral_deposits is not None:
                for mineral, deposit_map in chunk.mineral_deposits.items():
                    minerals[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] += deposit_map
    
    return minerals


def collect_magic_data(world_state, size):
    """Collect mana concentration data."""
    mana = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.mana_concentration is not None:
                mana[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.mana_concentration
    
    return mana


def collect_soil_data(world_state, size):
    """Collect soil quality data."""
    soil_quality = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.soil_ph is not None and chunk.soil_drainage is not None:
                ph_quality = 1.0 - np.abs(chunk.soil_ph - 6.75) / 3.0
                ph_quality = np.clip(ph_quality, 0, 1)
                
                drainage_quality = np.zeros_like(chunk.soil_drainage, dtype=np.float32)
                drainage_quality[chunk.soil_drainage == DrainageClass.WELL] = 1.0
                drainage_quality[chunk.soil_drainage == DrainageClass.MODERATELY_WELL] = 0.9
                drainage_quality[chunk.soil_drainage == DrainageClass.SOMEWHAT_EXCESSIVELY] = 0.8
                
                combined = (ph_quality + drainage_quality) / 2.0
                soil_quality[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = combined
    
    return soil_quality


def collect_hydrology_data(world_state, size):
    """Collect river data."""
    rivers = np.zeros((size, size), dtype=bool)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.river_presence is not None:
                rivers[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
    
    return rivers