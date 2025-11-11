"""
World Builder - Pass 16: Settlement Sites
Generates settlement locations, types, sizes, and specializations.

APPROACH:
- Hierarchical placement: Cities → Towns → Villages → Hamlets → Fortresses → Monasteries
- Competitive inhibition: Larger settlements suppress nearby sites
- Weighted scoring for site selection (water, defensibility, resources, climate, accessibility)
- Soil quality and magic concentration as additional factors
- Even distribution of specialization types
- 10-20% of sites marked as ruins for exploration content

SETTLEMENT TYPES:
1. Hamlet (20-100 pop): Small farming/resource extraction clusters
2. Village (100-500 pop): Agricultural centers, fishing villages
3. Town (500-5000 pop): Market towns, garrison posts, trade hubs
4. City (5,000-50,000 pop): Regional capitals, universities, port cities
5. Metropolis (50,000+ pop): Empire seats, magical academies
6. Fortress (variable): Military strongholds, border keeps
7. Monastery (20-200 pop): Religious retreats, scholarly enclaves
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from typing import List, Dict, Tuple, Set
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
    RUIN = 7  # Abandoned settlement


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
    settlement_type: int  # SettlementType
    population: int
    specialization: int  # SettlementSpecialization
    age_years: int  # Historical depth
    is_ruin: bool
    is_capital: bool  # Regional capital
    name: str = ""  # Can be generated later
    
    # Site quality scores
    water_score: float = 0.0
    defense_score: float = 0.0
    resource_score: float = 0.0
    climate_score: float = 0.0
    access_score: float = 0.0


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate settlement sites across the world.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating settlement sites...")
    
    size = world_state.size
    seed = params.seed
    rng = np.random.default_rng(seed + 16000)
    
    # STEP 1: Collect global data for site selection
    print(f"    - Collecting environmental data for site analysis...")
    
    elevation_global, temp_global, precip_global = collect_climate_data(world_state, size)
    biome_global, agricultural_yield = collect_biome_data(world_state, size)
    mineral_deposits = collect_resource_data(world_state, size)
    mana_concentration = collect_magic_data(world_state, size)
    soil_quality = collect_soil_data(world_state, size)
    river_global = collect_hydrology_data(world_state, size)
    
    land_mask = elevation_global > 0
    
    # STEP 2: Calculate site suitability scores
    print(f"    - Calculating site suitability scores...")
    
    suitability_scores = calculate_suitability_scores(
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
        size
    )
    
    # STEP 3: Hierarchical settlement placement
    print(f"    - Placing settlements hierarchically...")
    
    settlements = []
    occupied_sites = np.zeros((size, size), dtype=bool)
    
    # Track specialization counts for balanced distribution
    specialization_counts = {spec: 0 for spec in range(8)}
    
    # Place Metropolises (1-3 for a 512x512 world, scales with size)
    num_metropolises = max(1, (size // 512) * 2)
    print(f"      Placing {num_metropolises} metropolises...")
    
    metropolises = place_settlements_by_type(
        SettlementType.METROPOLIS,
        num_metropolises,
        suitability_scores,
        occupied_sites,
        land_mask,
        size,
        rng,
        specialization_counts,
        mana_concentration,
        mineral_deposits,
        agricultural_yield,
        river_global
    )
    settlements.extend(metropolises)
    print(f"        Placed {len(metropolises)} metropolises")
    
    # Place Cities (5-15 depending on world size)
    num_cities = max(5, (size // 512) * 10)
    print(f"      Placing {num_cities} cities...")
    
    cities = place_settlements_by_type(
        SettlementType.CITY,
        num_cities,
        suitability_scores,
        occupied_sites,
        land_mask,
        size,
        rng,
        specialization_counts,
        mana_concentration,
        mineral_deposits,
        agricultural_yield,
        river_global
    )
    settlements.extend(cities)
    print(f"        Placed {len(cities)} cities")
    
    # Place Towns (30-60)
    num_towns = max(30, (size // 512) * 40)
    print(f"      Placing {num_towns} towns...")
    
    towns = place_settlements_by_type(
        SettlementType.TOWN,
        num_towns,
        suitability_scores,
        occupied_sites,
        land_mask,
        size,
        rng,
        specialization_counts,
        mana_concentration,
        mineral_deposits,
        agricultural_yield,
        river_global
    )
    settlements.extend(towns)
    print(f"        Placed {len(towns)} towns")
    
    # Place Villages (100-200)
    num_villages = max(100, (size // 512) * 150)
    print(f"      Placing {num_villages} villages...")
    
    villages = place_settlements_by_type(
        SettlementType.VILLAGE,
        num_villages,
        suitability_scores,
        occupied_sites,
        land_mask,
        size,
        rng,
        specialization_counts,
        mana_concentration,
        mineral_deposits,
        agricultural_yield,
        river_global
    )
    settlements.extend(villages)
    print(f"        Placed {len(villages)} villages")
    
    # Place Hamlets (200-400)
    num_hamlets = max(200, (size // 512) * 300)
    print(f"      Placing {num_hamlets} hamlets...")
    
    hamlets = place_settlements_by_type(
        SettlementType.HAMLET,
        num_hamlets,
        suitability_scores,
        occupied_sites,
        land_mask,
        size,
        rng,
        specialization_counts,
        mana_concentration,
        mineral_deposits,
        agricultural_yield,
        river_global
    )
    settlements.extend(hamlets)
    print(f"        Placed {len(hamlets)} hamlets")
    
    # Place Fortresses (10-20) - defensive positions
    num_fortresses = max(10, (size // 512) * 15)
    print(f"      Placing {num_fortresses} fortresses...")
    
    fortresses = place_fortresses(
        num_fortresses,
        elevation_global,
        occupied_sites,
        land_mask,
        settlements,  # Protect existing settlements
        size,
        rng
    )
    settlements.extend(fortresses)
    print(f"        Placed {len(fortresses)} fortresses")
    
    # Place Monasteries (10-20) - isolated religious sites
    num_monasteries = max(10, (size // 512) * 15)
    print(f"      Placing {num_monasteries} monasteries...")
    
    monasteries = place_monasteries(
        num_monasteries,
        elevation_global,
        mana_concentration,
        occupied_sites,
        land_mask,
        size,
        rng
    )
    settlements.extend(monasteries)
    print(f"        Placed {len(monasteries)} monasteries")
    
    # STEP 4: Mark some settlements as ruins (10-15%)
    print(f"    - Marking abandoned settlements as ruins...")
    
    num_ruins = int(len(settlements) * 0.12)  # 12% ruins
    ruin_candidates = [s for s in settlements if s.settlement_type in [
        SettlementType.HAMLET, SettlementType.VILLAGE, SettlementType.TOWN
    ]]
    
    if len(ruin_candidates) > num_ruins:
        ruin_indices = rng.choice(len(ruin_candidates), size=num_ruins, replace=False)
        for idx in ruin_indices:
            ruin_candidates[idx].is_ruin = True
            ruin_candidates[idx].population = 0  # Abandoned
    
    num_actual_ruins = sum(1 for s in settlements if s.is_ruin)
    print(f"        Marked {num_actual_ruins} settlements as ruins")
    
    # STEP 5: Mark regional capitals (largest city in each region)
    print(f"    - Designating regional capitals...")
    
    # Group by rough regions (divide world into quadrants/sectors)
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
                # Mark largest as capital
                capital = max(region_settlements, key=lambda s: s.population)
                capital.is_capital = True
    
    num_capitals = sum(1 for s in settlements if s.is_capital)
    print(f"        Designated {num_capitals} regional capitals")
    
    # STEP 6: Store settlements in chunks
    print(f"    - Storing settlement data in chunks...")
    
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Find settlements in this chunk
            chunk_settlements = [
                s for s in settlements
                if s.chunk_x == chunk_x and s.chunk_y == chunk_y
            ]
            
            chunk.settlements = chunk_settlements
    
    # STEP 7: Generate global settlement map for visualization
    print(f"    - Creating settlement visualization map...")
    
    settlement_map = create_settlement_map(settlements, size)
    
    # Store in chunks as settlement_presence array
    for chunk_y in range(size // CHUNK_SIZE):
        for chunk_x in range(size // CHUNK_SIZE):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            chunk.settlement_presence = settlement_map[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE]
    
    # STEP 8: Report statistics
    print(f"  - Settlement generation statistics:")
    print(f"    Total settlements: {len(settlements)}")
    
    type_counts = {}
    for s in settlements:
        stype = s.settlement_type
        type_counts[stype] = type_counts.get(stype, 0) + 1
    
    type_names = {
        SettlementType.HAMLET: "Hamlets",
        SettlementType.VILLAGE: "Villages",
        SettlementType.TOWN: "Towns",
        SettlementType.CITY: "Cities",
        SettlementType.METROPOLIS: "Metropolises",
        SettlementType.FORTRESS: "Fortresses",
        SettlementType.MONASTERY: "Monasteries",
    }
    
    for stype, name in type_names.items():
        count = type_counts.get(stype, 0)
        print(f"      {name}: {count}")
    
    print(f"      Ruins: {num_actual_ruins}")
    print(f"      Capitals: {num_capitals}")
    
    print(f"    Specialization distribution:")
    spec_names = {
        SettlementSpecialization.AGRICULTURAL: "Agricultural",
        SettlementSpecialization.MINING: "Mining",
        SettlementSpecialization.FISHING_PORT: "Fishing/Port",
        SettlementSpecialization.TRADE_HUB: "Trade Hub",
        SettlementSpecialization.FORTRESS_MILITARY: "Fortress",
        SettlementSpecialization.RELIGIOUS: "Religious",
        SettlementSpecialization.MAGICAL: "Magical",
        SettlementSpecialization.MANUFACTURING: "Manufacturing",
    }
    
    for spec, name in spec_names.items():
        count = specialization_counts.get(spec, 0)
        if count > 0:
            print(f"      {name}: {count}")
    
    # Store settlement list at world level
    world_state.settlements = settlements


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
            
            # Sum all mineral deposits
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
    """Collect soil quality data (pH and drainage)."""
    soil_quality = np.zeros((size, size), dtype=np.float32)
    
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Combine soil pH (optimal 6-7.5) and drainage
            if chunk.soil_ph is not None and chunk.soil_drainage is not None:
                # pH quality: closer to 6.5-7 is better
                ph_quality = 1.0 - np.abs(chunk.soil_ph - 6.75) / 3.0
                ph_quality = np.clip(ph_quality, 0, 1)
                
                # Drainage quality: well-drained is best
                from config import DrainageClass
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
    size
):
    """
    Calculate site suitability scores for settlements.
    
    Weighted factors:
    - Water access (30%)
    - Defensibility (20%)
    - Resources (25%)
    - Climate (15%)
    - Accessibility (10%)
    """
    suitability = np.zeros((size, size), dtype=np.float32)
    
    # Water access score (30%) - distance to river
    print(f"        Calculating water access scores...")
    distance_to_water = distance_transform_edt(~river)
    water_score = np.exp(-distance_to_water / 50.0)  # Within ~5km is ideal
    
    # Defensibility score (20%) - elevation advantage, not too steep
    print(f"        Calculating defensibility scores...")
    from utils.spatial import calculate_slope
    slope = calculate_slope(elevation)
    
    # Hills are defensible, mountains are not
    elevation_defense = np.zeros_like(elevation)
    elevation_defense[elevation > 50] = 0.3
    elevation_defense[elevation > 200] = 0.7
    elevation_defense[elevation > 500] = 1.0
    elevation_defense[elevation > 1500] = 0.6  # Too high
    elevation_defense[elevation > 2500] = 0.2  # Mountains
    
    # Not too steep
    slope_defense = np.where(slope < 0.15, 1.0, 0.5)
    
    defense_score = (elevation_defense + slope_defense) / 2.0
    
    # Resource score (25%) - agricultural + minerals + soil quality
    print(f"        Calculating resource scores...")
    resource_score = (
        agricultural_yield * 0.5 +
        np.clip(mineral_deposits * 2.0, 0, 1) * 0.3 +
        soil_quality * 0.2
    )
    
    # Climate score (15%) - moderate temperature, adequate rainfall
    print(f"        Calculating climate scores...")
    temp_ideal = np.zeros_like(temperature)
    temp_ideal[(temperature >= 5) & (temperature <= 25)] = 1.0
    temp_ideal[(temperature >= 0) & (temperature < 5)] = 0.6
    temp_ideal[(temperature > 25) & (temperature <= 30)] = 0.6
    temp_ideal[temperature < 0] = 0.3
    temp_ideal[temperature > 30] = 0.3
    
    precip_ideal = np.zeros_like(precipitation)
    precip_ideal[(precipitation >= 400) & (precipitation <= 1500)] = 1.0
    precip_ideal[(precipitation >= 250) & (precipitation < 400)] = 0.6
    precip_ideal[(precipitation > 1500) & (precipitation <= 2000)] = 0.7
    precip_ideal[precipitation < 250] = 0.2
    precip_ideal[precipitation > 2000] = 0.5
    
    climate_score = (temp_ideal + precip_ideal) / 2.0
    
    # Accessibility score (10%) - flat terrain, near water for trade
    print(f"        Calculating accessibility scores...")
    flat_score = np.where(slope < 0.1, 1.0, np.exp(-slope / 0.2))
    river_access = np.exp(-distance_to_water / 100.0)
    
    access_score = (flat_score + river_access) / 2.0
    
    # Combine weighted scores
    suitability = (
        water_score * 0.30 +
        defense_score * 0.20 +
        resource_score * 0.25 +
        climate_score * 0.15 +
        access_score * 0.10
    )
    
    # Apply land mask (no settlements in ocean)
    suitability[~land_mask] = 0
    
    # Smooth suitability slightly
    suitability = gaussian_filter(suitability, sigma=2.0)
    
    return suitability


def place_settlements_by_type(
    settlement_type,
    target_count,
    suitability_scores,
    occupied_sites,
    land_mask,
    size,
    rng,
    specialization_counts,
    mana_concentration,
    mineral_deposits,
    agricultural_yield,
    river
):
    """Place settlements of a specific type using competitive inhibition."""
    settlements = []
    
    # Minimum spacing based on settlement type
    min_spacing = {
        SettlementType.METROPOLIS: 150,
        SettlementType.CITY: 80,
        SettlementType.TOWN: 40,
        SettlementType.VILLAGE: 20,
        SettlementType.HAMLET: 10,
    }
    
    spacing = min_spacing.get(settlement_type, 20)
    
    # Population ranges
    pop_ranges = {
        SettlementType.METROPOLIS: (50000, 100000),
        SettlementType.CITY: (5000, 50000),
        SettlementType.TOWN: (500, 5000),
        SettlementType.VILLAGE: (100, 500),
        SettlementType.HAMLET: (20, 100),
    }
    
    pop_min, pop_max = pop_ranges.get(settlement_type, (100, 500))
    
    attempts = 0
    max_attempts = target_count * 10
    
    while len(settlements) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Find best available site
        available_scores = suitability_scores.copy()
        available_scores[occupied_sites] = 0
        
        if available_scores.max() < 0.1:
            break  # No more good sites
        
        # Add some randomness to avoid always picking absolute best
        threshold = available_scores.max() * 0.7
        candidates = np.argwhere(available_scores > threshold)
        
        if len(candidates) == 0:
            break
        
        # Pick random from top candidates
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        
        # Check minimum spacing from existing settlements
        too_close = False
        for s in settlements:
            dist = np.sqrt((s.x - x)**2 + (s.y - y)**2)
            if dist < spacing:
                too_close = True
                break
        
        if too_close:
            # Mark this spot as occupied to avoid repeated checks
            occupied_sites[max(0, x-spacing//2):min(size, x+spacing//2),
                          max(0, y-spacing//2):min(size, y+spacing//2)] = True
            continue
        
        # Determine specialization based on local resources
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
        
        # Generate population
        population = rng.integers(pop_min, pop_max)
        
        # Generate age (older settlements for larger types)
        age_base = {
            SettlementType.METROPOLIS: 800,
            SettlementType.CITY: 500,
            SettlementType.TOWN: 300,
            SettlementType.VILLAGE: 150,
            SettlementType.HAMLET: 50,
        }
        
        age = int(rng.normal(age_base.get(settlement_type, 100), 50))
        age = max(10, age)
        
        # Create settlement
        chunk_x = x // CHUNK_SIZE
        chunk_y = y // CHUNK_SIZE
        
        settlement = Settlement(
            settlement_id=len(settlements),
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
            water_score=suitability_scores[x, y],  # Store for reference
        )
        
        settlements.append(settlement)
        
        # Mark surrounding area as occupied
        occupied_sites[max(0, x-spacing):min(size, x+spacing),
                      max(0, y-spacing):min(size, y+spacing)] = True
    
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
    """Determine settlement specialization based on local resources and balance."""
    
    # Get local values
    mana = mana_concentration[x, y]
    minerals = mineral_deposits[x, y]
    agriculture = agricultural_yield[x, y]
    has_river = river[x, y]
    
    # Calculate scores for each specialization
    scores = {}
    
    # Agricultural - needs good farmland
    scores[SettlementSpecialization.AGRICULTURAL] = agriculture * 2.0
    
    # Mining - needs mineral deposits
    scores[SettlementSpecialization.MINING] = minerals * 3.0
    
    # Fishing/Port - needs water access
    scores[SettlementSpecialization.FISHING_PORT] = 1.0 if has_river else 0.1
    
    # Trade Hub - benefits from water + moderate everything
    scores[SettlementSpecialization.TRADE_HUB] = (
        (1.0 if has_river else 0.3) * (agriculture + minerals + mana) / 3.0
    )
    
    # Fortress - military specialization (assigned separately)
    scores[SettlementSpecialization.FORTRESS_MILITARY] = 0.3
    
    # Religious - benefits from isolation/beauty
    scores[SettlementSpecialization.RELIGIOUS] = 0.5
    
    # Magical - needs high mana
    scores[SettlementSpecialization.MAGICAL] = mana * 3.0
    
    # Manufacturing - needs minerals + water
    scores[SettlementSpecialization.MANUFACTURING] = (
        minerals * 1.5 + (1.0 if has_river else 0.3)
    )
    
    # Fortresses and monasteries get specific specializations
    if settlement_type == SettlementType.FORTRESS:
        return SettlementSpecialization.FORTRESS_MILITARY
    
    if settlement_type == SettlementType.MONASTERY:
        return SettlementSpecialization.RELIGIOUS
    
    # Balance specializations - boost underrepresented types
    total_placed = sum(specialization_counts.values())
    if total_placed > 0:
        for spec in scores:
            current_ratio = specialization_counts.get(spec, 0) / total_placed
            target_ratio = 1.0 / 8.0  # Aim for 12.5% each
            
            if current_ratio < target_ratio:
                # Boost underrepresented
                scores[spec] *= 1.5
            elif current_ratio > target_ratio * 1.5:
                # Reduce overrepresented
                scores[spec] *= 0.7
    
    # Pick specialization weighted by scores
    valid_specs = [s for s, score in scores.items() if score > 0.1]
    
    if not valid_specs:
        # Fallback to agricultural
        specialization = SettlementSpecialization.AGRICULTURAL
    else:
        spec_weights = np.array([scores[s] for s in valid_specs])
        spec_weights = spec_weights / spec_weights.sum()
        
        specialization = rng.choice(valid_specs, p=spec_weights)
    
    # Update count
    specialization_counts[specialization] = specialization_counts.get(specialization, 0) + 1
    
    return specialization


def place_fortresses(
    target_count,
    elevation,
    occupied_sites,
    land_mask,
    existing_settlements,
    size,
    rng
):
    """Place fortresses at strategic defensive positions."""
    fortresses = []
    
    # Fortresses prefer:
    # - High ground (defensibility)
    # - Near borders (implied by distance from other settlements)
    # - Mountain passes, river crossings, strategic chokepoints
    
    from utils.spatial import calculate_slope
    slope = calculate_slope(elevation)
    
    # Calculate strategic value
    strategic_value = np.zeros((size, size), dtype=np.float32)
    
    # High elevation is good
    strategic_value[elevation > 500] += 0.5
    strategic_value[elevation > 1000] += 0.3
    
    # Not too steep
    strategic_value[slope > 0.3] -= 0.5
    
    # Near existing settlements (to protect them)
    for settlement in existing_settlements:
        if settlement.settlement_type in [SettlementType.CITY, SettlementType.METROPOLIS]:
            # Create protection zone around cities
            dist_map = np.sqrt((np.arange(size)[:, None] - settlement.x)**2 +
                              (np.arange(size)[None, :] - settlement.y)**2)
            
            # Ideal distance: 30-80 cells from city
            protection_value = np.zeros_like(dist_map)
            protection_value[(dist_map > 30) & (dist_map < 80)] = 1.0
            protection_value[(dist_map > 20) & (dist_map <= 30)] = 0.7
            
            strategic_value += protection_value * 0.5
    
    strategic_value[~land_mask] = 0
    strategic_value[occupied_sites] = 0
    
    # Place fortresses
    min_spacing = 60
    attempts = 0
    max_attempts = target_count * 10
    
    while len(fortresses) < target_count and attempts < max_attempts:
        attempts += 1
        
        if strategic_value.max() < 0.1:
            break
        
        # Pick high-value location
        threshold = strategic_value.max() * 0.6
        candidates = np.argwhere(strategic_value > threshold)
        
        if len(candidates) == 0:
            break
        
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        
        # Check spacing from other fortresses
        too_close = False
        for f in fortresses:
            dist = np.sqrt((f.x - x)**2 + (f.y - y)**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if too_close:
            strategic_value[max(0, x-min_spacing//2):min(size, x+min_spacing//2),
                           max(0, y-min_spacing//2):min(size, y+min_spacing//2)] = 0
            continue
        
        # Create fortress
        population = rng.integers(50, 500)  # Variable garrison size
        age = rng.integers(100, 600)
        
        chunk_x = x // CHUNK_SIZE
        chunk_y = y // CHUNK_SIZE
        
        fortress = Settlement(
            settlement_id=len(existing_settlements) + len(fortresses),
            x=x,
            y=y,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            settlement_type=SettlementType.FORTRESS,
            population=population,
            specialization=SettlementSpecialization.FORTRESS_MILITARY,
            age_years=age,
            is_ruin=False,
            is_capital=False,
        )
        
        fortresses.append(fortress)
        
        # Mark as occupied
        occupied_sites[max(0, x-min_spacing):min(size, x+min_spacing),
                      max(0, y-min_spacing):min(size, y+min_spacing)] = True
        strategic_value[max(0, x-min_spacing):min(size, x+min_spacing),
                       max(0, y-min_spacing):min(size, y+min_spacing)] = 0
    
    return fortresses


def place_monasteries(
    target_count,
    elevation,
    mana_concentration,
    occupied_sites,
    land_mask,
    size,
    rng
):
    """Place monasteries in isolated, spiritually significant locations."""
    monasteries = []
    
    # Monasteries prefer:
    # - Isolation (far from other settlements)
    # - High mana concentration (spiritual energy)
    # - Moderate elevation (mountains or forests)
    
    spiritual_value = np.zeros((size, size), dtype=np.float32)
    
    # High mana is attractive
    spiritual_value += mana_concentration * 0.6
    
    # Moderate to high elevation
    spiritual_value[(elevation > 300) & (elevation < 2000)] += 0.4
    
    # Isolation bonus (far from occupied sites)
    distance_from_settlements = distance_transform_edt(~occupied_sites)
    isolation = np.clip(distance_from_settlements / 100.0, 0, 1)
    spiritual_value += isolation * 0.5
    
    spiritual_value[~land_mask] = 0
    spiritual_value[occupied_sites] = 0
    
    # Place monasteries
    min_spacing = 80
    attempts = 0
    max_attempts = target_count * 10
    
    while len(monasteries) < target_count and attempts < max_attempts:
        attempts += 1
        
        if spiritual_value.max() < 0.1:
            break
        
        threshold = spiritual_value.max() * 0.7
        candidates = np.argwhere(spiritual_value > threshold)
        
        if len(candidates) == 0:
            break
        
        idx = rng.integers(0, len(candidates))
        x, y = candidates[idx]
        
        # Check spacing
        too_close = False
        for m in monasteries:
            dist = np.sqrt((m.x - x)**2 + (m.y - y)**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if too_close:
            spiritual_value[max(0, x-min_spacing//2):min(size, x+min_spacing//2),
                           max(0, y-min_spacing//2):min(size, y+min_spacing//2)] = 0
            continue
        
        # Create monastery
        population = rng.integers(20, 200)
        age = rng.integers(50, 800)
        
        chunk_x = x // CHUNK_SIZE
        chunk_y = y // CHUNK_SIZE
        
        monastery = Settlement(
            settlement_id=1000000 + len(monasteries),  # Use high IDs to distinguish
            x=x,
            y=y,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            settlement_type=SettlementType.MONASTERY,
            population=population,
            specialization=SettlementSpecialization.RELIGIOUS,
            age_years=age,
            is_ruin=False,
            is_capital=False,
        )
        
        monasteries.append(monastery)
        
        # Mark as occupied
        occupied_sites[max(0, x-min_spacing):min(size, x+min_spacing),
                      max(0, y-min_spacing):min(size, y+min_spacing)] = True
        spiritual_value[max(0, x-min_spacing):min(size, x+min_spacing),
                       max(0, y-min_spacing):min(size, y+min_spacing)] = 0
    
    return monasteries


def create_settlement_map(settlements, size):
    """Create a visualization map of settlement types."""
    settlement_map = np.zeros((size, size), dtype=np.uint8)
    
    for s in settlements:
        # Map settlement type to visualization value
        if s.is_ruin:
            value = 8  # Ruins
        else:
            value = s.settlement_type + 1  # 1-7 for living settlements
        
        # Mark a small area around settlement (3x3 or 5x5 based on size)
        radius = 1 if s.settlement_type <= SettlementType.VILLAGE else 2
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = s.x + dx, s.y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    settlement_map[nx, ny] = value
    
    return settlement_map