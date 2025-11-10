"""
World Builder - Pass 14: Natural Resources
Identifies harvestable resources for civilizations - minerals, timber, crops, fisheries.

SCIENTIFIC BASIS:
- Mineral deposits correlate with geology and tectonic activity
- Timber quality depends on climate, soil, and biome type
- Agricultural productivity follows soil fertility and climate
- Fisheries depend on water bodies and aquatic productivity

RESOURCE CATEGORIES:
1. Mining: Concentrated ore veins, quarries, gemstones
2. Timber: Hardwood vs softwood, based on forest type
3. Agriculture: Fertile farmland with yield estimates
4. Fisheries: River and coastal fishing grounds
5. Rare Resources: Gemstones, magical materials (low frequency)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from config import (
    WorldGenerationParams,
    CHUNK_SIZE,
    Mineral,
    RockType,
    BiomeType,
    TimberType,
    QuarryType,
)
from models.world import WorldState
from utils.noise import NoiseGenerator


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate natural resource distributions for economic gameplay.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating natural resource distributions...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    seed = params.seed
    
    # STEP 1: Collect global data
    print(f"    - Collecting environmental data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    temp_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    bedrock_global = np.zeros((size, size), dtype=np.uint8)
    biome_global = np.zeros((size, size), dtype=np.uint8)
    soil_ph_global = np.zeros((size, size), dtype=np.float32)
    soil_drainage_global = np.zeros((size, size), dtype=np.uint8)
    vegetation_density_global = np.zeros((size, size), dtype=np.float32)
    canopy_height_global = np.zeros((size, size), dtype=np.float32)
    agricultural_suitability_global = np.zeros((size, size), dtype=np.float32)
    river_global = np.zeros((size, size), dtype=bool)
    tectonic_stress_global = np.zeros((size, size), dtype=np.float32)
    mineral_richness_global = {}
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.temperature_c is not None:
                temp_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.temperature_c
            if chunk.precipitation_mm is not None:
                precip_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
            if chunk.bedrock_type is not None:
                bedrock_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.bedrock_type
            if chunk.biome_type is not None:
                biome_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
            if chunk.soil_ph is not None:
                soil_ph_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.soil_ph
            if chunk.soil_drainage is not None:
                soil_drainage_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.soil_drainage
            if chunk.vegetation_density is not None:
                vegetation_density_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.vegetation_density
            if chunk.forest_canopy_height is not None:
                canopy_height_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.forest_canopy_height
            if chunk.agricultural_suitability is not None:
                agricultural_suitability_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.agricultural_suitability
            if chunk.river_presence is not None:
                river_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
            if chunk.tectonic_stress is not None:
                tectonic_stress_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.tectonic_stress
            
            # Collect mineral richness
            if chunk.mineral_richness is not None:
                for mineral, data in chunk.mineral_richness.items():
                    if mineral not in mineral_richness_global:
                        mineral_richness_global[mineral] = np.zeros((size, size), dtype=np.float32)
                    mineral_richness_global[mineral][x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = data
    
    land_mask = elevation_global > 0
    ocean_mask = elevation_global <= 0
    
    # STEP 2: Generate concentrated mineral deposits (veins)
    print(f"    - Concentrating mineral ore veins...")
    
    mineral_deposits_global = generate_mineral_deposits(
        mineral_richness_global,
        tectonic_stress_global,
        elevation_global,
        land_mask,
        seed
    )
    
    # STEP 3: Generate quarry sites (building stone)
    print(f"    - Identifying quarry sites for building materials...")
    
    quarry_quality_global, quarry_type_global = generate_quarry_sites(
        bedrock_global,
        elevation_global,
        land_mask,
        seed
    )
    
    # STEP 4: Generate timber resources
    print(f"    - Assessing timber resources...")
    
    timber_quality_global, timber_type_global = generate_timber_resources(
        biome_global,
        vegetation_density_global,
        canopy_height_global,
        temp_global,
        precip_global,
        elevation_global
    )
    
    # STEP 5: Generate agricultural zones
    print(f"    - Mapping agricultural productivity...")
    
    agricultural_yield_global = generate_agricultural_zones(
        agricultural_suitability_global,
        soil_ph_global,
        soil_drainage_global,
        temp_global,
        precip_global,
        elevation_global,
        land_mask
    )
    
    # STEP 6: Generate fishing grounds
    print(f"    - Locating fishing grounds...")
    
    fishing_quality_global = generate_fishing_grounds(
        river_global,
        ocean_mask,
        elevation_global,
        temp_global,
        biome_global
    )
    
    # STEP 7: Generate rare resources (gemstones, magical materials)
    print(f"    - Placing rare and magical resources...")
    
    rare_resources_global = generate_rare_resources(
        mineral_deposits_global,
        tectonic_stress_global,
        biome_global,
        elevation_global,
        land_mask,
        seed
    )
    
    # STEP 8: Calculate resource accessibility
    print(f"    - Evaluating resource accessibility...")
    
    accessibility_global = calculate_resource_accessibility(
        elevation_global,
        land_mask
    )
    
    # STEP 9: Store results in chunks
    print(f"    - Storing resource data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.mineral_deposits = {}
            for mineral in Mineral:
                chunk.mineral_deposits[mineral] = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            chunk.quarry_quality = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.quarry_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.timber_quality = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.timber_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            chunk.agricultural_yield = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.fishing_quality = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.rare_resources = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.resource_accessibility = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Copy data from global arrays
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    # Mineral deposits
                    for mineral in Mineral:
                        if mineral in mineral_deposits_global:
                            chunk.mineral_deposits[mineral][local_x, local_y] = mineral_deposits_global[mineral][global_x, global_y]
                    
                    # Other resources
                    chunk.quarry_quality[local_x, local_y] = quarry_quality_global[global_x, global_y]
                    chunk.quarry_type[local_x, local_y] = quarry_type_global[global_x, global_y]
                    chunk.timber_quality[local_x, local_y] = timber_quality_global[global_x, global_y]
                    chunk.timber_type[local_x, local_y] = timber_type_global[global_x, global_y]
                    chunk.agricultural_yield[local_x, local_y] = agricultural_yield_global[global_x, global_y]
                    chunk.fishing_quality[local_x, local_y] = fishing_quality_global[global_x, global_y]
                    chunk.rare_resources[local_x, local_y] = rare_resources_global[global_x, global_y]
                    chunk.resource_accessibility[local_x, local_y] = accessibility_global[global_x, global_y]
    
    # STEP 10: Report statistics
    print(f"  - Resource generation statistics:")
    
    # Count cells with significant resources
    land_cells = land_mask.sum()
    
    # Mineral deposits
    total_mineral_cells = 0
    for mineral in Mineral:
        if mineral in mineral_deposits_global:
            mineral_cells = (mineral_deposits_global[mineral] > 0.3).sum()
            if mineral_cells > 0:
                percentage = (mineral_cells / land_cells * 100)
                print(f"    {mineral.name:15s} deposits: {percentage:5.2f}% of land")
                total_mineral_cells += mineral_cells
    
    # Timber
    timber_cells = (timber_quality_global > 0.5).sum()
    if timber_cells > 0:
        print(f"    High-quality timber: {timber_cells / land_cells * 100:5.2f}% of land")
    
    # Agriculture
    fertile_cells = (agricultural_yield_global > 0.7).sum()
    if fertile_cells > 0:
        print(f"    Fertile farmland: {fertile_cells / land_cells * 100:5.2f}% of land")
    
    # Fishing
    fishing_cells = (fishing_quality_global > 0.5).sum()
    if fishing_cells > 0:
        total_cells = size * size
        print(f"    Quality fishing grounds: {fishing_cells / total_cells * 100:5.2f}% of world")
    
    # Rare resources
    rare_cells = (rare_resources_global > 0.5).sum()
    if rare_cells > 0:
        print(f"    Rare resource sites: {rare_cells} locations ({rare_cells / land_cells * 100:5.3f}% of land)")
    
    print(f"  - Natural resources generated")


def generate_mineral_deposits(
    mineral_richness_global: dict,
    tectonic_stress: np.ndarray,
    elevation: np.ndarray,
    land_mask: np.ndarray,
    seed: int
) -> dict:
    """
    Generate concentrated mineral ore veins from diffuse mineral richness.
    
    Uses local maxima detection to create distinct ore veins.
    """
    mineral_deposits = {}
    
    for mineral, richness in mineral_richness_global.items():
        # Apply concentration algorithm
        # Use maximum filter to find local peaks
        local_max = maximum_filter(richness, size=5)
        
        # Cells that are local maxima and have significant richness
        is_peak = (richness == local_max) & (richness > 0.2)
        
        # Apply tectonic stress bonus (high stress = more concentrated deposits)
        stress_bonus = tectonic_stress * 0.5
        
        # Calculate deposit quality (0-1 scale)
        deposit_quality = np.zeros_like(richness)
        deposit_quality[is_peak] = richness[is_peak] + stress_bonus[is_peak]
        deposit_quality[is_peak] = np.clip(deposit_quality[is_peak], 0, 1)
        
        # Apply land mask (no deposits in ocean)
        deposit_quality[~land_mask] = 0
        
        # Smooth slightly to create vein-like structures
        deposit_quality = gaussian_filter(deposit_quality, sigma=2.0)
        
        mineral_deposits[mineral] = deposit_quality
    
    return mineral_deposits


def generate_quarry_sites(
    bedrock: np.ndarray,
    elevation: np.ndarray,
    land_mask: np.ndarray,
    seed: int
) -> tuple:
    """
    Identify quality quarry sites for building stone extraction.
    
    Returns:
        Tuple of (quarry_quality, quarry_type) arrays
    """
    size = bedrock.shape[0]
    
    quarry_quality = np.zeros((size, size), dtype=np.float32)
    quarry_type = np.zeros((size, size), dtype=np.uint8)
    
    # Noise for variation
    noise_gen = NoiseGenerator(seed=seed + 14000, scale=size / 10.0, octaves=4)
    variation = noise_gen.generate_perlin_2d(size, size, normalize=True)
    
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                continue
            
            rock_type = RockType(bedrock[x, y])
            elev = elevation[x, y]
            
            # Determine quarry type and base quality from bedrock
            if rock_type == RockType.IGNEOUS:
                # Granite, basalt - excellent building stone
                quarry_type[x, y] = QuarryType.GRANITE
                base_quality = 0.8
            
            elif rock_type == RockType.METAMORPHIC:
                # Marble, slate - high value decorative stone
                quarry_type[x, y] = QuarryType.MARBLE
                base_quality = 0.9
            
            elif rock_type == RockType.LIMESTONE:
                # Limestone - good general building stone
                quarry_type[x, y] = QuarryType.LIMESTONE
                base_quality = 0.7
            
            elif rock_type == RockType.SEDIMENTARY:
                # Sandstone - moderate quality
                quarry_type[x, y] = QuarryType.SANDSTONE
                base_quality = 0.6
            
            else:
                base_quality = 0.5
                quarry_type[x, y] = QuarryType.SANDSTONE
            
            # Accessibility modifier (lower elevation = easier access)
            if elev < 500:
                access_modifier = 1.0
            elif elev < 1500:
                access_modifier = 0.8
            else:
                access_modifier = 0.5
            
            # Add variation
            quality = base_quality * access_modifier * (0.8 + variation[x, y] * 0.4)
            quarry_quality[x, y] = np.clip(quality, 0, 1)
    
    return quarry_quality, quarry_type


def generate_timber_resources(
    biome: np.ndarray,
    vegetation_density: np.ndarray,
    canopy_height: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    elevation: np.ndarray
) -> tuple:
    """
    Generate timber quality and type based on forest characteristics.
    
    Returns:
        Tuple of (timber_quality, timber_type) arrays
    """
    size = biome.shape[0]
    
    timber_quality = np.zeros((size, size), dtype=np.float32)
    timber_type = np.zeros((size, size), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            biome_type = BiomeType(biome[x, y])
            veg_density = vegetation_density[x, y]
            height = canopy_height[x, y]
            temp = temperature[x, y]
            precip = precipitation[x, y]
            
            # Only forested biomes have significant timber
            forest_biomes = [
                BiomeType.TROPICAL_RAINFOREST,
                BiomeType.TROPICAL_SEASONAL_FOREST,
                BiomeType.TEMPERATE_RAINFOREST,
                BiomeType.TEMPERATE_DECIDUOUS_FOREST,
                BiomeType.BOREAL_FOREST,
                BiomeType.MEDITERRANEAN,
            ]
            
            if biome_type not in forest_biomes or height < 5:
                timber_quality[x, y] = 0
                timber_type[x, y] = TimberType.NONE
                continue
            
            # Determine timber type based on biome and climate
            if biome_type in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TROPICAL_SEASONAL_FOREST]:
                # Tropical hardwoods (mahogany, teak)
                timber_type[x, y] = TimberType.TROPICAL_HARDWOOD
                base_quality = 0.9
            
            elif biome_type == BiomeType.TEMPERATE_DECIDUOUS_FOREST:
                # Temperate hardwoods (oak, maple, walnut)
                timber_type[x, y] = TimberType.HARDWOOD
                base_quality = 0.85
            
            elif biome_type == BiomeType.TEMPERATE_RAINFOREST:
                # Mix of hardwood and softwood
                if temp > 12:
                    timber_type[x, y] = TimberType.HARDWOOD
                    base_quality = 0.8
                else:
                    timber_type[x, y] = TimberType.SOFTWOOD
                    base_quality = 0.75
            
            elif biome_type == BiomeType.BOREAL_FOREST:
                # Softwoods (pine, spruce, fir)
                timber_type[x, y] = TimberType.SOFTWOOD
                base_quality = 0.7
            
            elif biome_type == BiomeType.MEDITERRANEAN:
                # Mediterranean hardwoods (olive, cork oak)
                timber_type[x, y] = TimberType.HARDWOOD
                base_quality = 0.75
            
            else:
                timber_type[x, y] = TimberType.SOFTWOOD
                base_quality = 0.6
            
            # Quality modifiers
            density_modifier = veg_density
            height_modifier = min(height / 30.0, 1.0)  # Taller = better timber
            
            quality = base_quality * density_modifier * height_modifier
            timber_quality[x, y] = np.clip(quality, 0, 1)
    
    return timber_quality, timber_type


def generate_agricultural_zones(
    agricultural_suitability: np.ndarray,
    soil_ph: np.ndarray,
    soil_drainage: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    elevation: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    Generate agricultural productivity zones with yield estimates.
    
    Returns:
        Array of agricultural yield (0-1 scale, tons per hectare equivalent)
    """
    size = agricultural_suitability.shape[0]
    
    agricultural_yield = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                continue
            
            base_suitability = agricultural_suitability[x, y]
            
            if base_suitability < 0.3:
                # Too poor for agriculture
                agricultural_yield[x, y] = 0
                continue
            
            # Climate optimality (temperate climates best for general crops)
            temp = temperature[x, y]
            precip = precipitation[x, y]
            
            if 10 <= temp <= 20 and 500 <= precip <= 1200:
                climate_modifier = 1.0
            elif 5 <= temp <= 25 and 400 <= precip <= 1500:
                climate_modifier = 0.85
            elif 0 <= temp <= 30 and 300 <= precip <= 2000:
                climate_modifier = 0.7
            else:
                climate_modifier = 0.5
            
            # Soil pH optimality
            ph = soil_ph[x, y]
            if 6.0 <= ph <= 7.5:
                ph_modifier = 1.0
            elif 5.5 <= ph <= 8.0:
                ph_modifier = 0.85
            else:
                ph_modifier = 0.6
            
            # Elevation penalty
            elev = elevation[x, y]
            if elev < 500:
                elev_modifier = 1.0
            elif elev < 1000:
                elev_modifier = 0.8
            elif elev < 1500:
                elev_modifier = 0.6
            else:
                elev_modifier = 0.3
            
            # Calculate final yield
            yield_value = (base_suitability * climate_modifier * 
                          ph_modifier * elev_modifier)
            
            agricultural_yield[x, y] = np.clip(yield_value, 0, 1)
    
    # Smooth agricultural zones slightly
    agricultural_yield = gaussian_filter(agricultural_yield, sigma=1.5)
    
    return agricultural_yield


def generate_fishing_grounds(
    river: np.ndarray,
    ocean_mask: np.ndarray,
    elevation: np.ndarray,
    temperature: np.ndarray,
    biome: np.ndarray
) -> np.ndarray:
    """
    Generate fishing quality for rivers and coastal areas.
    
    Returns:
        Array of fishing quality (0-1 scale)
    """
    size = river.shape[0]
    
    fishing_quality = np.zeros((size, size), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            # River fishing
            if river[x, y]:
                # Rivers have moderate fishing potential
                base_quality = 0.6
                
                # Temperature affects fish diversity
                temp = temperature[x, y]
                if 10 <= temp <= 20:
                    temp_modifier = 1.0
                elif 5 <= temp <= 25:
                    temp_modifier = 0.85
                else:
                    temp_modifier = 0.6
                
                fishing_quality[x, y] = base_quality * temp_modifier
            
            # Coastal fishing
            elif ocean_mask[x, y]:
                depth = -elevation[x, y]
                temp = temperature[x, y]
                biome_type = BiomeType(biome[x, y])
                
                # Continental shelf (shallow ocean) best for fishing
                if biome_type == BiomeType.OCEAN_SHELF:
                    base_quality = 0.9
                elif biome_type == BiomeType.OCEAN_SHALLOW:
                    base_quality = 0.7
                elif biome_type == BiomeType.OCEAN_CORAL_REEF:
                    base_quality = 0.95  # Reefs have highest biodiversity
                elif biome_type == BiomeType.OCEAN_DEEP:
                    base_quality = 0.5
                elif biome_type == BiomeType.OCEAN_TRENCH:
                    base_quality = 0.2  # Deep ocean has less life
                else:
                    base_quality = 0.6
                
                # Temperature affects productivity
                if 15 <= temp <= 25:
                    temp_modifier = 1.0
                elif 10 <= temp <= 28:
                    temp_modifier = 0.9
                elif 5 <= temp <= 30:
                    temp_modifier = 0.7
                else:
                    temp_modifier = 0.5
                
                fishing_quality[x, y] = base_quality * temp_modifier
    
    # Smooth fishing grounds
    fishing_quality = gaussian_filter(fishing_quality, sigma=2.0)
    
    return fishing_quality


def generate_rare_resources(
    mineral_deposits: dict,
    tectonic_stress: np.ndarray,
    biome: np.ndarray,
    elevation: np.ndarray,
    land_mask: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Generate rare and magical resources at low frequency.
    
    These are "dungeon" locations with high-value resources.
    
    Returns:
        Array of rare resource quality (0-1 scale)
    """
    size = land_mask.shape[0]
    
    rare_resources = np.zeros((size, size), dtype=np.float32)
    
    rng = np.random.default_rng(seed + 14500)
    
    # Create noise for rare resource probability
    noise_gen = NoiseGenerator(seed=seed + 14600, scale=size / 5.0, octaves=6)
    rarity_noise = noise_gen.generate_perlin_2d(size, size, normalize=True)
    
    # Identify potential rare resource locations
    # 1. High tectonic stress (deep mines with rare ores)
    # 2. Ancient forests (magical materials)
    # 3. Mountain peaks (crystal formations)
    
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                continue
            
            stress = tectonic_stress[x, y]
            biome_type = BiomeType(biome[x, y])
            elev = elevation[x, y]
            noise = rarity_noise[x, y]
            
            rare_quality = 0.0
            
            # High tectonic stress + high noise = rare mineral deposits
            if stress > 0.7 and noise > 0.85:
                # Check if there's already a mineral deposit here
                has_minerals = False
                for mineral, deposits in mineral_deposits.items():
                    if deposits[x, y] > 0.5:
                        has_minerals = True
                        break
                
                if has_minerals:
                    # Rare mineral variant (gemstones, magical ores)
                    rare_quality = 0.7 + rng.random() * 0.3
            
            # Ancient forests with very high vegetation = magical groves
            elif biome_type in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_RAINFOREST]:
                if noise > 0.92:
                    rare_quality = 0.6 + rng.random() * 0.4
            
            # Mountain peaks = crystal formations
            elif elev > 2500 and noise > 0.9:
                rare_quality = 0.7 + rng.random() * 0.3
            
            rare_resources[x, y] = rare_quality
    
    return rare_resources


def calculate_resource_accessibility(
    elevation: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    Calculate how accessible resources are (affects extraction difficulty).
    
    Returns:
        Array of accessibility (0-1 scale, where 1 = easy access)
    """
    size = elevation.shape[0]
    
    accessibility = np.ones((size, size), dtype=np.float32)
    
    # Calculate slope
    from utils.spatial import calculate_slope
    slope = calculate_slope(elevation)
    
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                accessibility[x, y] = 0
                continue
            
            elev = elevation[x, y]
            local_slope = slope[x, y]
            
            # Elevation penalty (higher = harder to reach)
            if elev < 500:
                elev_factor = 1.0
            elif elev < 1000:
                elev_factor = 0.9
            elif elev < 1500:
                elev_factor = 0.75
            elif elev < 2000:
                elev_factor = 0.6
            elif elev < 3000:
                elev_factor = 0.4
            else:
                elev_factor = 0.2
            
            # Slope penalty (steeper = harder to work)
            if local_slope < 0.1:
                slope_factor = 1.0
            elif local_slope < 0.2:
                slope_factor = 0.9
            elif local_slope < 0.3:
                slope_factor = 0.7
            elif local_slope < 0.5:
                slope_factor = 0.5
            else:
                slope_factor = 0.3
            
            accessibility[x, y] = elev_factor * slope_factor
    
    # Smooth accessibility
    accessibility = gaussian_filter(accessibility, sigma=2.0)
    
    return accessibility