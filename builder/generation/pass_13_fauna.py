"""
World Builder - Pass 13: Fauna Distribution (FIXED VERSION)
Generates wildlife populations and migration patterns based on biomes and resources.

FIXES APPLIED:
1. Apex territories: Lowered thresholds using percentile-based approach
2. Migration routes: Improved detection with combined herbivore populations
3. Added extensive diagnostic output for debugging

SCIENTIFIC BASIS:
- Carrying capacity from Net Primary Productivity (NPP)
- Predator-prey biomass pyramids (10:1 herbivore:carnivore ratio)
- Territory size scaling with body mass (home range theory)
- Migration driven by resource seasonality
- Distance-to-water constraints for most terrestrial species
- Biome-specific fauna assemblages
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, label, binary_dilation, binary_erosion
from typing import Dict, List, Tuple

from config import WorldGenerationParams, CHUNK_SIZE, BiomeType, FaunaCategory
from models.world import WorldState
from utils.noise import NoiseGenerator

# Biome-specific fauna templates (relative abundance by category)
BIOME_FAUNA_TEMPLATES = {
    BiomeType.TROPICAL_RAINFOREST: {
        FaunaCategory.HERBIVORE_BROWSER: 0.8,
        FaunaCategory.HERBIVORE_MIXED: 0.6,
        FaunaCategory.PREDATOR_SMALL: 0.7,
        FaunaCategory.PREDATOR_MEDIUM: 0.4,
        FaunaCategory.PREDATOR_APEX: 0.3,
        FaunaCategory.OMNIVORE: 0.9,
        FaunaCategory.AVIAN_SONGBIRD: 1.0,
        FaunaCategory.AVIAN_RAPTOR: 0.5,
        FaunaCategory.INSECT: 1.0,
        FaunaCategory.AQUATIC_AMPHIBIAN: 0.9,
    },
    BiomeType.TROPICAL_SEASONAL_FOREST: {
        FaunaCategory.HERBIVORE_BROWSER: 0.7,
        FaunaCategory.HERBIVORE_GRAZER: 0.5,
        FaunaCategory.PREDATOR_MEDIUM: 0.6,
        FaunaCategory.PREDATOR_APEX: 0.4,
        FaunaCategory.OMNIVORE: 0.7,
        FaunaCategory.AVIAN_SONGBIRD: 0.8,
        FaunaCategory.AVIAN_MIGRATORY: 0.6,
        FaunaCategory.INSECT: 0.9,
    },
    BiomeType.SAVANNA: {
        FaunaCategory.HERBIVORE_GRAZER: 1.0,
        FaunaCategory.PREDATOR_MEDIUM: 0.7,
        FaunaCategory.PREDATOR_APEX: 0.6,
        FaunaCategory.OMNIVORE: 0.5,
        FaunaCategory.AVIAN_RAPTOR: 0.8,
        FaunaCategory.AVIAN_MIGRATORY: 0.7,
        FaunaCategory.INSECT: 0.8,
    },
    BiomeType.TEMPERATE_DECIDUOUS_FOREST: {
        FaunaCategory.HERBIVORE_BROWSER: 0.8,
        FaunaCategory.HERBIVORE_GRAZER: 0.4,
        FaunaCategory.PREDATOR_SMALL: 0.8,
        FaunaCategory.PREDATOR_MEDIUM: 0.6,
        FaunaCategory.PREDATOR_APEX: 0.3,
        FaunaCategory.OMNIVORE: 0.8,
        FaunaCategory.AVIAN_SONGBIRD: 0.9,
        FaunaCategory.AVIAN_MIGRATORY: 0.8,
        FaunaCategory.INSECT: 0.9,
    },
    BiomeType.TEMPERATE_RAINFOREST: {
        FaunaCategory.HERBIVORE_BROWSER: 0.9,
        FaunaCategory.PREDATOR_MEDIUM: 0.7,
        FaunaCategory.PREDATOR_APEX: 0.5,
        FaunaCategory.OMNIVORE: 0.8,
        FaunaCategory.AVIAN_SONGBIRD: 1.0,
        FaunaCategory.AQUATIC_AMPHIBIAN: 0.8,
        FaunaCategory.AQUATIC_FISH: 0.9,
        FaunaCategory.INSECT: 1.0,
    },
    BiomeType.TEMPERATE_GRASSLAND: {
        FaunaCategory.HERBIVORE_GRAZER: 0.9,
        FaunaCategory.PREDATOR_SMALL: 0.7,
        FaunaCategory.PREDATOR_MEDIUM: 0.5,
        FaunaCategory.PREDATOR_APEX: 0.3,
        FaunaCategory.AVIAN_RAPTOR: 0.7,
        FaunaCategory.AVIAN_MIGRATORY: 0.6,
        FaunaCategory.INSECT: 0.8,
    },
    BiomeType.BOREAL_FOREST: {
        FaunaCategory.HERBIVORE_BROWSER: 0.7,
        FaunaCategory.PREDATOR_MEDIUM: 0.6,
        FaunaCategory.PREDATOR_APEX: 0.4,
        FaunaCategory.OMNIVORE: 0.6,
        FaunaCategory.AVIAN_SONGBIRD: 0.6,
        FaunaCategory.AVIAN_MIGRATORY: 0.7,
        FaunaCategory.INSECT: 0.6,
    },
    BiomeType.TUNDRA: {
        FaunaCategory.HERBIVORE_GRAZER: 0.5,
        FaunaCategory.HERBIVORE_BROWSER: 0.4,
        FaunaCategory.PREDATOR_MEDIUM: 0.4,
        FaunaCategory.PREDATOR_APEX: 0.3,
        FaunaCategory.AVIAN_MIGRATORY: 0.8,
        FaunaCategory.INSECT: 0.4,
    },
    BiomeType.HOT_DESERT: {
        FaunaCategory.HERBIVORE_GRAZER: 0.3,
        FaunaCategory.PREDATOR_SMALL: 0.5,
        FaunaCategory.PREDATOR_MEDIUM: 0.2,
        FaunaCategory.OMNIVORE: 0.4,
        FaunaCategory.AVIAN_RAPTOR: 0.5,
        FaunaCategory.INSECT: 0.5,
    },
    BiomeType.COLD_DESERT: {
        FaunaCategory.HERBIVORE_GRAZER: 0.4,
        FaunaCategory.PREDATOR_SMALL: 0.4,
        FaunaCategory.PREDATOR_MEDIUM: 0.3,
        FaunaCategory.AVIAN_RAPTOR: 0.4,
        FaunaCategory.INSECT: 0.3,
    },
    BiomeType.MEDITERRANEAN: {
        FaunaCategory.HERBIVORE_GRAZER: 0.6,
        FaunaCategory.HERBIVORE_BROWSER: 0.5,
        FaunaCategory.PREDATOR_SMALL: 0.7,
        FaunaCategory.PREDATOR_MEDIUM: 0.4,
        FaunaCategory.OMNIVORE: 0.7,
        FaunaCategory.AVIAN_SONGBIRD: 0.8,
        FaunaCategory.AVIAN_MIGRATORY: 0.7,
        FaunaCategory.INSECT: 0.8,
    },
    BiomeType.MANGROVE: {
        FaunaCategory.AQUATIC_FISH: 1.0,
        FaunaCategory.AQUATIC_AMPHIBIAN: 0.8,
        FaunaCategory.AQUATIC_WATERFOWL: 0.9,
        FaunaCategory.AVIAN_SONGBIRD: 0.7,
        FaunaCategory.INSECT: 0.9,
    },
    # Ocean biomes have primarily aquatic fauna
    BiomeType.OCEAN_CORAL_REEF: {
        FaunaCategory.AQUATIC_FISH: 1.0,
    },
    BiomeType.OCEAN_SHELF: {
        FaunaCategory.AQUATIC_FISH: 0.9,
        FaunaCategory.AQUATIC_WATERFOWL: 0.6,
    },
    BiomeType.OCEAN_SHALLOW: {
        FaunaCategory.AQUATIC_FISH: 0.8,
    },
    BiomeType.OCEAN_DEEP: {
        FaunaCategory.AQUATIC_FISH: 0.5,
    },
    BiomeType.OCEAN_TRENCH: {
        FaunaCategory.AQUATIC_FISH: 0.3,
    },
}


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate fauna distribution across the world.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating fauna distribution...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    seed = params.seed
    
    # STEP 1: Collect global environmental data
    print(f"    - Collecting environmental data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    biome_global = np.zeros((size, size), dtype=np.uint8)
    vegetation_density_global = np.zeros((size, size), dtype=np.float32)
    temp_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    river_global = np.zeros((size, size), dtype=bool)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.biome_type is not None:
                biome_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.biome_type
            if chunk.vegetation_density is not None:
                vegetation_density_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.vegetation_density
            if chunk.temperature_c is not None:
                temp_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.temperature_c
            if chunk.precipitation_mm is not None:
                precip_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
            if chunk.river_presence is not None:
                river_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.river_presence
    
    # STEP 2: Calculate distance to water
    print(f"    - Calculating distance to water sources...")
    
    land_mask = elevation_global > 0
    water_sources = river_global | ~land_mask
    distance_to_water = distance_transform_edt(~water_sources)
    
    # STEP 3: Calculate Net Primary Productivity (NPP) as carrying capacity base
    print(f"    - Calculating Net Primary Productivity...")
    
    npp_global = calculate_npp(
        temp_global,
        precip_global,
        vegetation_density_global,
        biome_global,
        land_mask
    )
    
    # STEP 4: Calculate terrain ruggedness for species that prefer rough terrain
    print(f"    - Calculating terrain ruggedness...")
    
    from utils.spatial import calculate_slope
    slope_global = calculate_slope(elevation_global)
    
    # Ruggedness: combination of slope variation
    from scipy.ndimage import generic_filter
    ruggedness_global = generic_filter(slope_global, np.std, size=5)
    
    # STEP 5: Calculate base carrying capacity for each fauna category
    print(f"    - Calculating carrying capacity by fauna category...")
    
    carrying_capacity = calculate_carrying_capacity(
        npp_global,
        distance_to_water,
        biome_global,
        land_mask,
        size
    )
    
    # STEP 6: Generate fauna density maps
    print(f"    - Generating fauna density distributions...")
    
    rng = np.random.default_rng(seed + 13000)
    
    fauna_density_maps = {}
    
    for fauna_cat in FaunaCategory:
        print(f"      - {fauna_cat.name}...")
        
        density_map = generate_fauna_density(
            fauna_cat,
            carrying_capacity,
            biome_global,
            vegetation_density_global,
            distance_to_water,
            ruggedness_global,
            temp_global,
            precip_global,
            elevation_global,
            land_mask,
            rng
        )
        
        fauna_density_maps[fauna_cat] = density_map
    
    # STEP 7: Generate apex predator territories (FIXED)
    print(f"    - Generating apex predator territories...")
    
    apex_territories = generate_apex_territories(
        fauna_density_maps[FaunaCategory.PREDATOR_APEX],
        land_mask,
        size,
        rng
    )
    
    # STEP 8: Generate migration routes for herbivores (FIXED)
    print(f"    - Generating herbivore migration routes...")
    
    migration_routes = generate_migration_routes(
        fauna_density_maps[FaunaCategory.HERBIVORE_GRAZER],
        fauna_density_maps[FaunaCategory.HERBIVORE_BROWSER],
        elevation_global,
        temp_global,
        vegetation_density_global,
        land_mask,
        size,
        rng
    )
    
    # STEP 9: Store fauna data in chunks
    print(f"    - Storing fauna data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize fauna density dictionary
            chunk.fauna_density = {}
            
            for fauna_cat in FaunaCategory:
                chunk.fauna_density[fauna_cat] = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Initialize other fauna data
            chunk.apex_predator_territories = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint32)
            chunk.migration_routes = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x >= size or global_y >= size:
                        continue
                    
                    # Store fauna densities
                    for fauna_cat in FaunaCategory:
                        chunk.fauna_density[fauna_cat][local_x, local_y] = fauna_density_maps[fauna_cat][global_x, global_y]
                    
                    # Store territories
                    chunk.apex_predator_territories[local_x, local_y] = apex_territories[global_x, global_y]
                    
                    # Store migration routes
                    chunk.migration_routes[local_x, local_y] = migration_routes[global_x, global_y]
    
    # STEP 10: Calculate statistics
    print(f"  - Fauna distribution statistics:")
    
    for fauna_cat in [FaunaCategory.HERBIVORE_GRAZER, FaunaCategory.HERBIVORE_BROWSER,
                      FaunaCategory.PREDATOR_APEX, FaunaCategory.AQUATIC_FISH]:
        density_map = fauna_density_maps[fauna_cat]
        non_zero = density_map[density_map > 0]
        
        if len(non_zero) > 0:
            print(f"    {fauna_cat.name:25s}: Coverage {(len(non_zero) / density_map.size * 100):5.1f}%, "
                  f"Mean density {non_zero.mean():.3f}")
    
    num_territories = len(np.unique(apex_territories)) - 1  # Exclude 0
    print(f"    Apex predator territories: {num_territories}")
    
    migration_coverage = migration_routes.sum() / land_mask.sum() * 100 if land_mask.sum() > 0 else 0
    print(f"    Migration route coverage: {migration_coverage:.1f}% of land")
    
    print(f"  - Fauna distribution complete")


def calculate_npp(
    temp: np.ndarray,
    precip: np.ndarray,
    vegetation_density: np.ndarray,
    biome: np.ndarray,
    land_mask: np.ndarray
) -> np.ndarray:
    """
    Calculate Net Primary Productivity (NPP) as a proxy for carrying capacity.
    
    NPP is determined by temperature, precipitation, and vegetation density.
    
    Args:
        temp: Temperature array (°C)
        precip: Precipitation array (mm/year)
        vegetation_density: Vegetation density (0-1)
        biome: Biome type array
        land_mask: Boolean mask of land cells
        
    Returns:
        NPP array (arbitrary units, 0-1 scale)
    """
    npp = np.zeros_like(temp)
    
    # Temperature factor (optimum around 20-25°C)
    temp_factor = np.zeros_like(temp)
    temp_factor = np.where(
        (temp >= 15) & (temp <= 30),
        1.0,
        np.where(
            temp < 15,
            np.clip(temp / 15, 0, 1),
            np.clip(1.0 - (temp - 30) / 20, 0, 1)
        )
    )
    
    # Precipitation factor (more water = more growth, up to a point)
    precip_factor = np.clip(precip / 2000, 0, 1)
    
    # Combine factors
    npp = temp_factor * precip_factor * vegetation_density
    
    # Boost for highly productive biomes
    for biome_type in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_RAINFOREST]:
        npp = np.where(biome == biome_type, npp * 1.3, npp)
    
    # Reduce for low-productivity biomes
    for biome_type in [BiomeType.TUNDRA, BiomeType.HOT_DESERT, BiomeType.COLD_DESERT, BiomeType.ICE]:
        npp = np.where(biome == biome_type, npp * 0.3, npp)
    
    # Zero out ocean
    npp = np.where(land_mask, npp, 0)
    
    return npp


def calculate_carrying_capacity(
    npp: np.ndarray,
    distance_to_water: np.ndarray,
    biome: np.ndarray,
    land_mask: np.ndarray,
    size: int
) -> np.ndarray:
    """
    Calculate base carrying capacity from NPP and water availability.
    
    Returns:
        Carrying capacity array (biomass units)
    """
    # Base capacity from NPP
    capacity = npp.copy()
    
    # Water availability penalty
    # Most terrestrial animals need water within 10km (10 cells)
    water_factor = np.exp(-distance_to_water / 10.0)
    capacity = capacity * (0.3 + 0.7 * water_factor)
    
    # Smooth for realistic gradients
    capacity = gaussian_filter(capacity, sigma=2.0)
    
    return capacity


def generate_fauna_density(
    fauna_cat: FaunaCategory,
    carrying_capacity: np.ndarray,
    biome: np.ndarray,
    vegetation_density: np.ndarray,
    distance_to_water: np.ndarray,
    ruggedness: np.ndarray,
    temp: np.ndarray,
    precip: np.ndarray,
    elevation: np.ndarray,
    land_mask: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate fauna density for a specific category.
    
    Returns:
        Density map (relative population density, 0-1 scale)
    """
    size = carrying_capacity.shape[0]
    density = np.zeros((size, size), dtype=np.float32)
    
    # Start with carrying capacity as base
    density = carrying_capacity.copy()
    
    # Apply biome-specific preferences
    for biome_type, fauna_prefs in BIOME_FAUNA_TEMPLATES.items():
        if fauna_cat in fauna_prefs:
            preference = fauna_prefs[fauna_cat]
            density = np.where(biome == biome_type, density * preference, density)
        else:
            # Not present in this biome
            density = np.where(biome == biome_type, 0, density)
    
    # Category-specific habitat preferences
    if fauna_cat in [FaunaCategory.HERBIVORE_GRAZER]:
        # Prefer open grasslands
        density *= (1.0 - vegetation_density * 0.5)
    
    elif fauna_cat in [FaunaCategory.HERBIVORE_BROWSER]:
        # Prefer forests
        density *= (0.5 + vegetation_density * 0.5)
    
    elif fauna_cat in [FaunaCategory.PREDATOR_APEX]:
        # Need large territories, prefer remote areas
        # Reduce density overall (low population)
        density *= 0.1
        # Prefer rugged terrain
        ruggedness_norm = ruggedness / (ruggedness.max() + 0.001)
        density *= (0.7 + ruggedness_norm * 0.3)
    
    elif fauna_cat in [FaunaCategory.PREDATOR_MEDIUM, FaunaCategory.PREDATOR_SMALL]:
        # Predators follow prey (herbivores)
        # Calculate herbivore density
        herbivore_density = carrying_capacity * vegetation_density
        herbivore_density = gaussian_filter(herbivore_density, sigma=3.0)
        
        # Predators are ~10% of herbivore biomass
        density = herbivore_density * 0.1
    
    elif fauna_cat == FaunaCategory.AQUATIC_FISH:
        # Fish in water bodies
        density = np.where(~land_mask, 0.8, 0)  # Ocean
        density = np.where(distance_to_water < 2, 0.6, density)  # Rivers/lakes
    
    elif fauna_cat == FaunaCategory.AQUATIC_WATERFOWL:
        # Near water
        water_proximity = np.exp(-distance_to_water / 5.0)
        density *= water_proximity
    
    elif fauna_cat == FaunaCategory.AVIAN_MIGRATORY:
        # Prefer temperate/boreal zones
        latitude = np.linspace(-90, 90, size)
        latitude_field = np.tile(latitude[np.newaxis, :], (size, 1))
        
        # Peak at 30-60 degrees
        migration_factor = np.where(
            (np.abs(latitude_field) >= 30) & (np.abs(latitude_field) <= 60),
            1.0,
            0.5
        )
        density *= migration_factor
    
    # Add random variation
    noise_gen = NoiseGenerator(
        seed=rng.integers(0, 1000000),
        octaves=4,
        persistence=0.5,
        scale=size / 8.0
    )
    variation = noise_gen.generate_perlin_2d(size, size, normalize=True)
    density *= (0.7 + variation * 0.6)
    
    # Smooth and normalize
    density = gaussian_filter(density, sigma=1.5)
    density = np.clip(density, 0, 1)
    
    return density


def generate_apex_territories(
    apex_density: np.ndarray,
    land_mask: np.ndarray,
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate territorial boundaries for apex predators.
    
    FIXED: Uses percentile-based thresholds instead of fixed values to handle sparse populations.
    
    Returns:
        Territory ID map (0 = no territory, 1+ = territory IDs)
    """
    territories = np.zeros((size, size), dtype=np.uint32)
    
    # Find suitable territories using percentile approach
    land_density = apex_density[land_mask]
    
    if len(land_density) == 0 or land_density.max() < 0.001:
        print(f"      - No apex predator population found (max density: {apex_density.max():.4f})")
        return territories
    
    # Use top 30% of areas with any apex presence
    density_threshold = np.percentile(land_density[land_density > 0.001], 70)
    density_threshold = max(density_threshold, 0.01)  # Minimum floor
    
    suitable = (apex_density > density_threshold) & land_mask
    
    if not suitable.any():
        print(f"      - No suitable territory areas (threshold: {density_threshold:.3f}, max: {apex_density.max():.3f})")
        return territories
    
    print(f"      - Apex density range: {land_density.min():.4f} to {land_density.max():.4f}")
    print(f"      - Territory threshold: {density_threshold:.4f}")
    print(f"      - Suitable area: {suitable.sum() / land_mask.sum() * 100:.1f}% of land")
    
    # Place territory centers
    territory_centers = []
    
    # Find high-density areas for territory centers (top 40% of suitable areas)
    suitable_density = apex_density[suitable]
    if len(suitable_density) > 0:
        center_threshold = np.percentile(suitable_density, 60)
    else:
        center_threshold = density_threshold
    
    candidates = np.argwhere((apex_density >= center_threshold) & suitable)
    
    print(f"      - Territory center candidates: {len(candidates)}")
    
    if len(candidates) == 0:
        print(f"      - No territory centers found")
        return territories
    
    # Space territories - scale with world size
    min_distance = max(25, size // 20)  # ~25-50 cells depending on size
    max_territories = max(5, size // 100)  # Reasonable number of territories
    
    print(f"      - Minimum territory spacing: {min_distance} cells")
    print(f"      - Maximum territories: {max_territories}")
    
    # Shuffle candidates for random placement
    rng.shuffle(candidates)
    
    for candidate in candidates:
        if len(territory_centers) >= max_territories:
            break
        
        x, y = candidate
        
        # Check distance to existing centers
        too_close = False
        for center in territory_centers:
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            territory_centers.append((x, y))
    
    num_territories = len(territory_centers)
    print(f"      - Placed {num_territories} apex predator territories")
    
    if num_territories == 0:
        return territories
    
    # Expand territories using Voronoi-like approach
    print(f"      - Expanding territories...")
    
    # Pre-calculate coordinates for efficiency
    coords = np.array(np.meshgrid(np.arange(size), np.arange(size), indexing='ij'))
    
    for i, (cx, cy) in enumerate(territory_centers):
        territory_id = i + 1
        
        # Calculate distance from this center
        distances_sq = (coords[0] - cx)**2 + (coords[1] - cy)**2
        
        # Only assign if on land and has some apex presence
        mask = land_mask & (apex_density > 0.001)
        
        if i == 0:
            # First territory - claim everything in range
            in_range = (distances_sq < (min_distance * 1.5)**2) & mask
            territories[in_range] = territory_id
        else:
            # Subsequent territories - only claim if closest and in range
            max_radius = min_distance * 2
            potential = (distances_sq < max_radius**2) & mask
            
            if not potential.any():
                continue
            
            # For each potential cell, check if this is the closest center
            pot_coords = np.argwhere(potential)
            for px, py in pot_coords:
                # Find nearest center
                min_dist_sq = float('inf')
                nearest_id = 0
                
                for j, (ox, oy) in enumerate(territory_centers):
                    d_sq = (px - ox)**2 + (py - oy)**2
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq
                        nearest_id = j + 1
                
                # Assign if this is the nearest territory and within range
                if nearest_id == territory_id and min_dist_sq < max_radius**2:
                    territories[px, py] = territory_id
    
    # Smooth territory boundaries
    for tid in range(1, num_territories + 1):
        territory_mask = territories == tid
        # Dilate slightly then erode to smooth
        territory_mask = binary_dilation(territory_mask, iterations=2)
        territory_mask = binary_erosion(territory_mask, iterations=1)
        territory_mask = territory_mask & land_mask
        territories[territory_mask] = tid
    
    # Count territory sizes
    for tid in range(1, num_territories + 1):
        size_cells = (territories == tid).sum()
        print(f"        Territory {tid}: {size_cells} cells")
    
    return territories


def generate_migration_routes(
    herbivore_grazer_density: np.ndarray,
    herbivore_browser_density: np.ndarray,
    elevation: np.ndarray,
    temp: np.ndarray,
    vegetation_density: np.ndarray,
    land_mask: np.ndarray,
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate seasonal migration routes for herbivores.
    
    FIXED: Uses combined herbivore populations and percentile-based thresholds
    to ensure migration routes are generated even with sparse populations.
    
    Returns:
        Boolean array marking migration corridors
    """
    migration_routes = np.zeros((size, size), dtype=bool)
    
    # Combine herbivore populations for better coverage
    total_herbivore = herbivore_grazer_density + herbivore_browser_density
    land_herbivores = total_herbivore * land_mask
    
    if land_herbivores.max() < 0.001:
        print(f"      - No significant herbivore populations for migration")
        return migration_routes
    
    print(f"      - Herbivore density range: {land_herbivores[land_mask].min():.4f} to {land_herbivores[land_mask].max():.4f}")
    
    # Use percentile-based approach for elevation zones
    land_elev = elevation[land_mask]
    elev_p75 = np.percentile(land_elev, 75)
    elev_p50 = np.percentile(land_elev, 50)
    elev_p25 = np.percentile(land_elev, 25)
    
    # Temperature-based seasonal ranges (more robust than just elevation)
    temp_land = temp[land_mask]
    temp_p75 = np.percentile(temp_land, 75)  # Warm areas
    temp_p50 = np.percentile(temp_land, 50)  # Median temperature
    temp_p25 = np.percentile(temp_land, 25)  # Cool areas
    
    # SUMMER RANGES: High elevation OR cool temperatures + good vegetation + herbivores
    # Lowered herbivore threshold from 0.3 to 0.15
    summer_range = (
        ((elevation > elev_p75) | (temp < temp_p25)) &
        (vegetation_density > 0.3) &
        (total_herbivore > 0.15) &
        land_mask
    )
    
    # WINTER RANGES: Low elevation + warm temperatures + good vegetation + herbivores
    winter_range = (
        (elevation < elev_p25) &
        (temp > temp_p50) &
        (vegetation_density > 0.25) &
        (total_herbivore > 0.15) &
        land_mask
    )
    
    summer_area = summer_range.sum()
    winter_area = winter_range.sum()
    
    print(f"      - Summer ranges: {summer_area} cells ({summer_area / land_mask.sum() * 100:.1f}% of land)")
    print(f"      - Winter ranges: {winter_area} cells ({winter_area / land_mask.sum() * 100:.1f}% of land)")
    
    if summer_area == 0 or winter_area == 0:
        print(f"      - Insufficient seasonal range area, using high-density herbivore corridors")
        # Still mark high-density herbivore areas as potential corridors
        migration_routes = (total_herbivore > 0.2) & land_mask
        migration_routes = binary_dilation(migration_routes, iterations=3)
        final_coverage = migration_routes.sum()
        print(f"      - Fallback migration routes: {final_coverage} cells ({final_coverage / land_mask.sum() * 100:.1f}% of land)")
        return migration_routes
    
    # Find connected regions
    summer_regions, num_summer = label(summer_range)
    winter_regions, num_winter = label(winter_range)
    
    print(f"      - Summer regions: {num_summer}, Winter regions: {num_winter}")
    
    # MIGRATION CORRIDORS: Areas between elevation zones with decent herbivore presence
    mid_elevation = (elevation >= elev_p25) & (elevation <= elev_p75) & land_mask
    
    # Lower threshold for corridors - top 50% of herbivore areas in mid-elevation
    mid_elev_herbivores = total_herbivore[mid_elevation]
    if len(mid_elev_herbivores) > 0 and mid_elev_herbivores.max() > 0:
        corridor_threshold = np.percentile(mid_elev_herbivores[mid_elev_herbivores > 0], 50)
        corridor_threshold = max(corridor_threshold, 0.1)  # Minimum threshold
    else:
        corridor_threshold = 0.1
    
    corridors = (
        mid_elevation &
        (total_herbivore > corridor_threshold) &
        (vegetation_density > 0.2)
    )
    
    print(f"      - Migration corridor threshold: {corridor_threshold:.3f}")
    print(f"      - Initial corridor cells: {corridors.sum()}")
    
    # Combine all migration areas
    migration_routes = summer_range | winter_range | corridors
    
    # Dilate to create connected pathways
    migration_routes = binary_dilation(migration_routes, iterations=3)
    
    # Ensure connection between summer and winter ranges
    # Add high-density herbivore pathways
    high_herbivore_paths = (total_herbivore > 0.25) & land_mask
    high_herbivore_paths = binary_dilation(high_herbivore_paths, iterations=2)
    migration_routes = migration_routes | high_herbivore_paths
    
    # Final dilation for smooth corridors
    migration_routes = binary_dilation(migration_routes, iterations=2)
    
    final_coverage = migration_routes.sum()
    print(f"      - Final migration routes: {final_coverage} cells ({final_coverage / land_mask.sum() * 100:.1f}% of land)")
    
    return migration_routes