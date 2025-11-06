"""
World Builder - Pass 5: Atmospheric Dynamics (IMPROVED v2)
Calculates wind patterns using the three-cell circulation model with Coriolis effect.

IMPROVEMENTS IN V2:
- Dynamic elevation thresholds based on actual world data (no magic numbers)
- Configurable percentile values for elevation categories
- Adapts to worlds with different elevation ranges
- Enhanced Coriolis curvature with Rossby waves
- Longitudinal variation and wave patterns
- Realistic swirling patterns like Earth

SCIENTIFIC BASIS:
The atmosphere is organized into three circulation cells per hemisphere:
1. Hadley Cell (0° to ~30°): Thermally direct, driven by equatorial heating
2. Ferrel Cell (~30° to ~60°): Mechanically driven by adjacent cells
3. Polar Cell (~60° to 90°): Thermally direct, driven by polar cooling

The Coriolis effect (planetary rotation) deflects moving air:
- Northern Hemisphere: Deflects air to the RIGHT
- Southern Hemisphere: Deflects air to the LEFT

Rossby waves create meandering patterns in the jet streams and westerlies,
producing the characteristic swirls and curves seen in Earth's atmosphere.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def calculate_wind_for_latitude(
    latitude: float,
    hemisphere: str,
    planetary_data,
    coriolis_strength: float = 1.0
) -> tuple:
    """
    Calculate wind direction and speed for a given latitude using three-cell model.
    
    Args:
        latitude: Absolute latitude (0-90)
        hemisphere: 'north' or 'south'
        planetary_data: Planetary data from world state
        coriolis_strength: Multiplier for Coriolis effect strength
    
    Returns:
        (wind_direction, wind_speed) in degrees and m/s
    """
    # Cell boundaries (can be adjusted based on planet parameters)
    # For Earth-like planets, use standard values
    hadley_max = 30.0  # Hadley cell extends to ~30°
    ferrel_max = 60.0  # Ferrel cell extends to ~60°
    
    # Determine which cell we're in and calculate wind patterns
    
    # =========================================================================
    # HADLEY CELL (0° to 30°)
    # =========================================================================
    if latitude < hadley_max:
        # Hadley cell: Air flows from 30° toward equator at surface
        # Coriolis deflects this flow westward → TRADE WINDS (easterlies)
        
        # Distance from equator (0-30)
        distance_from_equator = latitude / hadley_max
        
        # ITCZ (0°): Convergence zone, very light winds
        # Horse Latitudes (30°): Divergence zone, light winds
        # Peak winds around 15°
        
        # Wind speed peaks in middle of cell
        if distance_from_equator < 0.5:
            # Approaching ITCZ - winds weaken
            speed_factor = distance_from_equator * 2.0  # 0 to 1
        else:
            # Approaching horse latitudes - winds weaken
            speed_factor = (1.0 - distance_from_equator) * 2.0  # 1 to 0
        
        base_speed = 5.5 * speed_factor  # m/s (Trade winds: 3-7 m/s typical)
        
        # Wind direction: From subtropical high (30°) toward equator
        # Base direction is toward equator, then Coriolis deflects
        
        if hemisphere == 'north':
            # Northern Hemisphere: Coriolis deflects to RIGHT
            # Air moving south (180°) deflected right → from NE (45°)
            wind_direction = 45.0  # Northeast Trade Winds
        else:
            # Southern Hemisphere: Coriolis deflects to LEFT
            # Air moving north (0°) deflected left → from SE (135°)
            wind_direction = 135.0  # Southeast Trade Winds
    
    # =========================================================================
    # FERREL CELL (30° to 60°)
    # =========================================================================
    elif latitude < ferrel_max:
        # Ferrel cell: Air flows from 30° toward 60° at surface
        # Coriolis deflects this flow eastward → WESTERLIES
        
        # Distance within Ferrel cell (0-1)
        cell_latitude = latitude - hadley_max
        cell_range = ferrel_max - hadley_max
        distance_in_cell = cell_latitude / cell_range
        
        # Wind speed profile:
        # - Weak at 30° (horse latitudes - divergence)
        # - Strongest in mid-cell (40-50°)
        # - Moderate at 60° (subpolar low - convergence, but stormy)
        
        # Wind peaks around 45° latitude (middle of Ferrel cell)
        if distance_in_cell < 0.5:
            # Accelerating from horse latitudes
            speed_factor = distance_in_cell * 2.0  # 0 to 1
        else:
            # Approaching subpolar low - winds remain strong
            speed_factor = 0.8 + (1.0 - distance_in_cell) * 0.4  # 1 to 0.8
        
        # Westerlies are strongest wind belt (especially in southern hemisphere)
        base_speed = 8.0 * speed_factor  # m/s (Westerlies: 5-12 m/s typical)
        
        # Wind direction: From subtropical high (30°) toward subpolar low (60°)
        # Base direction is poleward, then Coriolis deflects
        
        if hemisphere == 'north':
            # Northern Hemisphere: Coriolis deflects to RIGHT
            # Air moving north (0°) deflected right → from SW (225°)
            wind_direction = 225.0  # Prevailing Westerlies (from southwest)
        else:
            # Southern Hemisphere: Coriolis deflects to LEFT
            # Air moving south (180°) deflected left → from NW (315°)
            wind_direction = 315.0  # Prevailing Westerlies (from northwest)
    
    # =========================================================================
    # POLAR CELL (60° to 90°)
    # =========================================================================
    else:
        # Polar cell: Air flows from pole (90°) toward 60° at surface
        # Coriolis deflects this flow westward → POLAR EASTERLIES
        
        # Distance from pole (0 at 90°, 1 at 60°)
        cell_latitude = latitude - ferrel_max
        cell_range = 90.0 - ferrel_max
        distance_from_pole = (90.0 - latitude) / cell_range
        
        # Wind speed profile:
        # - Weak at pole (90° - divergence, but cold/dense)
        # - Moderate in mid-cell
        # - Moderate at 60° (subpolar low - convergence)
        
        # Polar easterlies are weakest of the three wind belts
        if distance_from_pole < 0.5:
            # Near pole - very light winds
            speed_factor = 0.3 + distance_from_pole  # 0.3 to 0.8
        else:
            # Approaching subpolar low - winds strengthen slightly
            speed_factor = 0.8 + (1.0 - distance_from_pole) * 0.4  # 0.8 to 1.2
        
        base_speed = 4.0 * speed_factor  # m/s (Polar easterlies: 2-5 m/s typical)
        
        # Wind direction: From pole (90°) toward subpolar low (60°)
        # Base direction is equatorward, then Coriolis deflects
        
        if hemisphere == 'north':
            # Northern Hemisphere: Coriolis deflects to RIGHT
            # Air moving south (180°) deflected right → from NE (45°)
            wind_direction = 45.0  # Polar Easterlies (from northeast)
        else:
            # Southern Hemisphere: Coriolis deflects to LEFT
            # Air moving north (0°) deflected left → from SE (135°)
            wind_direction = 135.0  # Polar Easterlies (from southeast)
    
    # Apply Coriolis strength modifier
    wind_speed = base_speed * coriolis_strength
    
    return wind_direction, wind_speed


def calculate_rossby_wave_perturbation(
    longitude: float,
    latitude: float,
    hemisphere: str,
    world_size: int,
    seed: int,
    coriolis_strength: float
) -> tuple:
    """
    Calculate Rossby wave perturbation to create meandering wind patterns.
    
    Rossby waves are large-scale meandering patterns in the atmospheric flow
    caused by the conservation of angular momentum on a rotating sphere.
    They create the characteristic swirls and curves in Earth's wind patterns.
    
    Args:
        longitude: Longitude position (0-world_size)
        latitude: Absolute latitude (0-90)
        hemisphere: 'north' or 'south'
        world_size: Size of world in pixels
        seed: Random seed for wave generation
        coriolis_strength: Strength of Coriolis effect
    
    Returns:
        (direction_perturbation, speed_multiplier) in degrees and ratio
    """
    abs_latitude = abs(latitude)
    
    # Rossby waves are strongest in mid-latitudes (westerlies)
    # and weakest near equator and poles
    
    # Wave amplitude based on latitude
    if abs_latitude < 30:
        # Trade winds - moderate waves
        wave_amplitude = 25.0
        wave_number = 3  # Fewer, larger waves
    elif abs_latitude < 60:
        # Westerlies - STRONG meandering (jet stream region)
        wave_amplitude = 60.0  # Much stronger curvature
        wave_number = 5  # More complex wave patterns
    else:
        # Polar easterlies - weak waves
        wave_amplitude = 15.0
        wave_number = 2
    
    # Scale wave amplitude by Coriolis strength
    wave_amplitude *= coriolis_strength
    
    # Calculate normalized longitude (0 to 2π)
    lon_normalized = (longitude / world_size) * 2 * np.pi
    
    # Create wave pattern using multiple harmonics for complexity
    # This creates the meandering, swirling patterns seen on Earth
    
    # Use seed to create consistent but varied wave patterns
    np.random.seed(seed + int(latitude))
    phase_offset = np.random.random() * 2 * np.pi
    
    # Primary wave (largest scale)
    wave_1 = np.sin(wave_number * lon_normalized + phase_offset)
    
    # Secondary wave (medium scale) - offset phase
    wave_2 = 0.5 * np.sin((wave_number + 2) * lon_normalized + phase_offset + 1.0)
    
    # Tertiary wave (small scale turbulence)
    wave_3 = 0.25 * np.sin((wave_number + 5) * lon_normalized + phase_offset + 2.5)
    
    # Combine waves
    combined_wave = wave_1 + wave_2 + wave_3
    
    # Direction perturbation (degrees)
    # This creates the curved flow patterns
    direction_perturbation = combined_wave * wave_amplitude
    
    # Speed multiplier based on wave position
    # Winds are faster in troughs and ridges, slower in transitions
    wave_intensity = abs(combined_wave)
    speed_multiplier = 1.0 + (wave_intensity * 0.3)  # ±30% speed variation
    
    return direction_perturbation, speed_multiplier


def calculate_elevation_thresholds(world_state: WorldState, params: WorldGenerationParams) -> dict:
    """
    Calculate dynamic elevation thresholds based on actual world data.
    
    Uses percentiles instead of hardcoded values to adapt to different world scales.
    
    Args:
        world_state: World state containing elevation data
        params: Generation parameters with percentile configuration
    
    Returns:
        Dictionary with elevation threshold values
    """
    print(f"    - Calculating dynamic elevation thresholds...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    # Collect all elevation data from chunks
    all_elevations = []
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None or chunk.elevation is None:
                continue
            
            all_elevations.append(chunk.elevation.flatten())
    
    if not all_elevations:
        # Fallback to default values if no elevation data
        print(f"    - WARNING: No elevation data found, using defaults")
        return {
            'low_threshold': 500.0,
            'mid_threshold': 2000.0,
            'high_threshold': 3000.0,
            'disruptive_threshold': 3000.0,
        }
    
    elevations = np.concatenate(all_elevations)
    
    # Separate land and ocean
    land_mask = elevations > 0
    land_elevations = elevations[land_mask]
    
    if len(land_elevations) == 0:
        # No land, use ocean depths
        print(f"    - WARNING: No land found, using ocean depths for thresholds")
        ocean_elevations = np.abs(elevations[~land_mask])
        
        # Use ocean depth percentiles
        low_threshold = np.percentile(ocean_elevations, 
                                     getattr(params, 'terrain_low_percentile', 25))
        mid_threshold = np.percentile(ocean_elevations, 
                                     getattr(params, 'terrain_mid_percentile', 50))
        high_threshold = np.percentile(ocean_elevations, 
                                      getattr(params, 'terrain_high_percentile', 75))
        disruptive_threshold = np.percentile(ocean_elevations,
                                            getattr(params, 'terrain_disruptive_percentile', 90))
    else:
        # Use land elevation percentiles
        low_threshold = np.percentile(land_elevations, 
                                     getattr(params, 'terrain_low_percentile', 25))
        mid_threshold = np.percentile(land_elevations, 
                                     getattr(params, 'terrain_mid_percentile', 50))
        high_threshold = np.percentile(land_elevations, 
                                      getattr(params, 'terrain_high_percentile', 75))
        disruptive_threshold = np.percentile(land_elevations,
                                            getattr(params, 'terrain_disruptive_percentile', 90))
    
    thresholds = {
        'low_threshold': float(low_threshold),
        'mid_threshold': float(mid_threshold),
        'high_threshold': float(high_threshold),
        'disruptive_threshold': float(disruptive_threshold),
    }
    
    print(f"    - Elevation thresholds (meters):")
    print(f"      Low (p{getattr(params, 'terrain_low_percentile', 25)}): {low_threshold:.1f}m")
    print(f"      Mid (p{getattr(params, 'terrain_mid_percentile', 50)}): {mid_threshold:.1f}m")
    print(f"      High (p{getattr(params, 'terrain_high_percentile', 75)}): {high_threshold:.1f}m")
    print(f"      Disruptive (p{getattr(params, 'terrain_disruptive_percentile', 90)}): {disruptive_threshold:.1f}m")
    
    return thresholds


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate atmospheric circulation patterns using three-cell model with enhanced curvature.
    
    This creates realistic wind patterns based on:
    - Differential solar heating (equator vs poles)
    - Coriolis effect from planetary rotation
    - Pressure zones from rising/sinking air
    - Rossby waves creating meandering patterns
    - Dynamic terrain thresholds (no magic numbers)
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Calculating wind patterns using three-cell circulation model...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    # Calculate Coriolis strength modifier based on rotation rate
    # Faster rotation = stronger Coriolis effect
    # Earth baseline: 24 hour rotation
    coriolis_strength = (24.0 / params.rotation_hours) ** 0.5
    
    print(f"    - Rotation period: {params.rotation_hours:.1f} hours")
    print(f"    - Coriolis strength: {coriolis_strength:.2f}x Earth-normal")
    print(f"    - Generating Rossby wave patterns for curved flow...")
    
    # Calculate dynamic elevation thresholds from actual world data
    thresholds = calculate_elevation_thresholds(world_state, params)
    low_threshold = thresholds['low_threshold']
    mid_threshold = thresholds['mid_threshold']
    high_threshold = thresholds['high_threshold']
    disruptive_threshold = thresholds['disruptive_threshold']
    
    # Statistics tracking
    trade_wind_cells = 0
    westerly_cells = 0
    polar_cells = 0
    calm_cells = 0
    
    # Generate wind patterns for each chunk
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Initialize wind arrays
            chunk.wind_direction = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
            chunk.wind_speed = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            for local_y in range(CHUNK_SIZE):
                global_y = chunk_y * CHUNK_SIZE + local_y
                
                # Calculate latitude (-90 to +90)
                latitude_normalized = global_y / size  # 0 to 1
                latitude = (latitude_normalized - 0.5) * 180  # -90 to +90
                
                # Determine hemisphere
                hemisphere = 'north' if latitude >= 0 else 'south'
                abs_latitude = abs(latitude)
                
                # Calculate base wind from circulation model
                base_direction, base_speed = calculate_wind_for_latitude(
                    abs_latitude,
                    hemisphere,
                    planetary,
                    coriolis_strength
                )
                
                # Track statistics
                if abs_latitude < 30:
                    trade_wind_cells += 1
                elif abs_latitude < 60:
                    westerly_cells += 1
                else:
                    polar_cells += 1
                
                if base_speed < 2.0:
                    calm_cells += 1
                
                # NOW ADD LONGITUDINAL VARIATION (this creates the curves!)
                for local_x in range(CHUNK_SIZE):
                    global_x = chunk_x * CHUNK_SIZE + local_x
                    
                    # Calculate Rossby wave perturbation based on longitude
                    # This creates the characteristic meandering patterns
                    dir_perturbation, speed_multiplier = calculate_rossby_wave_perturbation(
                        global_x,
                        latitude,
                        hemisphere,
                        size,
                        params.seed,
                        coriolis_strength
                    )
                    
                    # Apply wave perturbation to wind direction
                    wind_direction = base_direction + dir_perturbation
                    wind_speed = base_speed * speed_multiplier
                    
                    # Get elevation for terrain modification
                    elevation = chunk.elevation[local_x, local_y] if chunk.elevation is not None else 0
                    
                    # Terrain modification using dynamic thresholds
                    if elevation > 0:
                        # LAND - Mountains disrupt wind flow
                        # Higher elevations = stronger disruption but also stronger winds aloft
                        
                        # Elevation factor for orographic deflection
                        # Scale based on how close elevation is to disruptive threshold
                        elevation_factor = min(elevation / disruptive_threshold, 1.0)
                        
                        # Orographic deflection (in addition to Rossby waves)
                        # Use deterministic noise based on position
                        deflection_seed = (global_x * 73856093) ^ (global_y * 19349663) ^ params.seed
                        np.random.seed(deflection_seed % (2**26))
                        terrain_deflection = (np.random.random() - 0.5) * 20 * elevation_factor
                        
                        wind_direction = wind_direction + terrain_deflection
                        
                        # Elevation effect on speed - use dynamic thresholds
                        if elevation < low_threshold:
                            # Low elevations - significant surface friction
                            # Linearly interpolate from 60% to 100% of base speed
                            speed_modifier = 0.6 + (elevation / low_threshold) * 0.4  # 0.6 to 1.0
                        elif elevation < high_threshold:
                            # Mid elevations - normal to slightly increased
                            # Linearly interpolate from 100% to 130%
                            range_size = high_threshold - low_threshold
                            progress = (elevation - low_threshold) / range_size
                            speed_modifier = 1.0 + progress * 0.3  # 1.0 to 1.3
                        else:
                            # High elevations - enhanced winds
                            # Cap at 180% for very high elevations
                            range_size = disruptive_threshold - high_threshold
                            if range_size > 0:
                                progress = min((elevation - high_threshold) / range_size, 1.0)
                                speed_modifier = 1.3 + progress * 0.5  # 1.3 to 1.8
                            else:
                                speed_modifier = 1.3
                        
                        wind_speed = wind_speed * speed_modifier
                    else:
                        # OCEAN - minimal friction, consistent winds
                        # Ocean winds are typically stronger and more consistent
                        depth = abs(elevation)
                        
                        # Deep ocean has slightly stronger winds (less disruption)
                        # Use mid_threshold as the depth cutoff for "deep ocean"
                        if depth > mid_threshold:
                            wind_speed = wind_speed * 1.1
                        else:
                            wind_speed = wind_speed * 1.05
                    
                    # Normalize wind direction to 0-360
                    wind_direction = wind_direction % 360
                    
                    # Store in chunk
                    chunk.wind_direction[local_x, local_y] = int(wind_direction)
                    chunk.wind_speed[local_x, local_y] = wind_speed
    
    # Print statistics
    total_cells = num_chunks * num_chunks * CHUNK_SIZE * CHUNK_SIZE
    
    print(f"  - Wind belt distribution:")
    print(f"    Trade Winds (0-30°): {trade_wind_cells/total_cells*100:.1f}%")
    print(f"    Westerlies (30-60°): {westerly_cells/total_cells*100:.1f}%")
    print(f"    Polar Easterlies (60-90°): {polar_cells/total_cells*100:.1f}%")
    print(f"    Calm zones (<2 m/s): {calm_cells/total_cells*100:.1f}%")
    print(f"  - Wind patterns generated with enhanced Rossby wave curvature")