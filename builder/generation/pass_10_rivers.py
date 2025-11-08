"""
World Builder - Pass 10: Surface Hydrology (PARTICLE-BASED)
Generates river networks using particle-based hydraulic simulation.

Based on SimpleHydrology by Nick McDonald:
https://nickmcd.me/2023/12/12/meandering-rivers-in-particle-based-hydraulic-erosion-simulations/

APPROACH:
- Spawn water particles proportional to precipitation
- Particles flow downhill following gradient descent
- Track discharge (accumulated volume) at each cell
- Track momentum for stream coherence
- Rivers form where discharge is high
- Exponential averaging for temporal smoothing

OPTIMIZED with Numba JIT compilation for performance.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from numba import jit

from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate river networks using particle-based flow simulation.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    print(f"  - Generating river networks (particle-based simulation)...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    # STEP 1: Collect global elevation and precipitation data
    print(f"    - Collecting elevation and precipitation data...")
    
    elevation_global = np.zeros((size, size), dtype=np.float32)
    precip_global = np.zeros((size, size), dtype=np.float32)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            if chunk.elevation is not None:
                elevation_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.elevation
            if chunk.precipitation_mm is not None:
                precip_global[x_start:x_start+CHUNK_SIZE, y_start:y_start+CHUNK_SIZE] = chunk.precipitation_mm
    
    # STEP 2: Initialize discharge and momentum tracking maps
    print(f"    - Initializing discharge and momentum tracking...")
    
    discharge_map = np.zeros((size, size), dtype=np.float32)
    discharge_track = np.zeros((size, size), dtype=np.float32)
    
    momentum_x_map = np.zeros((size, size), dtype=np.float32)
    momentum_y_map = np.zeros((size, size), dtype=np.float32)
    momentum_x_track = np.zeros((size, size), dtype=np.float32)
    momentum_y_track = np.zeros((size, size), dtype=np.float32)
    
    # STEP 3: Calculate gradients for flow direction
    print(f"    - Calculating elevation gradients...")
    
    # Use finite differences for gradient (more accurate than triangular mesh)
    grad_y, grad_x = np.gradient(elevation_global)
    
    # STEP 4: Simulate water particles
    print(f"    - Simulating water particle flow...")
    
    # INCREASED iterations for better channel formation
    num_iterations = 2  # Increased from 3
    particles_per_mm = 0.15  # Slightly increased for more samples
    
    land_mask = elevation_global > 0
    num_land_cells = land_mask.sum()
    
    print(f"    - Land cells: {num_land_cells:,}")
    print(f"    - Precipitation range: {precip_global[land_mask].min():.1f} - {precip_global[land_mask].max():.1f} mm")
    
    for iteration in range(num_iterations):
        # Reset tracking for this iteration
        discharge_track.fill(0)
        momentum_x_track.fill(0)
        momentum_y_track.fill(0)
        
        # Spawn and simulate particles (JIT-optimized)
        simulate_all_particles(
            size,
            land_mask,
            precip_global,
            particles_per_mm,
            elevation_global,
            grad_x,
            grad_y,
            discharge_map,
            momentum_x_map,
            momentum_y_map,
            discharge_track,
            momentum_x_track,
            momentum_y_track
        )
        
        # Exponential averaging for temporal smoothing
        # INCREASED learning rate for faster channel formation
        learning_rate = 0.5  # Increased from 0.1
        
        discharge_map = (1.0 - learning_rate) * discharge_map + learning_rate * discharge_track
        momentum_x_map = (1.0 - learning_rate) * momentum_x_map + learning_rate * momentum_x_track
        momentum_y_map = (1.0 - learning_rate) * momentum_y_map + learning_rate * momentum_y_track
        
        if (iteration + 1) % 2 == 0 or iteration == num_iterations - 1:
            print(f"      Iteration {iteration + 1}/{num_iterations} complete")
            print(f"        Discharge range: {discharge_map[land_mask].min():.6f} - {discharge_map[land_mask].max():.6f}")
    
    # STEP 5: Apply MINIMAL smoothing to preserve channels
    print(f"    - Applying minimal smoothing to preserve channels...")
    
    # Much lighter smoothing to preserve narrow river channels
    discharge_map = gaussian_filter(discharge_map, sigma=0.01)  # Reduced from 1.5
    
    print(f"    - Post-smoothing discharge range: {discharge_map[land_mask].min():.6f} - {discharge_map[land_mask].max():.6f}")
    
    # STEP 6: Normalize discharge using logarithmic scaling
    print(f"    - Normalizing discharge with log scaling...")
    
    discharge_normalized = np.zeros_like(discharge_map)
    
    # Use logarithmic normalization to handle wide range
    discharge_log = np.log10(discharge_map + 1.0)
    
    # Normalize to [0, 1] range
    log_min = discharge_log[land_mask].min()
    log_max = discharge_log[land_mask].max()
    
    if log_max > log_min:
        discharge_normalized[land_mask] = (discharge_log[land_mask] - log_min) / (log_max - log_min)
    
    print(f"    - Log discharge range: {log_min:.6f} - {log_max:.6f}")
    print(f"    - Normalized discharge mean: {discharge_normalized[land_mask].mean():.6f}")
    print(f"    - Normalized discharge median: {np.median(discharge_normalized[land_mask]):.6f}")
    
    # STEP 7: Determine rivers based on discharge threshold
    print(f"    - Identifying river channels...")
    
    # Get discharge values only on land
    land_discharge = discharge_normalized[land_mask]
    
    # Calculate percentile thresholds
    p90 = np.percentile(land_discharge, 90)
    p95 = np.percentile(land_discharge, 95)
    p97 = np.percentile(land_discharge, 97)
    p99 = np.percentile(land_discharge, 99)
    
    print(f"    - Discharge percentiles:")
    print(f"      90th: {p90:.6f}")
    print(f"      95th: {p95:.6f}")
    print(f"      97th: {p97:.6f}")
    print(f"      99th: {p99:.6f}")
    
    # Use 95th percentile for more visible river networks
    river_threshold = p95
    
    print(f"    - River threshold (95th percentile): {river_threshold:.6f}")
    
    # Only mark rivers on land cells that exceed threshold
    river_presence = np.logical_and(land_mask, discharge_normalized > river_threshold)
    
    num_river_cells = river_presence.sum()
    print(f"    - River cells identified: {num_river_cells:,} ({num_river_cells / num_land_cells * 100:.2f}% of land)")
    
    # Calculate river flow magnitude (m³/s)
    # Use original discharge values, scaled appropriately
    river_flow = np.where(river_presence, discharge_map * 0.001, 0.0)
    
    # STEP 8: Store results in chunks
    print(f"    - Storing river data in chunks...")
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            
            # Initialize arrays
            chunk.river_presence = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            chunk.river_flow = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.drainage_basin_id = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint32)
            
            # Store discharge for groundwater pass
            if not hasattr(chunk, 'discharge'):
                chunk.discharge = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            # Copy data
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    global_x = x_start + local_x
                    global_y = y_start + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.river_presence[local_x, local_y] = river_presence[global_x, global_y]
                        chunk.river_flow[local_x, local_y] = river_flow[global_x, global_y]
                        chunk.discharge[local_x, local_y] = discharge_normalized[global_x, global_y]
    
    # STEP 9: Calculate statistics
    if num_river_cells > 0:
        rivers = discharge_normalized[river_presence]
        
        print(f"  - River network statistics:")
        print(f"    Total river cells: {num_river_cells:,}")
        print(f"    River coverage: {num_river_cells / num_land_cells * 100:.2f}% of land")
        print(f"    Mean discharge: {rivers.mean():.3f}")
        print(f"    Max discharge: {rivers.max():.3f}")
        print(f"    Mean flow: {river_flow[river_presence].mean():.1f} m³/s")
        print(f"    Max flow: {river_flow[river_presence].max():.1f} m³/s")
    else:
        print(f"  - WARNING: No river network generated!")
    
    print(f"  - River networks generated (particle-based method)")


@jit(nopython=True)
def simulate_all_particles(
    size: int,
    land_mask: np.ndarray,
    precip_global: np.ndarray,
    particles_per_mm: float,
    elevation_global: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    discharge_map: np.ndarray,
    momentum_x_map: np.ndarray,
    momentum_y_map: np.ndarray,
    discharge_track: np.ndarray,
    momentum_x_track: np.ndarray,
    momentum_y_track: np.ndarray
):
    """
    JIT-compiled function to spawn and simulate all particles.
    """
    # Spawn particles at each land cell proportional to precipitation
    for y in range(size):
        for x in range(size):
            if not land_mask[x, y]:
                continue
            
            precip = precip_global[x, y]
            num_particles = int(precip * particles_per_mm)
            
            # Use much smaller volume to prevent saturation
            volume = precip / 100000.0
            
            # Spawn multiple particles per cell for high precipitation
            for _ in range(max(1, num_particles)):
                simulate_particle(
                    x, y,
                    elevation_global,
                    grad_x, grad_y,
                    discharge_map,
                    momentum_x_map, momentum_y_map,
                    discharge_track,
                    momentum_x_track, momentum_y_track,
                    size,
                    volume
                )


@jit(nopython=True)
def simulate_particle(
    start_x: int,
    start_y: int,
    elevation: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    discharge_map: np.ndarray,
    momentum_x_map: np.ndarray,
    momentum_y_map: np.ndarray,
    discharge_track: np.ndarray,
    momentum_x_track: np.ndarray,
    momentum_y_track: np.ndarray,
    size: int,
    volume: float
):
    """
    Simulate a single water particle flowing downhill.
    JIT-compiled for performance.
    """
    # Particle state
    pos_x, pos_y = float(start_x), float(start_y)
    speed_x, speed_y = 0.0, 0.0
    
    # Simulation parameters
    max_steps = 5  # INCREASED from 300 to let particles flow further
    gravity = 1.0
    momentum_transfer = 10  # INCREASED from 0.3 for stronger channel coherence
    
    for step in range(max_steps):
        # Current integer position
        ix, iy = int(pos_x), int(pos_y)
        
        # Boundary check
        if ix < 0 or ix >= size or iy < 0 or iy >= size:
            break
        
        # Stop if we've reached ocean
        if elevation[ix, iy] <= 0:
            break
        
        # Get local gradient (negative for downhill)
        local_grad_x = -grad_x[ix, iy]
        local_grad_y = -grad_y[ix, iy]
        
        # Gravity force (downhill)
        speed_x += gravity * local_grad_x
        speed_y += gravity * local_grad_y
        
        # Get local stream momentum
        stream_momentum_x = momentum_x_map[ix, iy]
        stream_momentum_y = momentum_y_map[ix, iy]
        local_discharge = discharge_map[ix, iy]
        
        # Apply momentum transfer from stream (coupling to other particles)
        if stream_momentum_x != 0 or stream_momentum_y != 0:
            # Momentum transfer proportional to discharge
            stream_magnitude = np.sqrt(stream_momentum_x**2 + stream_momentum_y**2)
            
            if stream_magnitude > 0:
                # Normalize stream momentum
                stream_dir_x = stream_momentum_x / stream_magnitude
                stream_dir_y = stream_momentum_y / stream_magnitude
                
                # Transfer momentum from stream to particle
                transfer_strength = momentum_transfer * local_discharge
                speed_x += transfer_strength * stream_dir_x
                speed_y += transfer_strength * stream_dir_y
        
        # Accumulate discharge at current cell
        discharge_track[ix, iy] += volume
        
        # Accumulate momentum at current cell
        momentum_x_track[ix, iy] += volume * speed_x
        momentum_y_track[ix, iy] += volume * speed_y
        
        # Normalize speed to move exactly one cell per step
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
        
        if speed_magnitude < 0.001:
            # Stagnant - stop simulation
            break
        
        # Normalize to unit speed
        speed_x /= speed_magnitude
        speed_y /= speed_magnitude
        
        # Move particle
        pos_x += speed_x
        pos_y += speed_y
        
        # If we've moved very little, we're in a local minimum
        if abs(speed_x) < 0.01 and abs(speed_y) < 0.01:
            break