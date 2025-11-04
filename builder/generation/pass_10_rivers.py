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
"""

import numpy as np
from scipy.ndimage import gaussian_filter

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
    
    # Number of particles per cell proportional to precipitation
    # More rain = more particles spawned
    num_iterations = 5  # Multiple passes for accumulation
    particles_per_mm = 0.02  # Particles per mm of precipitation
    
    land_mask = elevation_global > 0
    
    for iteration in range(num_iterations):
        # Reset tracking for this iteration
        discharge_track.fill(0)
        momentum_x_track.fill(0)
        momentum_y_track.fill(0)
        
        # Spawn particles at each land cell proportional to precipitation
        for y in range(size):
            for x in range(size):
                if not land_mask[x, y]:
                    continue
                
                precip = precip_global[x, y]
                num_particles = int(precip * particles_per_mm)
                
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
                        precip / 1000.0  # Convert mm to m³ volume
                    )
        
        # Exponential averaging for temporal smoothing
        # This creates smooth, persistent discharge patterns
        learning_rate = 0.1
        
        discharge_map = (1.0 - learning_rate) * discharge_map + learning_rate * discharge_track
        momentum_x_map = (1.0 - learning_rate) * momentum_x_map + learning_rate * momentum_x_track
        momentum_y_map = (1.0 - learning_rate) * momentum_y_map + learning_rate * momentum_y_track
        
        print(f"      Iteration {iteration + 1}/{num_iterations} complete")
    
    # STEP 5: Apply smoothing to discharge for natural river patterns
    print(f"    - Smoothing discharge patterns...")
    
    discharge_map = gaussian_filter(discharge_map, sigma=1.5)
    
    # STEP 6: Normalize discharge using error function for [0, 1] range
    # Higher discharge = more likely to be a river
    discharge_normalized = np.zeros_like(discharge_map)
    
    # Use error function for smooth normalization (as in the article)
    # This creates a sigmoid-like curve
    for y in range(size):
        for x in range(size):
            if land_mask[x, y]:
                # Scale factor controls activation threshold
                discharge_normalized[x, y] = np.tanh(0.4 * discharge_map[x, y])
    
    # STEP 7: Determine rivers based on discharge threshold
    print(f"    - Identifying river channels...")
    
    # Rivers form where discharge is in top percentile
    river_threshold = np.percentile(discharge_normalized[land_mask], 97)  # Top 3%
    
    river_presence = discharge_normalized > river_threshold
    
    # Calculate river flow magnitude (m³/s)
    # Proportional to discharge
    river_flow = np.where(river_presence, discharge_map * 10.0, 0.0)
    
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
    rivers = discharge_normalized[river_presence]
    
    if len(rivers) > 0:
        print(f"  - River network statistics:")
        print(f"    Total river cells: {river_presence.sum()}")
        print(f"    River coverage: {river_presence.sum() / land_mask.sum() * 100:.2f}% of land")
        print(f"    Mean discharge: {rivers.mean():.3f}")
        print(f"    Max discharge: {rivers.max():.3f}")
        print(f"    Mean flow: {river_flow[river_presence].mean():.1f} m³/s")
    
    print(f"  - River networks generated (particle-based method)")


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
    
    Args:
        start_x, start_y: Starting position
        elevation: Global elevation map
        grad_x, grad_y: Elevation gradients
        discharge_map, momentum maps: Current state
        discharge_track, momentum tracks: Accumulation buffers
        size: World size
        volume: Particle volume (water mass)
    """
    # Particle state
    pos_x, pos_y = float(start_x), float(start_y)
    speed_x, speed_y = 0.0, 0.0
    
    # Simulation parameters
    max_steps = 500
    gravity = 1.0
    momentum_transfer = 0.3
    
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
        speed_x += gravity * local_grad_x / volume
        speed_y += gravity * local_grad_y / volume
        
        # Get local stream momentum
        stream_momentum_x = momentum_x_map[ix, iy]
        stream_momentum_y = momentum_y_map[ix, iy]
        local_discharge = discharge_map[ix, iy]
        
        # Apply momentum transfer from stream (coupling to other particles)
        if stream_momentum_x != 0 or stream_momentum_y != 0:
            # Momentum transfer proportional to alignment with stream
            stream_magnitude = np.sqrt(stream_momentum_x**2 + stream_momentum_y**2)
            speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
            
            if stream_magnitude > 0 and speed_magnitude > 0:
                # Dot product for alignment
                alignment = (stream_momentum_x * speed_x + stream_momentum_y * speed_y) / (stream_magnitude * speed_magnitude)
                
                # Transfer momentum from stream to particle
                transfer_factor = momentum_transfer * alignment / (volume + local_discharge + 0.001)
                speed_x += transfer_factor * stream_momentum_x
                speed_y += transfer_factor * stream_momentum_y
        
        # Accumulate discharge at current cell
        discharge_track[ix, iy] += volume
        
        # Accumulate momentum at current cell
        momentum_x_track[ix, iy] += volume * speed_x
        momentum_y_track[ix, iy] += volume * speed_y
        
        # Normalize speed to move exactly one cell per step (dynamic time-step)
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