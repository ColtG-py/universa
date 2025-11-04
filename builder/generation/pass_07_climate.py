"""
World Builder - Pass 7: Climate Simulation
Calculates temperature and precipitation based on latitude, elevation, and wind.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Calculate climate (temperature and precipitation)."""
    print(f"  - Calculating climate patterns...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            chunk.temperature_c = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.precipitation_mm = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint16)
            
            for local_y in range(CHUNK_SIZE):
                global_y = chunk_y * CHUNK_SIZE + local_y
                latitude = (global_y / size - 0.5) * 180
                
                # Base temperature from latitude
                base_temp = params.base_temperature_c - abs(latitude) * 0.5
                
                # Base precipitation (more at equator and mid-latitudes)
                if abs(latitude) < 30:
                    base_precip = 2000  # Tropical
                elif abs(latitude) < 60:
                    base_precip = 1000  # Temperate
                else:
                    base_precip = 300   # Polar
                
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    
                    # Temperature decreases with elevation (lapse rate)
                    temp = base_temp - (max(elevation, 0) * 0.0065)
                    chunk.temperature_c[local_x, local_y] = temp
                    
                    # Precipitation affected by elevation and wind
                    if elevation > 1000:
                        # Orographic lift increases precipitation
                        precip = base_precip * 1.5
                    elif elevation < 0:
                        # Ocean
                        precip = 0
                    else:
                        precip = base_precip
                    
                    chunk.precipitation_mm[local_x, local_y] = int(np.clip(precip, 0, 5000))
    
    print(f"  - Climate calculated")
