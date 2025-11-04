"""
World Builder - Pass 8: Erosion Simulation
Applies erosion to terrain based on slope and precipitation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from utils.spatial import calculate_slope


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Simulate erosion effects on terrain."""
    print(f"  - Simulating erosion ({params.erosion_iterations} iterations)...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    planetary = world_state.planetary_data
    
    for iteration in range(params.erosion_iterations):
        for chunk_y in range(num_chunks):
            for chunk_x in range(num_chunks):
                chunk = world_state.get_chunk(chunk_x, chunk_y)
                if chunk is None:
                    continue
                
                slope = calculate_slope(chunk.elevation)
                precip = chunk.precipitation_mm / 1000.0  # Normalize
                
                # Erosion rate proportional to slope and rainfall
                erosion_rate = slope * precip * planetary.erosion_modifier * 0.01 * params.erosion_strength
                
                # Apply erosion
                chunk.elevation = chunk.elevation - erosion_rate
                
                # Smooth to simulate sediment deposition
                chunk.elevation = gaussian_filter(chunk.elevation, sigma=0.5)
    
    print(f"  - Erosion simulation complete")
