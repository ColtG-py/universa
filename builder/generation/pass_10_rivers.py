"""
World Builder - Pass 10: Surface Hydrology
Generates river networks using flow accumulation.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from utils.spatial import calculate_flow_direction_d8, calculate_flow_accumulation


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Generate river networks."""
    print(f"  - Generating river networks...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Calculate flow direction
            flow_dir = calculate_flow_direction_d8(chunk.elevation)
            
            # Calculate flow accumulation with precipitation as weight
            precip_weight = chunk.precipitation_mm / 1000.0
            flow_accum = calculate_flow_accumulation(flow_dir, precip_weight)
            
            # Rivers form where flow accumulation is high
            threshold = np.percentile(flow_accum, 98)  # Top 2% becomes rivers
            
            chunk.river_presence = flow_accum > threshold
            chunk.river_flow = np.where(chunk.river_presence, flow_accum * 0.01, 0.0)
            
            # Initialize drainage basins (simplified)
            chunk.drainage_basin_id = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint32)
    
    print(f"  - River networks generated")
