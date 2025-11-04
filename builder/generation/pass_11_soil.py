"""Pass 11: Soil Formation"""
import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE, SoilType, DrainageClass
from models.world import WorldState

def execute(world_state: WorldState, params: WorldGenerationParams):
    """Generate soil properties."""
    print(f"  - Generating soil properties...")
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None: 
                continue
            
            chunk.soil_type = np.full((CHUNK_SIZE, CHUNK_SIZE), SoilType.LOAM, dtype=np.uint8)
            chunk.soil_ph = np.full((CHUNK_SIZE, CHUNK_SIZE), 7.0, dtype=np.float32)
            chunk.soil_drainage = np.full((CHUNK_SIZE, CHUNK_SIZE), DrainageClass.MODERATELY_WELL, dtype=np.uint8)
    
    print(f"  - Soil generated")
