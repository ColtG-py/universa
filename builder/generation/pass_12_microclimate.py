"""Pass 12: Microclimate"""
import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState

def execute(world_state: WorldState, params: WorldGenerationParams):
    """Calculate microclimate modifiers."""
    print(f"  - Calculating microclimate modifiers...")
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None: 
                continue
            chunk.microclimate_modifier = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
    
    print(f"  - Microclimate calculated")
