"""Pass 13: Geological Features"""
import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE, FeatureType
from models.world import WorldState, GeologicalFeature

def execute(world_state: WorldState, params: WorldGenerationParams):
    """Generate geological features."""
    print(f"  - Generating geological features...")
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    rng = np.random.default_rng(params.seed + 9000)
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None: 
                continue
            
            chunk.cave_presence = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=bool)
            
            # Randomly generate some features
            if params.enable_caves and rng.random() < 0.1:
                x, y = rng.integers(0, CHUNK_SIZE, 2)
                feature = GeologicalFeature(
                    type=FeatureType.CAVE_SYSTEM,
                    location_x=int(x), 
                    location_y=int(y),
                    chunk_x=chunk_x, 
                    chunk_y=chunk_y,
                    properties={"depth": int(rng.integers(10, 100))}
                )
                chunk.geological_features.append(feature)
    
    print(f"  - Features generated")
