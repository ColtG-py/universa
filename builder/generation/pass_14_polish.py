"""Pass 14: Final Polish"""
from config import WorldGenerationParams
from models.world import WorldState
from datetime import datetime

def execute(world_state: WorldState, params: WorldGenerationParams):
    """Apply final polish to generated world."""
    print(f"  - Applying final polish...")
    
    # Mark all chunks as generated
    for chunk in world_state.chunks.values():
        chunk.generated_at = datetime.utcnow()
    
    print(f"  - Polish complete")
