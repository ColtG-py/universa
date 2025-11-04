# Extending World Builder

This guide shows how to add new generation passes and extend the world generation system.

## Adding a New Generation Pass

### Step 1: Create the Pass Module

Create a new file in the `generation/` directory:

```python
# generation/pass_15_biomes.py
"""
Pass 15: Biome Classification
Classifies biomes based on temperature, precipitation, and elevation.
"""

import numpy as np
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState
from enum import IntEnum


class BiomeType(IntEnum):
    """Biome classifications"""
    TUNDRA = 0
    BOREAL_FOREST = 1
    TEMPERATE_FOREST = 2
    GRASSLAND = 3
    DESERT = 4
    TROPICAL_RAINFOREST = 5
    SAVANNA = 6
    OCEAN = 7


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Classify biomes based on climate and elevation.
    
    Uses temperature, precipitation, and elevation to determine biome type
    following a simplified Köppen climate classification.
    """
    print(f"  - Classifying biomes...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    biome_counts = {biome: 0 for biome in BiomeType}
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None:
                continue
            
            # Initialize biome array
            chunk.biome_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    temp = chunk.temperature_c[local_x, local_y]
                    precip = chunk.precipitation_mm[local_x, local_y]
                    
                    # Classify biome
                    if elevation < 0:
                        # Underwater
                        biome = BiomeType.OCEAN
                    elif temp < -10:
                        # Very cold
                        biome = BiomeType.TUNDRA
                    elif temp < 5:
                        # Cold
                        if precip > 400:
                            biome = BiomeType.BOREAL_FOREST
                        else:
                            biome = BiomeType.TUNDRA
                    elif temp < 20:
                        # Temperate
                        if precip > 800:
                            biome = BiomeType.TEMPERATE_FOREST
                        elif precip > 300:
                            biome = BiomeType.GRASSLAND
                        else:
                            biome = BiomeType.DESERT
                    else:
                        # Hot
                        if precip > 1500:
                            biome = BiomeType.TROPICAL_RAINFOREST
                        elif precip > 500:
                            biome = BiomeType.SAVANNA
                        else:
                            biome = BiomeType.DESERT
                    
                    chunk.biome_type[local_x, local_y] = biome
                    biome_counts[biome] += 1
    
    # Print statistics
    total_cells = sum(biome_counts.values())
    print(f"  - Biome distribution:")
    for biome, count in biome_counts.items():
        percentage = (count / total_cells * 100) if total_cells > 0 else 0
        print(f"    {biome.name:20s}: {percentage:5.1f}%")
```

### Step 2: Add to WorldChunk Data Structure

Update `models/world.py` to include the new data:

```python
class WorldChunk:
    def __init__(self, chunk_x: int, chunk_y: int, world_size: int):
        # ... existing initialization ...
        
        # Pass 15: Biomes
        self.biome_type: Optional[np.ndarray] = None  # uint8[256, 256] - BiomeType enum
```

### Step 3: Register with Pipeline

Update `generation/pipeline.py`:

```python
def create_pipeline(params: WorldGenerationParams) -> GenerationPipeline:
    """Factory function to create a fully configured generation pipeline."""
    pipeline = GenerationPipeline(params)
    
    # ... register existing passes ...
    
    # Register new pass
    from generation import pass_15_biomes
    pipeline.register_pass("pass_15_biomes", pass_15_biomes)
    
    return pipeline
```

### Step 4: Add to Configuration

Update `config.py`:

```python
GENERATION_PASSES = [
    "pass_01_planetary",
    # ... existing passes ...
    "pass_14_polish",
    "pass_15_biomes",  # Add new pass
]

PASS_WEIGHTS = {
    # ... existing weights ...
    "pass_15_biomes": 5,  # Time weight for progress tracking
}
```

### Step 5: Update Query Interface (Optional)

Update `WorldState.query_location()` in `models/world.py`:

```python
def query_location(self, x: int, y: int) -> Optional[Dict[str, Any]]:
    """Query all data at a specific world location."""
    # ... existing code ...
    
    if chunk.biome_type is not None:
        from generation.pass_15_biomes import BiomeType
        result["biome"] = BiomeType(chunk.biome_type[local_x, local_y]).name
    
    return result
```

## Adding New Parameters

### Step 1: Update WorldGenerationParams

```python
# config.py
class WorldGenerationParams(BaseModel):
    # ... existing parameters ...
    
    # Biome parameters
    enable_biomes: bool = Field(True, description="Generate biome classifications")
    biome_blend_strength: float = Field(0.5, ge=0.0, le=1.0, description="How much biomes blend")
```

### Step 2: Use in Pass

```python
def execute(world_state: WorldState, params: WorldGenerationParams):
    if not params.enable_biomes:
        print(f"  - Biomes disabled, skipping...")
        return
    
    blend = params.biome_blend_strength
    # Use blend parameter in biome calculation
```

## Adding Custom Data Structures

### Example: Add Vegetation Density

```python
# models/world.py
class WorldChunk:
    def __init__(self, chunk_x: int, chunk_y: int, world_size: int):
        # ... existing ...
        
        # Custom vegetation data
        self.vegetation_density: Optional[np.ndarray] = None  # float32[256, 256] - 0 to 1
        self.tree_coverage: Optional[np.ndarray] = None  # float32[256, 256] - 0 to 1
```

### Example: Add Custom Feature Type

```python
# config.py
class FeatureType(str, Enum):
    CAVE_SYSTEM = "cave_system"
    # ... existing ...
    ANCIENT_RUINS = "ancient_ruins"  # Add new type
    CRYSTAL_FORMATION = "crystal_formation"

# models/world.py - Already supports this via GeologicalFeature
```

## Modifying Existing Passes

### Example: Enhance Erosion with Custom Algorithm

```python
# generation/pass_08_erosion.py

def execute(world_state: WorldState, params: WorldGenerationParams):
    """Enhanced erosion with thermal and hydraulic components."""
    print(f"  - Simulating enhanced erosion...")
    
    for iteration in range(params.erosion_iterations):
        for chunk in world_state.chunks.values():
            if chunk.elevation is None:
                continue
            
            # Original erosion
            slope = calculate_slope(chunk.elevation)
            erosion_rate = slope * chunk.precipitation_mm / 1000.0
            
            # Add thermal erosion (gravity-driven)
            thermal_erosion = calculate_thermal_erosion(chunk.elevation)
            
            # Add hydraulic erosion (water-driven)
            hydraulic_erosion = calculate_hydraulic_erosion(
                chunk.elevation,
                chunk.river_flow
            )
            
            # Combine erosion types
            total_erosion = (erosion_rate * 0.5 + 
                           thermal_erosion * 0.3 + 
                           hydraulic_erosion * 0.2)
            
            chunk.elevation -= total_erosion * params.erosion_strength
```

## Creating Dependent Passes

Some passes depend on others. Handle this gracefully:

```python
def execute(world_state: WorldState, params: WorldGenerationParams):
    """Pass that depends on rivers and climate."""
    
    # Check dependencies
    has_rivers = any(
        chunk.river_presence is not None 
        for chunk in world_state.chunks.values()
    )
    
    has_climate = any(
        chunk.temperature_c is not None 
        for chunk in world_state.chunks.values()
    )
    
    if not has_rivers or not has_climate:
        print(f"  - Missing dependencies, skipping...")
        return
    
    # Proceed with generation
    # ...
```

## Adding Utility Functions

Create new utility modules:

```python
# utils/biome_utils.py
"""Biome classification utilities"""

import numpy as np

def classify_biome(temperature: float, precipitation: float, elevation: float) -> int:
    """Classify biome from climate parameters"""
    # Classification logic
    pass

def blend_biomes(biome_map: np.ndarray, blend_strength: float) -> np.ndarray:
    """Smooth biome transitions"""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(biome_map.astype(float), sigma=blend_strength)
```

## Performance Optimization

### Vectorize Operations

```python
# Slow: Loop over each cell
for x in range(256):
    for y in range(256):
        result[x, y] = calculate_value(data[x, y])

# Fast: Vectorized
result = np.vectorize(calculate_value)(data)

# Faster: NumPy operations
result = data * 2.0 + 1.0  # Native NumPy
```

### Use Numba JIT

```python
from numba import jit

@jit(nopython=True)
def fast_calculation(data):
    """JIT-compiled function for speed"""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i, j] = data[i, j] * 2.0
    return result
```

## Testing New Passes

### Unit Test Template

```python
# test_pass_15_biomes.py
import numpy as np
from config import WorldGenerationParams, WorldSize
from models.world import WorldState, WorldMetadata, WorldChunk
from generation import pass_15_biomes

def test_biome_generation():
    """Test biome pass"""
    # Setup
    params = WorldGenerationParams(seed=42, size=WorldSize.SMALL)
    metadata = WorldMetadata(seed=42, size=512, generation_params=params.dict())
    world_state = WorldState(metadata, params)
    
    # Create test chunk with climate data
    chunk = world_state.get_or_create_chunk(0, 0)
    chunk.elevation = np.random.randn(256, 256) * 1000
    chunk.temperature_c = np.random.randn(256, 256) * 10 + 15
    chunk.precipitation_mm = np.random.randint(0, 2000, (256, 256))
    
    # Execute pass
    pass_15_biomes.execute(world_state, params)
    
    # Verify
    assert chunk.biome_type is not None
    assert chunk.biome_type.shape == (256, 256)
    assert chunk.biome_type.min() >= 0
    assert chunk.biome_type.max() < 8  # Number of biome types
```

## Best Practices

### 1. Keep Passes Independent
```python
# Good: Self-contained
def execute(world_state, params):
    for chunk in world_state.chunks.values():
        process_chunk(chunk)

# Bad: Depends on global state
global_data = {}
def execute(world_state, params):
    global global_data
    # ...
```

### 2. Handle Missing Data
```python
def execute(world_state, params):
    for chunk in world_state.chunks.values():
        if chunk.elevation is None:
            continue  # Skip if not generated yet
        
        # Process chunk
```

### 3. Provide Progress Feedback
```python
def execute(world_state, params):
    print(f"  - Processing biomes...")
    
    total_chunks = len(world_state.chunks)
    for i, chunk in enumerate(world_state.chunks.values()):
        # Process chunk
        
        if i % 10 == 0:
            progress = (i / total_chunks) * 100
            print(f"    Progress: {progress:.1f}%")
```

### 4. Use Type Hints
```python
from typing import Optional
import numpy as np

def calculate_vegetation(
    temperature: np.ndarray,
    precipitation: np.ndarray,
    soil_type: np.ndarray
) -> np.ndarray:
    """Calculate vegetation density from climate and soil."""
    # Implementation
    pass
```

### 5. Document Well
```python
def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate biome classifications.
    
    This pass classifies each cell into a biome based on:
    - Temperature (from Pass 7)
    - Precipitation (from Pass 7)
    - Elevation (from Pass 3)
    
    Uses a simplified Köppen climate classification system.
    
    Args:
        world_state: Current world state
        params: Generation parameters
    
    Side Effects:
        - Sets chunk.biome_type for all chunks
        - Prints biome distribution statistics
    """
    # Implementation
```

## Example: Complete Custom Pass

Here's a complete example of a vegetation pass:

```python
# generation/pass_16_vegetation.py
"""
Pass 16: Vegetation Generation
Generates vegetation density and tree coverage based on biomes and climate.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from config import WorldGenerationParams, CHUNK_SIZE
from models.world import WorldState


def execute(world_state: WorldState, params: WorldGenerationParams):
    """Generate vegetation density and tree coverage."""
    print(f"  - Generating vegetation...")
    
    size = world_state.size
    num_chunks = size // CHUNK_SIZE
    
    for chunk_y in range(num_chunks):
        for chunk_x in range(num_chunks):
            chunk = world_state.get_chunk(chunk_x, chunk_y)
            if chunk is None or chunk.elevation is None:
                continue
            
            # Initialize arrays
            chunk.vegetation_density = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            chunk.tree_coverage = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
            
            for local_y in range(CHUNK_SIZE):
                for local_x in range(CHUNK_SIZE):
                    elevation = chunk.elevation[local_x, local_y]
                    temp = chunk.temperature_c[local_x, local_y]
                    precip = chunk.precipitation_mm[local_x, local_y]
                    
                    if elevation < 0:
                        # Underwater - no vegetation
                        continue
                    
                    # Vegetation density based on temperature and precipitation
                    # Optimal: warm and wet
                    temp_factor = 1.0 - abs(temp - 20) / 40.0
                    temp_factor = np.clip(temp_factor, 0, 1)
                    
                    precip_factor = precip / 2000.0
                    precip_factor = np.clip(precip_factor, 0, 1)
                    
                    vegetation = temp_factor * precip_factor
                    chunk.vegetation_density[local_x, local_y] = vegetation
                    
                    # Tree coverage (needs more precipitation)
                    if precip > 500 and temp > 0:
                        tree_factor = (precip - 500) / 1500.0
                        tree_factor = np.clip(tree_factor, 0, 1)
                        chunk.tree_coverage[local_x, local_y] = tree_factor * vegetation
            
            # Smooth vegetation for realistic transitions
            chunk.vegetation_density = gaussian_filter(chunk.vegetation_density, sigma=2.0)
            chunk.tree_coverage = gaussian_filter(chunk.tree_coverage, sigma=3.0)
    
    print(f"  - Vegetation generated")
```

## Conclusion

The World Builder system is designed for easy extension. Key principles:

1. **Modularity**: Each pass is independent
2. **Configuration**: Expose parameters through WorldGenerationParams
3. **Data Layers**: Add new arrays to WorldChunk as needed
4. **Documentation**: Comment your code well
5. **Testing**: Verify your passes work correctly

Happy extending! 🌱
