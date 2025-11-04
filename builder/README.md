# World Builder - Procedural World Generation Engine

A scientifically-accurate, modular procedural generation system for creating deeply immersive fantasy worlds. Built following the philosophy: *"If you wish to make an apple pie from scratch, you must first invent the universe."* - Carl Sagan

## Overview

World Builder generates complete fantasy worlds through a 14-pass pipeline that simulates planetary physics, tectonics, climate, hydrology, and geology. The system is:

- **Modular**: Each generation pass is independent and can be modified or replaced
- **Configurable**: All generation parameters can be customized
- **Deterministic**: Same seed always produces identical worlds
- **Scalable**: Chunk-based architecture for generating massive worlds
- **Scientific**: Based on real-world geological and climatological principles

## Features

### Core Generation Passes

1. **Planetary Foundation** - Establishes fundamental planetary parameters (gravity, rotation, axial tilt)
2. **Tectonic Plates** - Generates plate system using Voronoi diagrams
3. **Topography** - Creates base elevation using multi-octave noise
4. **Geology** - Determines bedrock types and mineral distributions
5. **Atmospheric Dynamics** - Calculates wind patterns and circulation
6. **Ocean Currents** - Simulates large-scale ocean circulation
7. **Climate** - Determines temperature and precipitation patterns
8. **Erosion** - Simulates weathering and sediment transport
9. **Groundwater** - Calculates aquifer depth and water tables
10. **Surface Hydrology** - Generates river networks and lakes
11. **Soil Formation** - Creates soil types and properties
12. **Microclimate** - Adds local climate variations
13. **Geological Features** - Places caves, hot springs, canyons, etc.
14. **Polish** - Final cleanup and optimization

### Configurable Parameters

```python
WorldGenerationParams(
    seed=42,                        # Deterministic generation
    size=WorldSize.MEDIUM,          # 512, 1024, 2048, or 4096
    
    # Planetary
    planet_radius_km=6371.0,
    gravity=9.8,
    axial_tilt=23.5,
    rotation_hours=24.0,
    
    # Tectonics
    num_plates=12,
    plate_speed_mm_year=50.0,
    
    # Atmosphere
    base_temperature_c=15.0,
    atmospheric_pressure_atm=1.0,
    
    # Hydrology
    ocean_percentage=0.7,
    
    # Noise (optional overrides)
    custom_noise_octaves=6,
    custom_noise_persistence=0.5,
    custom_noise_lacunarity=2.0,
    
    # Erosion
    erosion_iterations=3,
    erosion_strength=1.0,
    
    # Features
    enable_caves=True,
    enable_hot_springs=True,
    enable_waterfalls=True,
)
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from config import WorldGenerationParams, WorldSize
from generation.pipeline import create_pipeline

# Configure world parameters
params = WorldGenerationParams(
    seed=42,
    size=WorldSize.SMALL,  # 512x512
    num_plates=8,
    ocean_percentage=0.7,
)

# Create and execute pipeline
pipeline = create_pipeline(params)
world_state = pipeline.generate()

# Query specific locations
location_data = world_state.query_location(x=256, y=256)
print(f"Elevation: {location_data['elevation']}m")
print(f"Temperature: {location_data['temperature_c']}°C")
print(f"Biome: {location_data['biome']}")
```

### Run Demo

```bash
python demo.py
```

## Architecture

### Chunk-Based System

Worlds are divided into 256×256 tile chunks for efficient generation and storage:

```
World (1024×1024)
├── Chunk (0,0) [256×256]
├── Chunk (0,1) [256×256]
├── Chunk (0,2) [256×256]
├── Chunk (0,3) [256×256]
├── ...
└── Chunk (3,3) [256×256]
```

Each chunk contains all terrain layers:

```python
class WorldChunk:
    # Topography
    elevation: np.ndarray          # float32[256, 256]
    plate_id: np.ndarray           # uint8[256, 256]
    tectonic_stress: np.ndarray    # float32[256, 256]
    
    # Geology
    bedrock_type: np.ndarray       # uint8[256, 256]
    mineral_richness: Dict[Mineral, np.ndarray]
    
    # Climate
    temperature_c: np.ndarray      # float32[256, 256]
    precipitation_mm: np.ndarray   # uint16[256, 256]
    wind_direction: np.ndarray     # uint16[256, 256]
    wind_speed: np.ndarray         # float32[256, 256]
    
    # Hydrology
    water_table_depth: np.ndarray  # float32[256, 256]
    river_presence: np.ndarray     # bool[256, 256]
    river_flow: np.ndarray         # float32[256, 256]
    
    # Soil
    soil_type: np.ndarray          # uint8[256, 256]
    soil_ph: np.ndarray            # float32[256, 256]
    soil_drainage: np.ndarray      # uint8[256, 256]
    
    # Features
    geological_features: List[GeologicalFeature]
```

### Pipeline Architecture

```
┌─────────────────────────────────┐
│   Generation Pipeline           │
│                                 │
│  ┌──────────────────────────┐  │
│  │  Pass 1: Planetary       │  │
│  └───────────┬──────────────┘  │
│              ▼                  │
│  ┌──────────────────────────┐  │
│  │  Pass 2: Tectonics       │  │
│  └───────────┬──────────────┘  │
│              ▼                  │
│  ┌──────────────────────────┐  │
│  │  Pass 3: Topography      │  │
│  └───────────┬──────────────┘  │
│              ▼                  │
│            [...]                │
│              ▼                  │
│  ┌──────────────────────────┐  │
│  │  Pass 14: Polish         │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

### Extensibility

Add new passes by creating a module with an `execute()` function:

```python
# generation/pass_15_custom.py
def execute(world_state: WorldState, params: WorldGenerationParams):
    """Your custom generation logic"""
    for chunk in world_state.chunks.values():
        # Modify chunk data
        pass

# Register in pipeline
pipeline.register_pass("pass_15_custom", pass_15_custom)
```

## Data Structures

### World Generation Parameters

All parameters that control world generation are configurable:

- **Planetary**: Gravity, rotation, axial tilt
- **Tectonic**: Number of plates, movement speed
- **Climate**: Base temperature, atmospheric pressure
- **Noise**: Octaves, persistence, lacunarity
- **Erosion**: Iterations, strength
- **Features**: Enable/disable specific features

### World State

The `WorldState` object contains:
- **Metadata**: World ID, status, progress
- **Planetary Data**: Global constants
- **Tectonic System**: Plate information
- **Chunks**: Dictionary of generated chunks

### Querying

```python
# Query by location
data = world_state.query_location(x=512, y=768)

# Get specific chunk
chunk = world_state.get_chunk(chunk_x=2, chunk_y=3)

# Access chunk data
elevation = chunk.elevation[local_x, local_y]
temperature = chunk.temperature_c[local_x, local_y]
```

## Scientific Principles

### Climate System

- **Solar Input**: Based on latitude and axial tilt
- **Wind Patterns**: Hadley, Ferrel, and Polar cells
- **Orographic Effect**: Mountains create rain shadows
- **Temperature Lapse Rate**: -6.5°C per 1000m elevation

### Hydrology

- **Flow Direction**: D8 algorithm for realistic drainage
- **Flow Accumulation**: Precipitation-weighted water flow
- **River Formation**: Networks form naturally from topography
- **Groundwater**: Depth based on rock permeability and recharge

### Geology

- **Rock Types**: Igneous, sedimentary, metamorphic, limestone
- **Plate Boundaries**: Convergent, divergent, transform
- **Mountain Formation**: At plate collision zones
- **Mineral Distribution**: Based on rock type and tectonics

### Erosion

- **Slope-Based**: Steeper terrain erodes faster
- **Precipitation-Driven**: More rain = more erosion
- **Sediment Transport**: Material moves from high to low areas
- **Gravity Influence**: Planet gravity affects erosion rate

## Performance

### Optimization Techniques

- **NumPy Vectorization**: Efficient array operations
- **Chunk Independence**: Parallelizable generation
- **Deterministic Seeds**: No need to store raw noise data
- **Sparse Storage**: Only generated chunks in memory

### Typical Generation Times

| World Size | Dimensions | Chunks | Time (approx) |
|-----------|-----------|---------|---------------|
| Small     | 512×512   | 4       | 5-10 seconds  |
| Medium    | 1024×1024 | 16      | 20-40 seconds |
| Large     | 2048×2048 | 64      | 2-4 minutes   |
| Huge      | 4096×4096 | 256     | 10-20 minutes |

*Times vary based on hardware and enabled features*

## Future Enhancements

The system is designed for easy extension:

1. **Biomes** - Flora classification from climate + soil
2. **Ecology** - Fauna distribution and food webs
3. **Resources** - Harvestable materials and deposits
4. **Civilizations** - Settlement placement and cultures
5. **History** - Time simulation and lore generation
6. **Dynamic State** - Current conditions and changes

## Project Structure

```
world-builder/
├── config.py              # Configuration and constants
├── demo.py                # Demonstration script
├── requirements.txt       # Dependencies
│
├── models/
│   ├── __init__.py
│   └── world.py          # Data structures
│
├── generation/
│   ├── __init__.py
│   ├── pipeline.py       # Main orchestrator
│   ├── pass_01_planetary.py
│   ├── pass_02_tectonics.py
│   ├── pass_03_topography.py
│   └── ...               # 14 total passes
│
└── utils/
    ├── __init__.py
    ├── noise.py          # Noise generation
    └── spatial.py        # Spatial calculations
```

## License

This is a technical implementation of the World Builder specification. Use according to your project requirements.

## Credits

Based on real-world geological, climatological, and hydrological principles. Inspired by:
- Plate tectonics theory
- Köppen climate classification
- River network analysis
- Soil taxonomy
- Planetary science

## Contact

For questions about implementation or extending the system, refer to the code documentation and inline comments.
