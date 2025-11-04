# World Generation Engine - Quick Start Guide

## Installation

```bash
cd world-gen-service
pip install -r requirements.txt
```

## Generate Your First World (30 seconds)

```bash
python main.py
```

This runs a complete demo that:
1. Generates a 512×512 world
2. Exports visualization
3. Demonstrates chunk extraction
4. Shows different seeds

## Custom World Generation

```python
from models import WorldGenerationParams, WorldSize
from generation.pipeline import GenerationPipeline
from generation.base_pass import PassConfig

# Import all passes
from generation.pass_01_planetary import PlanetaryFoundationPass
from generation.pass_02_tectonics import TectonicPlatesPass
from generation.pass_03_topography import TopographyPass
from generation.pass_04_geology import GeologyPass
from generation.pass_07_climate import ClimatePass
from generation.stub_passes import (
    AtmospherePass, ErosionPass, GroundwaterPass,
    RiversPass, SoilPass, MicroclimatePass,
    FeaturesPass, PolishPass
)

# Configure your world
params = WorldGenerationParams(
    seed=12345,                    # Your random seed
    size=WorldSize.SMALL,          # SMALL, MEDIUM, LARGE, HUGE
    num_plates=15,                 # More plates = more varied terrain
    ocean_percentage=0.65,         # 0.0-1.0
    base_temperature_c=18.0,       # Base planet temperature
    gravity=9.8,                   # m/s²
)

# Create pipeline
pipeline = GenerationPipeline(params)

# Register all passes
pipeline.register_passes([
    PassConfig(PlanetaryFoundationPass),
    PassConfig(TectonicPlatesPass),
    PassConfig(TopographyPass),
    PassConfig(GeologyPass),
    PassConfig(AtmospherePass),
    PassConfig(ClimatePass),
    PassConfig(ErosionPass),
    PassConfig(GroundwaterPass),
    PassConfig(RiversPass),
    PassConfig(SoilPass),
    PassConfig(MicroclimatePass),
    PassConfig(FeaturesPass),
    PassConfig(PolishPass),
])

# Generate!
world = pipeline.generate()
```

## Accessing Generated Data

```python
# Terrain data
elevation = world.elevation              # Height map
plates = world.plate_id                  # Tectonic plates
stress = world.tectonic_stress           # Mountain-building zones

# Climate data
temperature = world.temperature_c        # Temperature map
precipitation = world.precipitation_mm   # Rainfall map
wind_dir = world.wind_direction         # Wind patterns
wind_speed = world.wind_speed           # Wind strength

# Geology
bedrock = world.bedrock_type            # Rock types
minerals = world.mineral_richness        # Ore deposits
soil_depth = world.soil_depth_cm        # Soil layer thickness

# Hydrology
water_table = world.water_table_depth    # Groundwater depth
rivers = world.river_presence           # River map
flow = world.river_flow                 # River flow rates

# Soil
soil_type = world.soil_type             # Soil classification
soil_ph = world.soil_ph                 # Soil pH
drainage = world.soil_drainage          # Drainage class

# Features
caves = world.cave_presence             # Cave locations
microclimates = world.microclimate_modifier  # Local variations
```

## Get a Specific Region (Chunk)

```python
# Get a 256×256 chunk at position (0, 0)
chunk = world.get_chunk(chunk_x=0, chunk_y=0, chunk_size=256)

# Access chunk data
chunk_elevation = chunk['elevation']
chunk_temp = chunk['temperature_c']
```

## Query a Specific Location

```python
x, y = 256, 128  # Coordinates

# Get all data at this location
elevation_here = world.elevation[x, y]
temp_here = world.temperature_c[x, y]
rock_type_here = world.bedrock_type[x, y]
is_river = world.river_presence[x, y]
has_cave = world.cave_presence[x, y]

print(f"Location ({x}, {y}):")
print(f"  Elevation: {elevation_here:.1f}m")
print(f"  Temperature: {temp_here:.1f}°C")
print(f"  River: {'Yes' if is_river else 'No'}")
print(f"  Cave: {'Yes' if has_cave else 'No'}")
```

## Customize Parameters

```python
# Earth-like planet
earth_params = WorldGenerationParams(
    seed=42,
    size=WorldSize.MEDIUM,
    planet_radius_km=6371,
    gravity=9.8,
    axial_tilt=23.5,
    rotation_hours=24.0,
    num_plates=12,
    ocean_percentage=0.7,
    base_temperature_c=15.0,
)

# Desert planet
desert_params = WorldGenerationParams(
    seed=42,
    size=WorldSize.MEDIUM,
    planet_radius_km=3000,
    gravity=6.0,
    axial_tilt=15.0,
    rotation_hours=36.0,
    num_plates=8,
    ocean_percentage=0.2,        # Less water
    base_temperature_c=28.0,     # Hotter
)

# Water world
water_params = WorldGenerationParams(
    seed=42,
    size=WorldSize.MEDIUM,
    planet_radius_km=7000,
    gravity=11.0,
    axial_tilt=30.0,
    rotation_hours=20.0,
    num_plates=15,
    ocean_percentage=0.95,       # Mostly ocean
    base_temperature_c=12.0,
)
```

## Enable/Disable Specific Passes

```python
# Only generate terrain (no climate, rivers, etc.)
params = WorldGenerationParams(
    seed=42,
    size=WorldSize.SMALL,
    enabled_passes=[
        "planetary_foundation",
        "tectonic_plates",
        "topography"
    ]
)

# Generate with only these passes
pipeline = GenerationPipeline(params)
pipeline.register_passes([...])  # Register all, but only enabled ones run
world = pipeline.generate()
```

## Progress Tracking

```python
def my_progress_callback(percent, current_pass):
    print(f"[{percent:5.1f}%] {current_pass}")

world = pipeline.generate(progress_callback=my_progress_callback)
```

## World Sizes

| Size   | Dimensions  | Cells      | Gen Time | Memory  |
|--------|-------------|------------|----------|---------|
| SMALL  | 512×512     | ~262k      | ~12s     | ~50MB   |
| MEDIUM | 1024×1024   | ~1M        | ~45s     | ~200MB  |
| LARGE  | 2048×2048   | ~4M        | ~3min    | ~800MB  |
| HUGE   | 4096×4096   | ~16M       | ~10min   | ~3GB    |

## Tips

1. **Start small** - Test with SMALL size first
2. **Use good seeds** - Try different seeds to find interesting terrain
3. **Adjust ocean percentage** - 0.7 is Earth-like, adjust for variety
4. **More plates = more varied terrain** - Try 8-20 plates
5. **Monitor memory** - Large worlds use significant RAM

## Common Patterns

### Find High Peaks
```python
# Find mountains over 2000m
mountains = world.elevation > 2000
mountain_coords = np.argwhere(mountains)
```

### Find Coastlines
```python
# Find land next to ocean
is_land = world.elevation > 0
is_ocean = world.elevation <= 0

# Use convolution to find borders
from scipy.ndimage import convolve
kernel = np.ones((3, 3))
ocean_neighbors = convolve(is_ocean.astype(int), kernel, mode='constant')
coastline = is_land & (ocean_neighbors > 0)
```

### Find Mineral-Rich Areas
```python
# Find gold deposits
gold_richness = world.mineral_richness[:, :, 2]  # Gold is mineral #2
rich_gold = gold_richness > 0.5
```

### Find Temperate Regions
```python
# Find areas with comfortable temperature and rainfall
temperate = (
    (world.temperature_c > 10) & 
    (world.temperature_c < 25) &
    (world.precipitation_mm > 500) &
    (world.precipitation_mm < 1500)
)
```

## Export Data

```python
import numpy as np

# Save elevation as numpy array
np.save('elevation.npy', world.elevation)

# Load it back
elevation = np.load('elevation.npy')

# Export to CSV
np.savetxt('elevation.csv', world.elevation, delimiter=',')
```

## Visualization

```python
from PIL import Image
import numpy as np

# Create simple height map
elevation = world.elevation
normalized = (elevation - elevation.min()) / (elevation.max() - elevation.min())
img_data = (normalized * 255).astype(np.uint8)
img = Image.fromarray(img_data)
img.save('heightmap.png')
```

## Next Steps

1. Read the full README.md for detailed documentation
2. Explore the generated visualization images
3. Try different world configurations
4. Create custom generation passes
5. Build your application on top of the world data

## Help

- Check README.md for full documentation
- See IMPLEMENTATION_SUMMARY.md for technical details
- Review the code in `generation/` for pass implementations
- Study `main.py` for working examples

Happy world building! 🌍
