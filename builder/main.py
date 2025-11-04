#!/usr/bin/env python3
"""
World Generation Engine - Demo Script

Demonstrates the complete modular world generation pipeline.
"""
import time
import numpy as np
from pathlib import Path

from config import WorldGenerationParams, WorldSize
from generation.pipeline import GenerationPipeline
from generation.base_pass import PassConfig

# Import all passes
from generation.pass_01_planetary import PlanetaryFoundationPass
from generation.pass_02_tectonics import TectonicPlatesPass
from generation.pass_03_topography import TopographyPass
from generation.pass_04_geology import GeologyPass
from generation.pass_07_climate import ClimatePass
from generation.stub_passes import (
    AtmospherePass,
    OceanCurrentsPass,
    ErosionPass,
    GroundwaterPass,
    RiversPass,
    SoilPass,
    MicroclimatePass,
    FeaturesPass,
    PolishPass,
)


def progress_callback(progress: float, current_pass: str):
    """Progress callback for monitoring generation"""
    print(f"  Progress: {progress:.1f}% - {current_pass}")


def visualize_elevation(world_data, output_path: str = "elevation_map.png"):
    """Create a simple visualization of the elevation map"""
    try:
        from PIL import Image
        
        elevation = world_data.elevation.copy()
        
        # Normalize for visualization
        land = elevation > 0
        ocean = elevation <= 0
        
        # Create RGB image
        img_array = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
        
        # Ocean (blue gradient)
        ocean_depth = np.clip(-elevation / 500, 0, 1)
        img_array[ocean, 2] = (100 + ocean_depth[ocean] * 155).astype(np.uint8)
        
        # Land (green to brown to white gradient)
        land_height = np.clip(elevation / 3000, 0, 1)
        img_array[land, 1] = (50 + (1 - land_height[land]) * 150).astype(np.uint8)
        img_array[land, 0] = (100 + land_height[land] * 100).astype(np.uint8)
        
        # Snow caps (white) on high peaks
        high_peaks = elevation > 2500
        img_array[high_peaks] = [250, 250, 250]
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(output_path)
        print(f"\n✓ Elevation map saved to: {output_path}")
        
    except ImportError:
        print("\n⚠ Pillow not installed - skipping visualization")


def print_world_statistics(world_data, params):
    """Print interesting statistics about the generated world"""
    print("\n" + "="*60)
    print(" WORLD GENERATION COMPLETE")
    print("="*60)
    
    print(f"\nWorld Seed: {params.seed}")
    size_val = params.size if isinstance(params.size, str) else params.size.value
    print(f"World Size: {size_val} x {size_val} cells")
    
    if world_data.elevation is not None:
        ocean_cells = np.sum(world_data.elevation < 0)
        land_cells = np.sum(world_data.elevation >= 0)
        total_cells = world_data.elevation.size
        
        print(f"\nTerrain:")
        print(f"  Ocean: {(ocean_cells/total_cells)*100:.1f}%")
        print(f"  Land:  {(land_cells/total_cells)*100:.1f}%")
        
        if land_cells > 0:
            land_elev = world_data.elevation[world_data.elevation >= 0]
            print(f"  Avg elevation: {land_elev.mean():.1f}m")
            print(f"  Max elevation: {land_elev.max():.1f}m")
    
    if world_data.temperature_c is not None:
        land_temp = world_data.temperature_c[world_data.elevation > 0]
        print(f"\nClimate:")
        print(f"  Avg temperature: {land_temp.mean():.1f}°C")
        print(f"  Temperature range: {land_temp.min():.1f}°C to {land_temp.max():.1f}°C")
    
    if world_data.precipitation_mm is not None:
        land_precip = world_data.precipitation_mm[world_data.elevation > 0]
        print(f"  Avg precipitation: {land_precip.mean():.0f}mm/year")
    
    if world_data.river_presence is not None:
        river_cells = np.sum(world_data.river_presence)
        print(f"\nHydrology:")
        print(f"  River cells: {river_cells:,}")
    
    if world_data.cave_presence is not None:
        cave_cells = np.sum(world_data.cave_presence)
        print(f"\nGeological Features:")
        print(f"  Cave systems: {cave_cells:,}")
    
    print("\n" + "="*60)


def generate_world(
    seed: int = 42,
    size: WorldSize = WorldSize.SMALL,
    num_plates: int = 12,
    ocean_percentage: float = 0.7,
    enabled_passes: list[str] = None
):
    """
    Generate a complete world
    
    Args:
        seed: Random seed for generation
        size: World size (SMALL, MEDIUM, LARGE, HUGE)
        num_plates: Number of tectonic plates
        ocean_percentage: Percentage of world covered by ocean (0-1)
        enabled_passes: List of pass names to enable (None = all passes)
    """
    print("="*60)
    print(" WORLD GENERATION ENGINE")
    print("="*60)
    
    # Create generation parameters
    params = WorldGenerationParams(
        seed=seed,
        size=size,
        planet_radius_km=6371.0,
        gravity=9.8,
        axial_tilt=23.5,
        rotation_hours=24.0,
        num_plates=num_plates,
        plate_speed_mm_year=50.0,
        base_temperature_c=15.0,
        atmospheric_pressure_atm=1.0,
        ocean_percentage=ocean_percentage,
        custom_noise_octaves=6,
        custom_noise_persistence=0.5,
        enabled_passes=enabled_passes
    )
    
    print(f"\nParameters:")
    print(f"  Seed: {seed}")
    print(f"  Size: {size.value} x {size.value} ({int(size.value)**2:,} cells)")
    print(f"  Tectonic plates: {num_plates}")
    print(f"  Target ocean coverage: {ocean_percentage*100:.0f}%")
    
    # Create pipeline and register all passes
    pipeline = GenerationPipeline(params)
    
    # Register passes in order (dependencies will be automatically resolved)
    passes = [
        PassConfig(PlanetaryFoundationPass, enabled=True),
        PassConfig(TectonicPlatesPass, enabled=True),
        PassConfig(TopographyPass, enabled=True),
        PassConfig(GeologyPass, enabled=True),
        PassConfig(AtmospherePass, enabled=True),
        PassConfig(OceanCurrentsPass, enabled=True),
        PassConfig(ClimatePass, enabled=True),
        PassConfig(ErosionPass, enabled=True),
        PassConfig(GroundwaterPass, enabled=True),
        PassConfig(RiversPass, enabled=True),
        PassConfig(SoilPass, enabled=True),
        PassConfig(MicroclimatePass, enabled=True),
        PassConfig(FeaturesPass, enabled=True),
        PassConfig(PolishPass, enabled=True),
    ]
    
    pipeline.register_passes(passes)
    
    # Generate world
    print(f"\nStarting generation with {len(passes)} passes...\n")
    start_time = time.time()
    
    try:
        world_data = pipeline.generate()
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Generation completed in {elapsed_time:.2f} seconds")
        
        # Print statistics
        print_world_statistics(world_data, params)
        
        # Create visualization
        visualize_elevation(world_data, f"world_seed{seed}_{size.value}.png")
        
        return world_data
        
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        raise


def demo_chunk_extraction(world_data):
    """Demonstrate chunk-based data access"""
    print("\n" + "="*60)
    print(" CHUNK EXTRACTION DEMO")
    print("="*60)
    
    # Extract a chunk
    chunk_x, chunk_y = 0, 0
    chunk_size = 256
    
    chunk_data = world_data.get_chunk(chunk_x, chunk_y, chunk_size)
    
    print(f"\nExtracted chunk ({chunk_x}, {chunk_y}):")
    print(f"  Chunk size: {chunk_size}x{chunk_size}")
    print(f"  Arrays in chunk: {list(chunk_data.keys())}")
    
    if 'elevation' in chunk_data:
        chunk_elev = chunk_data['elevation']
        print(f"\nChunk elevation stats:")
        print(f"  Min: {chunk_elev.min():.1f}m")
        print(f"  Max: {chunk_elev.max():.1f}m")
        print(f"  Mean: {chunk_elev.mean():.1f}m")


def demo_configurable_generation():
    """Demonstrate configurable pass selection"""
    print("\n" + "="*60)
    print(" CONFIGURABLE GENERATION DEMO")
    print("="*60)
    
    print("\nGenerating world with only basic passes (topography only)...\n")
    
    # Generate with limited passes
    world_data = generate_world(
        seed=12345,
        size=WorldSize.SMALL,
        enabled_passes=[
            "PlanetaryFoundationPass",
            "TectonicPlatesPass", 
            "TopographyPass"
        ]
    )
    
    return world_data


if __name__ == "__main__":
    print("\n" + "🌍 "*20)
    print("\nWORLD GENERATION ENGINE - DEMO")
    print("\n" + "🌍 "*20 + "\n")
    
    # Demo 1: Full world generation
    print("\n📍 DEMO 1: Full World Generation")
    world_data = generate_world(
        seed=424242,
        size=WorldSize.SMALL,
        num_plates=12,
        ocean_percentage=0.7
    )
    
    # Demo 2: Chunk extraction
    print("\n📍 DEMO 2: Chunk Extraction")
    demo_chunk_extraction(world_data)
    
    # Demo 3: Different world seeds
    print("\n📍 DEMO 3: Different Seeds")
    print("\nGenerating world with different seed...")
    world_data_2 = generate_world(
        seed=999,
        size=WorldSize.SMALL,
        num_plates=8,
        ocean_percentage=0.6
    )
    
    print("\n✓ All demos completed successfully!")
    print("\n" + "🌍 "*20 + "\n")
