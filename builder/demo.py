#!/usr/bin/env python3
"""
World Builder - Demo Script
Demonstrates world generation capabilities and outputs results.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import WorldGenerationParams, WorldSize
from generation.pipeline import create_pipeline
import numpy as np


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_small_world():
    """Generate a small demo world"""
    print_section("WORLD BUILDER - PROCEDURAL GENERATION DEMO")
    
    # Configure world generation parameters
    params = WorldGenerationParams(
        seed=42,
        size=WorldSize.SMALL,  # 512x512 world
        
        # Planetary parameters
        planet_radius_km=6371.0,
        gravity=9.8,
        axial_tilt=23.5,
        rotation_hours=24.0,
        
        # Tectonic parameters
        num_plates=8,
        plate_speed_mm_year=50.0,
        
        # Atmospheric parameters
        base_temperature_c=15.0,
        atmospheric_pressure_atm=1.0,
        
        # Hydrological parameters
        ocean_percentage=0.7,
        
        # Noise parameters
        custom_noise_octaves=6,
        custom_noise_persistence=0.5,
        custom_noise_lacunarity=2.0,
        
        # Erosion parameters
        erosion_iterations=2,
        erosion_strength=1.0,
        
        # Feature generation
        enable_caves=True,
        enable_hot_springs=True,
        enable_waterfalls=True,
    )
    
    print("World Configuration:")
    print(f"  Seed: {params.seed}")
    print(f"  Size: {params.size}x{params.size}")
    print(f"  Tectonic Plates: {params.num_plates}")
    print(f"  Base Temperature: {params.base_temperature_c}°C")
    print(f"  Ocean Coverage: {params.ocean_percentage*100}%")
    
    # Create and execute generation pipeline
    print_section("EXECUTING GENERATION PIPELINE")
    
    pipeline = create_pipeline(params)
    
    start_time = time.time()
    world_state = pipeline.generate()
    generation_time = time.time() - start_time
    
    print_section("GENERATION COMPLETE")
    print(f"Total generation time: {generation_time:.2f} seconds")
    print(f"World ID: {world_state.metadata.world_id}")
    print(f"Status: {world_state.metadata.status}")
    print(f"Number of chunks: {len(world_state.chunks)}")
    
    # Analyze generated world
    print_section("WORLD ANALYSIS")
    
    # Collect statistics
    all_elevations = []
    all_temperatures = []
    all_precipitation = []
    
    for chunk in world_state.chunks.values():
        if chunk.elevation is not None:
            all_elevations.append(chunk.elevation.flatten())
        if chunk.temperature_c is not None:
            all_temperatures.append(chunk.temperature_c.flatten())
        if chunk.precipitation_mm is not None:
            all_precipitation.append(chunk.precipitation_mm.flatten())
    
    if all_elevations:
        elevations = np.concatenate(all_elevations)
        print("Elevation Statistics:")
        print(f"  Minimum: {elevations.min():.1f}m")
        print(f"  Maximum: {elevations.max():.1f}m")
        print(f"  Mean: {elevations.mean():.1f}m")
        print(f"  Median: {np.median(elevations):.1f}m")
        
        land_cells = (elevations > 0).sum()
        ocean_cells = (elevations <= 0).sum()
        print(f"\nLand Coverage:")
        print(f"  Land: {land_cells / len(elevations) * 100:.1f}%")
        print(f"  Ocean: {ocean_cells / len(elevations) * 100:.1f}%")
    
    if all_temperatures:
        temperatures = np.concatenate(all_temperatures)
        print(f"\nTemperature Statistics:")
        print(f"  Minimum: {temperatures.min():.1f}°C")
        print(f"  Maximum: {temperatures.max():.1f}°C")
        print(f"  Mean: {temperatures.mean():.1f}°C")
    
    if all_precipitation:
        precipitation = np.concatenate(all_precipitation)
        print(f"\nPrecipitation Statistics:")
        print(f"  Minimum: {precipitation.min():.0f}mm/year")
        print(f"  Maximum: {precipitation.max():.0f}mm/year")
        print(f"  Mean: {precipitation.mean():.0f}mm/year")
    
    # Query specific locations
    print_section("SAMPLE LOCATION QUERIES")
    
    sample_locations = [
        (128, 128),  # Center
        (256, 256),  # Middle
        (64, 384),   # Various points
    ]
    
    for x, y in sample_locations:
        data = world_state.query_location(x, y)
        if data:
            print(f"\nLocation ({x}, {y}):")
            print(f"  Chunk: ({data.get('chunk_x')}, {data.get('chunk_y')})")
            if 'elevation' in data:
                print(f"  Elevation: {data['elevation']:.1f}m")
            if 'temperature_c' in data:
                print(f"  Temperature: {data['temperature_c']:.1f}°C")
            if 'precipitation_mm' in data:
                print(f"  Precipitation: {data['precipitation_mm']}mm/year")
            if 'bedrock_type' in data:
                print(f"  Bedrock: {data['bedrock_type']}")
            if 'soil_type' in data:
                print(f"  Soil: {data['soil_type']}")
    
    # Count geological features
    print_section("GEOLOGICAL FEATURES")
    
    feature_counts = {}
    for chunk in world_state.chunks.values():
        for feature in chunk.geological_features:
            feature_type = feature.type.value
            feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
    
    if feature_counts:
        print("Generated Features:")
        for feature_type, count in sorted(feature_counts.items()):
            print(f"  {feature_type}: {count}")
    else:
        print("No discrete features generated in this world.")
    
    print_section("DEMO COMPLETE")
    print("\nWorld generation successful!")
    print(f"World size: {params.size}x{params.size} ({len(world_state.chunks)} chunks)")
    print(f"Generation time: {generation_time:.2f}s")
    
    return world_state


def demo_chunk_query(world_state):
    """Demonstrate chunk-based queries"""
    print_section("CHUNK-BASED ACCESS")
    
    # Get a specific chunk
    chunk = world_state.get_chunk(0, 0)
    
    if chunk:
        print(f"Chunk (0, 0) Statistics:")
        if chunk.elevation is not None:
            print(f"  Elevation range: {chunk.elevation.min():.1f}m to {chunk.elevation.max():.1f}m")
        if chunk.temperature_c is not None:
            print(f"  Temperature range: {chunk.temperature_c.min():.1f}°C to {chunk.temperature_c.max():.1f}°C")
        if chunk.river_presence is not None:
            river_cells = chunk.river_presence.sum()
            print(f"  River cells: {river_cells} ({river_cells / 256**2 * 100:.2f}%)")


def demo_custom_parameters():
    """Demonstrate customizable generation parameters"""
    print_section("CUSTOM PARAMETER DEMO")
    
    # Create an extreme world
    params = WorldGenerationParams(
        seed=999,
        size=WorldSize.SMALL,
        
        # Extreme parameters
        gravity=15.0,  # High gravity (50% more than Earth)
        axial_tilt=45.0,  # Extreme seasons
        num_plates=20,  # Many tectonic plates
        ocean_percentage=0.9,  # Mostly water
        erosion_strength=3.0,  # Strong erosion
    )
    
    print("Creating world with extreme parameters:")
    print(f"  Gravity: {params.gravity} m/s²")
    print(f"  Axial tilt: {params.axial_tilt}°")
    print(f"  Tectonic plates: {params.num_plates}")
    print(f"  Ocean coverage: {params.ocean_percentage*100}%")
    print(f"  Erosion strength: {params.erosion_strength}x")
    
    print("\n(Generation skipped for demo brevity)")


def main():
    """Main demo function"""
    try:
        # Run main demo
        world_state = demo_small_world()
        
        # Demonstrate chunk queries
        demo_chunk_query(world_state)
        
        # Show custom parameters
        demo_custom_parameters()
        
        print_section("ALL DEMOS COMPLETE")
        print("\nThe World Builder generation engine is ready for use!")
        print("\nKey Features:")
        print("  ✓ Modular generation pipeline with 14 passes")
        print("  ✓ Fully configurable parameters")
        print("  ✓ Chunk-based architecture for scalability")
        print("  ✓ Deterministic generation (same seed = same world)")
        print("  ✓ Scientific accuracy in climate, geology, and hydrology")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
