#!/usr/bin/env python3
"""
World Builder - Interactive Demo with Napari Visualization
Demonstrates world generation capabilities with interactive layer exploration.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import WorldGenerationParams, WorldSize
from generation.pipeline import create_pipeline
from utils.visualizers import UnifiedNapariVisualizer, NAPARI_AVAILABLE, view_world_interactive


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_world_with_napari():
    """Generate a demo world and visualize interactively with Napari"""
    print_section("WORLD BUILDER - INTERACTIVE GENERATION WITH NAPARI")
    
    # Configure world generation parameters
    params = WorldGenerationParams(
        seed=420,
        size=WorldSize.SMALL,  # 512x512 world
        
        # Planetary parameters
        planet_radius_km=6371.0,
        gravity=9.8,
        axial_tilt=23.5,
        rotation_hours=24.0,
        
        # Tectonic parameters
        num_plates=58,
        plate_speed_mm_year=50.0,
        
        # Atmospheric parameters
        base_temperature_c=15.0,
        atmospheric_pressure_atm=1.0,
        
        # Hydrological parameters
        ocean_percentage=0.7,
        
        # Noise parameters
        custom_noise_octaves=7,
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
    
    # Count geological features
    print_section("GEOLOGICAL FEATURES")
    
    feature_counts = {}
    for chunk in world_state.chunks.values():
        for feature in chunk.geological_features:
            feature_type = feature.type
            feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
    
    if feature_counts:
        print("Generated Features:")
        for feature_type, count in sorted(feature_counts.items()):
            print(f"  {feature_type}: {count}")
    else:
        print("No discrete features generated in this world.")
    
    print_section("LAUNCHING INTERACTIVE NAPARI VIEWER")
    
    if not NAPARI_AVAILABLE:
        print("❌ Napari is not installed!")
        print("   Install with: pip install 'napari[all]'")
        print("\nSkipping interactive visualization...")
        return world_state
    
    print("🌍 Opening interactive world viewer...")
    print("\nTips for exploring your world:")
    print("  • Start with Elevation layer (already visible)")
    print("  • Toggle Climate → Temperature to see temperature patterns")
    print("  • Enable Rivers to see hydrology")
    print("  • Check out Biomes for ecological zones")
    print("  • Use opacity sliders to blend layers")
    print("\nClosing this window will end the demo.\n")
    
    # Launch interactive viewer
    view_world_interactive(world_state)
    
    print_section("DEMO COMPLETE")
    print("\nWorld generation and visualization successful!")
    print(f"World size: {params.size}x{params.size} ({len(world_state.chunks)} chunks)")
    print(f"Generation time: {generation_time:.2f}s")
    
    return world_state


def demo_specific_passes():
    """Demo showing specific passes only"""
    print_section("FOCUSED VIEW - CLIMATE AND BIOMES ONLY")
    
    # Quick world gen
    params = WorldGenerationParams(seed=123, size=WorldSize.TINY)
    pipeline = create_pipeline(params)
    world_state = pipeline.generate()
    
    if NAPARI_AVAILABLE:
        print("Opening viewer with Climate (Pass 7) and Biomes (Pass 12) only...")
        view_world_interactive(world_state, passes=[3, 7, 12])  # Include elevation for context
    else:
        print("❌ Napari not available for focused view demo")


def main():
    """Main demo function"""
    try:
        print("\n" + "🌍 "*20)
        print("WORLD BUILDER - INTERACTIVE NAPARI VISUALIZATION DEMO")
        print("🌍 "*20 + "\n")
        
        # Check napari availability
        if not NAPARI_AVAILABLE:
            print("⚠️  WARNING: Napari is not installed!")
            print("   This demo requires napari for interactive visualization.")
            print("   Install with: pip install 'napari[all]'")
            print("\nProceeding with world generation only...\n")
        
        # Demo 1: Full world with interactive visualization
        print("\n📍 DEMO 1: Complete World with Interactive Napari Viewer")
        world_state = demo_world_with_napari()
        
        # Optional: Demo 2 - focused view
        # Uncomment to try viewing specific passes only
        # print("\n📍 DEMO 2: Focused Pass View")
        # demo_specific_passes()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())