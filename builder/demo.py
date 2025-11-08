#!/usr/bin/env python3
"""
World Builder - Demo Script with Visualization
Demonstrates world generation capabilities and outputs visualizations for each layer.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import WorldGenerationParams, WorldSize
from generation.pipeline import create_pipeline
from utils.visualizers import UnifiedVisualizer


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_small_world_with_viz():
    """Generate a small demo world with full visualization"""
    print_section("WORLD BUILDER - PROCEDURAL GENERATION WITH VISUALIZATION")
    
    # Configure world generation parameters
    params = WorldGenerationParams(
        seed=123,
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
    
    # Generate visualizations using the new modular system
    print_section("GENERATING VISUALIZATIONS")
    
    visualizer = UnifiedVisualizer(output_dir="world_visualizations")
    visualizer.visualize_all(world_state, prefix=f"seed{params.seed}", dpi=150)
    
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
    
    print_section("DEMO COMPLETE")
    print("\nWorld generation successful!")
    print(f"World size: {params.size}x{params.size} ({len(world_state.chunks)} chunks)")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"\nVisualizations saved to: world_visualizations/")
    
    return world_state


def demo_multiple_seeds():
    """Generate multiple worlds with different seeds for comparison"""
    print_section("MULTI-SEED COMPARISON")
    
    seeds = [42, 123, 999]
    
    for seed in seeds:
        print(f"\nGenerating world with seed {seed}...")
        
        params = WorldGenerationParams(
            seed=seed,
            size=WorldSize.SMALL,
            num_plates=8,
            ocean_percentage=0.7,
            erosion_iterations=2,
        )
        
        pipeline = create_pipeline(params)
        world_state = pipeline.generate()
        
        # Generate visualization using topography visualizer
        visualizer = UnifiedVisualizer(output_dir="world_visualizations")
        
        # Just visualize topography for comparison
        visualizer.topography.visualize_from_chunks(
            world_state,
            f"seed{seed}_elevation.png",
            dpi=100
        )
        
        print(f"✓ World {seed} complete")
    
    print("\n✓ All seed comparisons complete!")
    print("Check world_visualizations/ to see the differences")


def demo_selective_visualization():
    """Generate a world and selectively visualize specific passes"""
    print_section("SELECTIVE VISUALIZATION DEMO")
    
    params = WorldGenerationParams(
        seed=999,
        size=WorldSize.SMALL,
        num_plates=10,
        ocean_percentage=0.65,
    )
    
    print("Generating world...")
    
    pipeline = create_pipeline(params)
    world_state = pipeline.generate()
    
    # Create visualizer
    visualizer = UnifiedVisualizer(output_dir="selective_viz")
    
    print("\nGenerating selective visualizations...")
    
    # Only visualize the most important layers
    print("  • Topography...")
    visualizer.topography.visualize_from_chunks(
        world_state,
        "topography.png",
        dpi=150
    )
    
    print("  • Rivers...")
    visualizer.rivers.visualize_from_chunks(
        world_state,
        "rivers.png",
        dpi=150
    )
    
    print("  • Climate...")
    visualizer.climate.visualize_from_chunks(
        world_state,
        "temperature.png",
        "precipitation.png",
        "climate_combined.png",
        dpi=150
    )
    
    print("\n✓ Selective visualization complete!")
    print("Check selective_viz/ directory")


def demo_summary_visualization():
    """Generate a world and create summary visualization only"""
    print_section("SUMMARY VISUALIZATION DEMO")
    
    params = WorldGenerationParams(
        seed=424242,
        size=WorldSize.SMALL,
        num_plates=10,
        ocean_percentage=0.65,
    )
    
    print("Generating world...")
    
    pipeline = create_pipeline(params)
    world_state = pipeline.generate()
    
    # Create visualizer and generate summary
    visualizer = UnifiedVisualizer(output_dir="summary_viz")
    visualizer.visualize_summary(world_state, prefix="summary", dpi=150)
    
    print("\n✓ Summary visualization complete!")
    print("Check summary_viz/ directory")


def demo_single_pass_visualization():
    """Demonstrate visualizing a single pass"""
    print_section("SINGLE PASS VISUALIZATION DEMO")
    
    params = WorldGenerationParams(
        seed=777,
        size=WorldSize.SMALL,
        num_plates=8,
        ocean_percentage=0.7,
    )
    
    print("Generating world...")
    
    pipeline = create_pipeline(params)
    world_state = pipeline.generate()
    
    # Create visualizer
    visualizer = UnifiedVisualizer(output_dir="single_pass_viz")
    
    # Visualize only Pass 10 (Rivers)
    print("\nVisualizing only Pass 10 (Rivers)...")
    visualizer.visualize_pass(world_state, pass_number=10, prefix="pass10", dpi=150)
    
    print("\n✓ Single pass visualization complete!")
    print("Check single_pass_viz/ directory")


def main():
    """Main demo function"""
    try:
        print("\n" + "🌍 "*20)
        print("WORLD BUILDER - MODULAR VISUALIZATION SYSTEM DEMO")
        print("🌍 "*20 + "\n")
        
        # Demo 1: Full world with all visualizations
        print("\n📍 DEMO 1: Complete World Generation with All Visualizations")
        world_state = demo_small_world_with_viz()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())