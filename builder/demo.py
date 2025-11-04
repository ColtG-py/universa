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
from utils.visualization import LayerVisualizer


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
        seed=12,
        size=WorldSize.SMALL,  # 512x512 world
        
        # Planetary parameters
        planet_radius_km=6371.0,
        gravity=9.8,
        axial_tilt=23.5,
        rotation_hours=24.0,
        
        # Tectonic parameters
        num_plates=30,
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
    
    # Generate visualizations
    print_section("GENERATING VISUALIZATIONS")
    
    visualizer = LayerVisualizer(output_dir="world_visualizations")
    visualizer.visualize_all_layers(world_state, prefix=f"seed{params.seed}", dpi=150)
    
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
        
        # Generate visualization
        visualizer = LayerVisualizer(output_dir="world_visualizations")
        
        # Just visualize elevation for comparison
        chunk_data = visualizer._collect_chunk_data(world_state)
        if chunk_data['elevation'] is not None:
            visualizer.visualize_elevation(
                chunk_data['elevation'],
                f"seed{seed}_elevation.png",
                dpi=100
            )
        
        print(f"✓ World {seed} complete")
    
    print("\n✓ All seed comparisons complete!")
    print("Check world_visualizations/ to see the differences")


def demo_layer_by_layer():
    """Generate a world and show each layer progressively"""
    print_section("LAYER-BY-LAYER GENERATION DEMO")
    
    params = WorldGenerationParams(
        seed=424242,
        size=WorldSize.SMALL,
        num_plates=10,
        ocean_percentage=0.65,
    )
    
    print("Generating world with detailed layer visualization...")
    
    pipeline = create_pipeline(params)
    world_state = pipeline.generate()
    
    # Create visualizer
    visualizer = LayerVisualizer(output_dir="layer_progression")
    
    # Collect all data
    chunk_data = visualizer._collect_chunk_data(world_state)
    
    # Visualize each layer individually
    print("\nGenerating individual layer visualizations...")
    
    layer_map = [
        ('elevation', 'Elevation', 'terrain'),
        ('plate_id', 'Tectonic Plates', 'tab20'),
        ('tectonic_stress', 'Tectonic Stress', 'YlOrRd'),
        ('temperature_c', 'Temperature', 'RdYlBu_r'),
        ('precipitation_mm', 'Precipitation', 'Blues'),
        ('bedrock_type', 'Bedrock Type', 'Set3'),
        ('river_presence', 'Rivers', 'Blues'),
        ('water_table_depth', 'Water Table', 'Blues_r'),
        ('soil_type', 'Soil Type', 'YlOrBr'),
        ('soil_ph', 'Soil pH', 'RdYlGn'),
    ]
    
    for layer_name, display_name, cmap in layer_map:
        if chunk_data[layer_name] is not None:
            visualizer.export_to_pil(
                chunk_data[layer_name],
                cmap,
                f"{layer_name}.png"
            )
            print(f"  ✓ {display_name}")
    
    print("\n✓ Layer-by-layer visualization complete!")
    print("Check layer_progression/ directory")


def main():
    """Main demo function"""
    try:
        print("\n" + "🌍 "*20)
        print("WORLD BUILDER - GENERATION & VISUALIZATION DEMO")
        print("🌍 "*20 + "\n")
        
        # Demo 1: Full world with all visualizations
        print("\n📍 DEMO 1: Complete World Generation with Visualization")
        world_state = demo_small_world_with_viz()
        
        # # Demo 2: Multiple seeds for comparison
        # print("\n📍 DEMO 2: Multi-Seed Comparison")
        # demo_multiple_seeds()
        
        # # Demo 3: Layer-by-layer visualization
        # print("\n📍 DEMO 3: Layer-by-Layer Visualization")
        # demo_layer_by_layer()
        
        print_section("ALL DEMOS COMPLETE")
        print("\nThe World Builder generation engine is ready for use!")
        print("\nKey Features:")
        print("  ✓ Modular generation pipeline with 14 passes")
        print("  ✓ Fully configurable parameters")
        print("  ✓ Chunk-based architecture for scalability")
        print("  ✓ Deterministic generation (same seed = same world)")
        print("  ✓ Scientific accuracy in climate, geology, and hydrology")
        print("  ✓ Comprehensive layer visualization")
        
        print("\nVisualization Directories:")
        print("  • world_visualizations/ - Full world renders")
        print("  • layer_progression/ - Individual layer exports")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())