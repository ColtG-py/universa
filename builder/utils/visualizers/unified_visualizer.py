"""
World Builder - Unified Visualizer
Coordinates all pass-specific visualizers for complete world visualization
"""

from pathlib import Path
from typing import Optional

from .pass_01_planetary_viz import Pass01PlanetaryVisualizer
from .pass_02_tectonics_viz import Pass02TectonicsVisualizer
from .pass_03_topography_viz import Pass03TopographyVisualizer
from .pass_04_geology_viz import Pass04GeologyVisualizer
from .pass_05_atmosphere_viz import Pass05AtmosphereVisualizer
from .pass_06_oceans_viz import Pass06OceansVisualizer
from .pass_07_climate_viz import Pass07ClimateVisualizer
from .pass_08_erosion_viz import Pass08ErosionVisualizer
from .pass_09_groundwater_viz import Pass09GroundwaterVisualizer
from .pass_10_rivers_viz import Pass10RiversVisualizer
from .pass_11_soil_viz import Pass11SoilVisualizer
from .pass_12_microclimate_viz import Pass12MicroclimateVisualizer


class UnifiedVisualizer:
    """
    Unified visualizer that coordinates all pass-specific visualizers.
    
    This class provides a single interface to generate visualizations for
    all generation passes in one call, or selective visualization of specific
    passes as needed.
    
    Example:
        >>> from builder.utils.visualizers import UnifiedVisualizer
        >>> 
        >>> visualizer = UnifiedVisualizer(output_dir="world_visualizations")
        >>> visualizer.visualize_all(world_state, prefix="seed123", dpi=150)
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize unified visualizer with all pass visualizers.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize all pass visualizers
        self.planetary = Pass01PlanetaryVisualizer(output_dir)
        self.tectonics = Pass02TectonicsVisualizer(output_dir)
        self.topography = Pass03TopographyVisualizer(output_dir)
        self.geology = Pass04GeologyVisualizer(output_dir)
        self.atmosphere = Pass05AtmosphereVisualizer(output_dir)
        self.oceans = Pass06OceansVisualizer(output_dir)
        self.climate = Pass07ClimateVisualizer(output_dir)
        self.erosion = Pass08ErosionVisualizer(output_dir)
        self.groundwater = Pass09GroundwaterVisualizer(output_dir)
        self.rivers = Pass10RiversVisualizer(output_dir)
        self.soil = Pass11SoilVisualizer(output_dir)
        self.microclimate = Pass12MicroclimateVisualizer(output_dir)
    
    def visualize_all(
        self,
        world_state,
        prefix: str = "world",
        dpi: int = 150,
        subsample_flow: int = 1
    ) -> None:
        """
        Generate visualizations for all available passes.
        
        Args:
            world_state: WorldState object with generated data
            prefix: Prefix for output filenames
            dpi: Resolution in dots per inch
            subsample_flow: Subsampling factor for wind/ocean flow visualizations (1=full resolution)
        """
        print("\n" + "="*70)
        print("  GENERATING ALL PASS VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Pass 01: Planetary (TODO - placeholder)
        # Currently not implemented
        
        # Pass 02: Tectonics
        print("Visualizing Pass 02: Tectonics...")
        try:
            self.tectonics.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_02_tectonics.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing tectonics: {e}")
        
        # Pass 03: Topography
        print("Visualizing Pass 03: Topography...")
        try:
            self.topography.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_03_topography.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing topography: {e}")
        
        # Pass 04: Geology
        print("Visualizing Pass 04: Geology...")
        try:
            self.geology.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_04_bedrock.png",
                f"{prefix}_pass_04_minerals.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing geology: {e}")
        
        # Pass 05: Atmosphere
        print("Visualizing Pass 05: Atmosphere...")
        try:
            self.atmosphere.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_05_atmosphere.png",
                dpi,
                subsample_flow
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing atmosphere: {e}")
        
        # Pass 06: Oceans
        print("Visualizing Pass 06: Oceans...")
        try:
            self.oceans.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_06_oceans.png",
                dpi,
                subsample_flow
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing oceans: {e}")
        
        # Pass 07: Climate
        print("Visualizing Pass 07: Climate...")
        try:
            self.climate.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_07_temperature.png",
                f"{prefix}_pass_07_precipitation.png",
                f"{prefix}_pass_07_climate.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing climate: {e}")
        
        # Pass 08: Erosion (TODO - placeholder)
        # Currently not implemented
        
        # Pass 09: Groundwater
        print("Visualizing Pass 09: Groundwater...")
        try:
            self.groundwater.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_09_groundwater.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing groundwater: {e}")
        
        # Pass 10: Rivers
        print("Visualizing Pass 10: Rivers...")
        try:
            self.rivers.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_10_rivers.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing rivers: {e}")
        
        # Pass 11: Soil
        print("Visualizing Pass 11: Soil...")
        try:
            self.soil.visualize_from_chunks(
                world_state,
                f"{prefix}_pass_11_soil_type.png",
                f"{prefix}_pass_11_soil_ph.png",
                f"{prefix}_pass_11_soil.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error visualizing soil: {e}")
        
        # Pass 12: Microclimate (TODO - placeholder)
        # Currently not implemented
        
        print("\n" + "="*70)
        print(f"  ✓ All visualizations saved to {self.output_dir}/")
        print("="*70 + "\n")
    
    def visualize_pass(
        self,
        world_state,
        pass_number: int,
        prefix: str = "world",
        dpi: int = 150
    ) -> None:
        """
        Visualize a specific generation pass.
        
        Args:
            world_state: WorldState object
            pass_number: Pass number to visualize (1-14)
            prefix: Prefix for output filenames
            dpi: Resolution in dots per inch
        """
        visualizer_map = {
            1: (self.planetary, "Planetary"),
            2: (self.tectonics, "Tectonics"),
            3: (self.topography, "Topography"),
            4: (self.geology, "Geology"),
            5: (self.atmosphere, "Atmosphere"),
            6: (self.oceans, "Oceans"),
            7: (self.climate, "Climate"),
            8: (self.erosion, "Erosion"),
            9: (self.groundwater, "Groundwater"),
            10: (self.rivers, "Rivers"),
            11: (self.soil, "Soil"),
            12: (self.microclimate, "Microclimate"),
        }
        
        if pass_number not in visualizer_map:
            raise ValueError(f"Invalid pass number: {pass_number}. Must be 1-14.")
        
        visualizer, name = visualizer_map[pass_number]
        print(f"\nVisualizing Pass {pass_number:02d}: {name}...")
        
        try:
            if hasattr(visualizer, 'visualize_from_chunks'):
                visualizer.visualize_from_chunks(
                    world_state,
                    f"{prefix}_pass_{pass_number:02d}.png",
                    dpi
                )
            else:
                visualizer.visualize(world_state)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    
    def visualize_summary(
        self,
        world_state,
        prefix: str = "world",
        dpi: int = 150
    ) -> None:
        """
        Generate a summary visualization with key layers only.
        
        Creates visualizations for the most important layers:
        - Topography
        - Climate (combined)
        - Rivers
        - Tectonics
        
        Args:
            world_state: WorldState object
            prefix: Prefix for output filenames
            dpi: Resolution in dots per inch
        """
        print("\n" + "="*70)
        print("  GENERATING SUMMARY VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Essential visualizations only
        try:
            print("• Topography...")
            self.topography.visualize_from_chunks(
                world_state,
                f"{prefix}_summary_topography.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        
        try:
            print("• Climate...")
            self.climate.visualize_from_chunks(
                world_state,
                f"{prefix}_summary_temperature.png",
                f"{prefix}_summary_precipitation.png",
                f"{prefix}_summary_climate.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        
        try:
            print("• Rivers...")
            self.rivers.visualize_from_chunks(
                world_state,
                f"{prefix}_summary_rivers.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        
        try:
            print("• Tectonics...")
            self.tectonics.visualize_from_chunks(
                world_state,
                f"{prefix}_summary_tectonics.png",
                dpi
            )
        except Exception as e:
            print(f"  ⚠ Error: {e}")
        
        print("\n" + "="*70)
        print(f"  ✓ Summary visualizations saved to {self.output_dir}/")
        print("="*70 + "\n")


def create_visualization_summary(
    world_state,
    output_dir: str = "visualizations",
    dpi: int = 150
) -> None:
    """
    Convenience function to generate all visualizations.
    Provides backwards compatibility with the original visualization system.
    
    Args:
        world_state: WorldState object
        output_dir: Directory for outputs
        dpi: Resolution
    """
    visualizer = UnifiedVisualizer(output_dir)
    visualizer.visualize_all(world_state, prefix="world", dpi=dpi)