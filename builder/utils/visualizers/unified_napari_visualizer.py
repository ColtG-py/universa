"""
World Builder - Unified Napari Visualizer
Coordinates all pass-specific visualizers in a single interactive napari viewer
"""

from pathlib import Path
from typing import Optional, List

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    print("⚠️  napari not installed. Install with: pip install 'napari[all]'")

from .pass_02_tectonics_viz_napari import Pass02TectonicsNapariVisualizer
from .pass_03_topography_viz_napari import Pass03TopographyNapariVisualizer
from .pass_04_geology_viz_napari import Pass04GeologyNapariVisualizer
from .pass_05_atmosphere_viz_napari import Pass05AtmosphereNapariVisualizer
from .pass_06_oceans_viz_napari import Pass06OceansNapariVisualizer
from .pass_07_climate_viz_napari import Pass07ClimateNapariVisualizer
from .pass_09_groundwater_viz_napari import Pass09GroundwaterNapariVisualizer
from .pass_10_rivers_viz_napari import Pass10RiversNapariVisualizer
from .pass_11_soil_viz_napari import Pass11SoilNapariVisualizer
from .pass_12_biomes_viz_napari import Pass12BiomesNapariVisualizer
from .pass_13_fauna_viz_napari import Pass13FaunaNapariVisualizer


class UnifiedNapariVisualizer:
    """
    Unified Napari visualizer that coordinates all pass-specific visualizers.
    
    This creates a single interactive napari viewer with all generation passes
    as toggleable layers. Users can interactively explore the world by:
    - Toggling layers on/off
    - Adjusting layer opacity
    - Changing colormaps
    - Panning and zooming
    
    Example:
        >>> from builder.utils.visualizers import UnifiedNapariVisualizer
        >>> 
        >>> visualizer = UnifiedNapariVisualizer()
        >>> visualizer.view_world(world_state)
    """
    
    def __init__(self):
        """Initialize unified Napari visualizer with all pass visualizers."""
        if not NAPARI_AVAILABLE:
            raise ImportError(
                "napari is not installed. Install with: pip install 'napari[all]'"
            )
        
        # Initialize all pass visualizers
        self.visualizers = {
            2: Pass02TectonicsNapariVisualizer(),
            3: Pass03TopographyNapariVisualizer(),
            4: Pass04GeologyNapariVisualizer(),
            5: Pass05AtmosphereNapariVisualizer(),
            6: Pass06OceansNapariVisualizer(),
            7: Pass07ClimateNapariVisualizer(),
            9: Pass09GroundwaterNapariVisualizer(),
            10: Pass10RiversNapariVisualizer(),
            11: Pass11SoilNapariVisualizer(),
            12: Pass12BiomesNapariVisualizer(),
            13: Pass13FaunaNapariVisualizer(),
        }
        
        self.viewer = None
    
    def view_world(
        self,
        world_state,
        passes: Optional[List[int]] = None,
        title: str = "World Builder - Interactive Layer Viewer"
    ) -> None:
        """
        Launch napari viewer with all available world layers.
        
        Args:
            world_state: WorldState object with generated data
            passes: Optional list of pass numbers to display. If None, shows all available.
            title: Window title
        
        Example:
            >>> visualizer = UnifiedNapariVisualizer()
            >>> # Show all passes
            >>> visualizer.view_world(world_state)
            >>> 
            >>> # Show only specific passes
            >>> visualizer.view_world(world_state, passes=[3, 7, 10, 12])
        """
        if not NAPARI_AVAILABLE:
            print("❌ napari not available. Cannot launch viewer.")
            return
        
        print("\n" + "="*70)
        print("  LAUNCHING INTERACTIVE NAPARI VIEWER")
        print("="*70 + "\n")
        
        # Create viewer
        self.viewer = napari.Viewer(title=title)
        
        # Add layers from each pass
        total_layers = 0
        
        for pass_num, visualizer in sorted(self.visualizers.items()):
            # Skip if user specified passes and this isn't in the list
            if passes is not None and pass_num not in passes:
                continue
            
            print(f"Loading Pass {pass_num:02d}: {visualizer.pass_name}...")
            
            try:
                # Pass 3 (Elevation) should be visible by default, others hidden
                default_visible = (pass_num == 3)
                
                layers_added = visualizer.add_layers(
                    self.viewer,
                    world_state,
                    default_visible=default_visible
                )
                total_layers += layers_added
            
            except Exception as e:
                print(f"  ⚠️  Error loading pass {pass_num}: {e}")
        
        if total_layers == 0:
            print("⚠️  No layers found to display!")
            return
        
        print(f"\n✓ Loaded {total_layers} layers into napari viewer")
        print("\nInteractive Controls:")
        print("  • Toggle layers: Use layer list in left panel")
        print("  • Adjust opacity: Use opacity slider per layer")
        print("  • Change colormaps: Use colormap dropdown per layer")
        print("  • Pan: Click and drag, or use arrow keys")
        print("  • Zoom: Mouse wheel, pinch gesture, or +/- keys")
        print("  • Reset view: Click the 'home' icon")
        print("  • Screenshot: File → Save Screenshot")
        
        print("\nTip: Only Elevation is visible by default. Toggle other layers on to explore!")
        print("="*70 + "\n")
        
        # Run the viewer
        napari.run()
    
    def view_pass(
        self,
        world_state,
        pass_number: int,
        title: Optional[str] = None
    ) -> None:
        """
        View a single generation pass.
        
        Args:
            world_state: WorldState object
            pass_number: Pass number to visualize (2-13)
            title: Optional window title
        """
        if pass_number not in self.visualizers:
            raise ValueError(
                f"Invalid pass number: {pass_number}. "
                f"Available passes: {sorted(self.visualizers.keys())}"
            )
        
        visualizer = self.visualizers[pass_number]
        
        if title is None:
            title = f"World Builder - Pass {pass_number:02d}: {visualizer.pass_name}"
        
        print(f"\nLaunching viewer for Pass {pass_number:02d}: {visualizer.pass_name}...")
        
        # Create viewer
        self.viewer = napari.Viewer(title=title)
        
        # Add layers from this pass only
        layers_added = visualizer.add_layers(
            self.viewer,
            world_state,
            default_visible=True
        )
        
        if layers_added == 0:
            print("⚠️  No layers available for this pass!")
            return
        
        print(f"✓ Loaded {layers_added} layer(s)")
        
        # Run the viewer
        napari.run()
    
    def save_screenshots(
        self,
        world_state,
        output_dir: str = "visualizations",
        passes: Optional[List[int]] = None
    ) -> None:
        """
        Generate and save screenshots for all passes without launching GUI.
        
        This is useful for automated documentation or batch processing.
        
        Args:
            world_state: WorldState object
            output_dir: Directory to save screenshots
            passes: Optional list of pass numbers. If None, captures all.
        """
        if not NAPARI_AVAILABLE:
            print("❌ napari not available. Cannot generate screenshots.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("  GENERATING NAPARI SCREENSHOTS")
        print("="*70 + "\n")
        
        # Create headless viewer
        self.viewer = napari.Viewer(show=False)
        
        # Add all layers first
        for pass_num, visualizer in sorted(self.visualizers.items()):
            if passes is not None and pass_num not in passes:
                continue
            
            try:
                visualizer.add_layers(self.viewer, world_state, default_visible=False)
            except Exception as e:
                print(f"  ⚠️  Error loading pass {pass_num}: {e}")
        
        # Save screenshot for each pass
        for pass_num, visualizer in sorted(self.visualizers.items()):
            if passes is not None and pass_num not in passes:
                continue
            
            print(f"Capturing Pass {pass_num:02d}: {visualizer.pass_name}...")
            
            # Find layers belonging to this pass
            pass_prefix = f"Pass {pass_num:02d}:"
            pass_layers = [
                layer.name for layer in self.viewer.layers
                if layer.name.startswith(pass_prefix)
            ]
            
            if not pass_layers:
                continue
            
            # Show only this pass's layers
            for layer in self.viewer.layers:
                layer.visible = layer.name in pass_layers
            
            # Save screenshot
            filename = output_path / f"pass_{pass_num:02d}_{visualizer.pass_name.lower().replace(' ', '_')}.png"
            screenshot = self.viewer.screenshot(canvas_only=True, flash=False)
            
            try:
                from PIL import Image
                img = Image.fromarray(screenshot)
                img.save(filename)
                print(f"  ✓ Saved: {filename}")
            except ImportError:
                # Fallback to numpy
                np.save(filename.with_suffix('.npy'), screenshot)
                print(f"  ✓ Saved: {filename.with_suffix('.npy')}")
        
        self.viewer.close()
        
        print("\n" + "="*70)
        print(f"  ✓ Screenshots saved to {output_dir}/")
        print("="*70 + "\n")


def view_world_interactive(
    world_state,
    passes: Optional[List[int]] = None,
    title: str = "World Builder - Interactive Layer Viewer"
) -> None:
    """
    Convenience function to quickly view world state in napari.
    
    Args:
        world_state: WorldState object
        passes: Optional list of specific passes to show
        title: Window title
    
    Example:
        >>> from utils.visualizers import view_world_interactive
        >>> view_world_interactive(world_state)
        >>> 
        >>> # Or show specific passes
        >>> view_world_interactive(world_state, passes=[3, 7, 10, 12])
    """
    if not NAPARI_AVAILABLE:
        print("❌ napari is not installed.")
        print("   Install with: pip install 'napari[all]'")
        return
    
    visualizer = UnifiedNapariVisualizer()
    visualizer.view_world(world_state, passes=passes, title=title)