"""
World Builder - Pass 03: Topography Napari Visualization
Interactive visualization of elevation and terrain
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass03TopographyNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for elevation and topography."""
    
    def __init__(self):
        super().__init__(pass_number=3, pass_name="Topography")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = True  # Elevation should be visible by default
    ) -> int:
        """
        Add topography layers to napari viewer.
        
        Args:
            viewer: Napari viewer instance
            world_state: WorldState object
            default_visible: Whether layers should be visible by default
        
        Returns:
            Number of layers added
        """
        if not NAPARI_AVAILABLE:
            return 0
        
        added_count = 0
        
        # Elevation
        elevation = self._stitch_chunks(world_state, 'elevation')
        if elevation is not None:
            viewer.add_image(
                elevation,
                name=self._layer_name("Elevation"),
                colormap='terrain',
                opacity=0.8,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Elevation')}")
        
        return added_count