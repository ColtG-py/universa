"""
World Builder - Pass 09: Groundwater Napari Visualization
Interactive visualization of water table depth and aquifer systems
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass09GroundwaterNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for groundwater systems and water table."""
    
    def __init__(self):
        super().__init__(pass_number=9, pass_name="Groundwater")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add groundwater layers to napari viewer.
        
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
        
        # Water Table Depth
        water_table_depth = self._stitch_chunks(world_state, 'water_table_depth')
        if water_table_depth is not None:
            viewer.add_image(
                water_table_depth,
                name=self._layer_name("Water Table Depth"),
                colormap='Blues_r',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Water Table Depth')}")
        
        return added_count