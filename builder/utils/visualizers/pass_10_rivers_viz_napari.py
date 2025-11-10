"""
World Builder - Pass 10: Rivers Napari Visualization
Interactive visualization of river networks and drainage patterns
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass10RiversNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for river networks and hydrological flow."""
    
    def __init__(self):
        super().__init__(pass_number=10, pass_name="Rivers")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add river layers to napari viewer.
        
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
        
        # River Presence
        river_presence = self._stitch_chunks(world_state, 'river_presence')
        if river_presence is not None:
            viewer.add_image(
                river_presence.astype(np.float32),
                name=self._layer_name("River Presence"),
                colormap='cyan',
                opacity=0.8,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('River Presence')}")
        
        # River Flow
        river_flow = self._stitch_chunks(world_state, 'river_flow')
        if river_flow is not None and river_flow.max() > 0:
            viewer.add_image(
                river_flow,
                name=self._layer_name("River Flow"),
                colormap='viridis',
                opacity=0.7,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('River Flow')}")
        
        # Drainage Basin ID
        drainage_basin_id = self._stitch_chunks(world_state, 'drainage_basin_id')
        if drainage_basin_id is not None:
            viewer.add_image(
                drainage_basin_id,
                name=self._layer_name("Drainage Basins"),
                colormap='tab20',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Drainage Basins')}")
        
        return added_count