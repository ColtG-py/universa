"""
World Builder - Pass 06: Oceans Napari Visualization
Interactive visualization of ocean currents and marine dynamics
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass06OceansNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for ocean currents and circulation patterns."""
    
    def __init__(self):
        super().__init__(pass_number=6, pass_name="Oceans")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add ocean layers to napari viewer.
        
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
        
        # Check if ocean current data exists
        sample_chunk = next(iter(world_state.chunks.values())) if world_state.chunks else None
        
        # Ocean Current Speed
        if sample_chunk and hasattr(sample_chunk, 'ocean_current_speed'):
            ocean_current_speed = self._stitch_chunks(world_state, 'ocean_current_speed')
            if ocean_current_speed is not None:
                viewer.add_image(
                    ocean_current_speed,
                    name=self._layer_name("Ocean Current Speed"),
                    colormap='YlOrRd',
                    opacity=0.6,
                    visible=default_visible,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Ocean Current Speed')}")
        
        # Ocean Current Direction
        if sample_chunk and hasattr(sample_chunk, 'ocean_current_direction'):
            ocean_current_direction = self._stitch_chunks(world_state, 'ocean_current_direction')
            if ocean_current_direction is not None:
                viewer.add_image(
                    ocean_current_direction,
                    name=self._layer_name("Ocean Current Direction"),
                    colormap='twilight',
                    opacity=0.5,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Ocean Current Direction')}")
        
        return added_count