"""
World Builder - Pass 05: Atmosphere Napari Visualization
Interactive visualization of wind patterns and atmospheric circulation
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass05AtmosphereNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for atmospheric dynamics and wind patterns."""
    
    def __init__(self):
        super().__init__(pass_number=5, pass_name="Atmosphere")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add atmosphere layers to napari viewer.
        
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
        
        # Wind Speed
        wind_speed = self._stitch_chunks(world_state, 'wind_speed')
        if wind_speed is not None:
            viewer.add_image(
                wind_speed,
                name=self._layer_name("Wind Speed"),
                colormap='cool',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Wind Speed')}")
        
        # Wind Direction (as an additional layer for debugging)
        wind_direction = self._stitch_chunks(world_state, 'wind_direction')
        if wind_direction is not None:
            viewer.add_image(
                wind_direction,
                name=self._layer_name("Wind Direction"),
                colormap='twilight',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Wind Direction')}")
        
        return added_count