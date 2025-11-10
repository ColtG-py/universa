"""
World Builder - Pass 07: Climate Napari Visualization
Interactive visualization of temperature and precipitation patterns
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass07ClimateNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for climate patterns - temperature and precipitation."""
    
    def __init__(self):
        super().__init__(pass_number=7, pass_name="Climate")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add climate layers to napari viewer.
        
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
        
        # Temperature
        temperature = self._stitch_chunks(world_state, 'temperature_c')
        if temperature is not None:
            viewer.add_image(
                temperature,
                name=self._layer_name("Temperature"),
                colormap='RdYlBu_r',
                opacity=0.7,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Temperature')}")
        
        # Precipitation
        precipitation = self._stitch_chunks(world_state, 'precipitation_mm')
        if precipitation is not None:
            viewer.add_image(
                precipitation,
                name=self._layer_name("Precipitation"),
                colormap='Blues',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Precipitation')}")
        
        return added_count