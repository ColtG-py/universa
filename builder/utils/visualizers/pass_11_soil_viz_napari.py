"""
World Builder - Pass 11: Soil Napari Visualization
Interactive visualization of soil types, pH, and drainage properties
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass11SoilNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for soil properties and characteristics."""
    
    def __init__(self):
        super().__init__(pass_number=11, pass_name="Soil")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add soil layers to napari viewer.
        
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
        
        # Soil Type
        soil_type = self._stitch_chunks(world_state, 'soil_type')
        if soil_type is not None:
            viewer.add_image(
                soil_type,
                name=self._layer_name("Soil Type"),
                colormap='YlOrBr',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Soil Type')}")
        
        # Soil pH
        soil_ph = self._stitch_chunks(world_state, 'soil_ph')
        if soil_ph is not None:
            viewer.add_image(
                soil_ph,
                name=self._layer_name("Soil pH"),
                colormap='RdYlGn',
                opacity=0.6,
                visible=default_visible,
                contrast_limits=(4, 10)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Soil pH')}")
        
        # Soil Drainage
        soil_drainage = self._stitch_chunks(world_state, 'soil_drainage')
        if soil_drainage is not None:
            viewer.add_image(
                soil_drainage,
                name=self._layer_name("Soil Drainage"),
                colormap='Blues',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Soil Drainage')}")
        
        return added_count