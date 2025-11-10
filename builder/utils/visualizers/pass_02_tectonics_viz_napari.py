"""
World Builder - Pass 02: Tectonics Napari Visualization
Interactive visualization of tectonic plates and stress patterns
"""

import numpy as np
from typing import Optional

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass02TectonicsNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for tectonic plate generation and dynamics."""
    
    def __init__(self):
        super().__init__(pass_number=2, pass_name="Tectonics")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add tectonic layers to napari viewer.
        
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
        
        # Plate IDs
        plate_id = self._stitch_chunks(world_state, 'plate_id')
        if plate_id is not None:
            viewer.add_image(
                plate_id,
                name=self._layer_name("Plate IDs"),
                colormap='tab20',
                opacity=0.7,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Plate IDs')}")
        
        # Tectonic Stress
        tectonic_stress = self._stitch_chunks(world_state, 'tectonic_stress')
        if tectonic_stress is not None:
            viewer.add_image(
                tectonic_stress,
                name=self._layer_name("Tectonic Stress"),
                colormap='YlOrRd',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Tectonic Stress')}")
        
        return added_count