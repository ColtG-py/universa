"""
World Builder - Pass 04: Geology Napari Visualization
Interactive visualization of bedrock types and mineral deposits
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE
from config import Mineral


class Pass04GeologyNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for geological features - bedrock and minerals."""
    
    def __init__(self):
        super().__init__(pass_number=4, pass_name="Geology")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add geology layers to napari viewer.
        
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
        
        # Bedrock Type
        bedrock_type = self._stitch_chunks(world_state, 'bedrock_type')
        if bedrock_type is not None:
            viewer.add_image(
                bedrock_type,
                name=self._layer_name("Bedrock Type"),
                colormap='viridis',
                opacity=0.7,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Bedrock Type')}")
        
        # Mineral Richness Layers
        sample_chunk = next(iter(world_state.chunks.values())) if world_state.chunks else None
        if sample_chunk and hasattr(sample_chunk, 'mineral_richness') and sample_chunk.mineral_richness:
            for mineral in Mineral:
                data = self._stitch_dict_attribute(world_state, 'mineral_richness', mineral)
                
                if data is not None and data.max() > 0:
                    viewer.add_image(
                        data,
                        name=self._layer_name(f"Mineral: {mineral.name.title()}"),
                        colormap='Reds',
                        opacity=0.5,
                        visible=False,  # Minerals hidden by default
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'Mineral: {mineral.name.title()}')}")
        
        return added_count