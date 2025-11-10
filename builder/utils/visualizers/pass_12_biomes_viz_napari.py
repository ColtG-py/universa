"""
World Builder - Pass 12: Biomes Napari Visualization
Interactive visualization of biome classification, vegetation density, and agricultural suitability
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass12BiomesNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for biome classification and vegetation properties."""
    
    def __init__(self):
        super().__init__(pass_number=12, pass_name="Biomes & Vegetation")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add biome and vegetation layers to napari viewer.
        
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
        
        sample_chunk = next(iter(world_state.chunks.values())) if world_state.chunks else None
        
        # Biome Type
        if sample_chunk and hasattr(sample_chunk, 'biome_type'):
            biome_type = self._stitch_chunks(world_state, 'biome_type')
            if biome_type is not None:
                viewer.add_image(
                    biome_type,
                    name=self._layer_name("Biome Type"),
                    colormap='viridis',
                    opacity=0.7,
                    visible=default_visible,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Biome Type')}")
        
        # Vegetation Density
        if sample_chunk and hasattr(sample_chunk, 'vegetation_density'):
            vegetation_density = self._stitch_chunks(world_state, 'vegetation_density')
            if vegetation_density is not None:
                viewer.add_image(
                    vegetation_density,
                    name=self._layer_name("Vegetation Density"),
                    colormap='YlGn',
                    opacity=0.6,
                    visible=False,
                    contrast_limits=(0, 1)
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Vegetation Density')}")
        
        # Forest Canopy Height
        if sample_chunk and hasattr(sample_chunk, 'forest_canopy_height'):
            canopy_height = self._stitch_chunks(world_state, 'forest_canopy_height')
            if canopy_height is not None:
                # Mask zero values (non-forest areas)
                canopy_masked = np.where(canopy_height > 0, canopy_height, np.nan)
                viewer.add_image(
                    canopy_masked,
                    name=self._layer_name("Canopy Height"),
                    colormap='BuGn',
                    opacity=0.6,
                    visible=False,
                    contrast_limits=(0, 50)
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Canopy Height')}")
        
        # Agricultural Suitability
        if sample_chunk and hasattr(sample_chunk, 'agricultural_suitability'):
            agricultural_suitability = self._stitch_chunks(world_state, 'agricultural_suitability')
            if agricultural_suitability is not None:
                viewer.add_image(
                    agricultural_suitability,
                    name=self._layer_name("Agricultural Suitability"),
                    colormap='RdYlGn',
                    opacity=0.6,
                    visible=False,
                    contrast_limits=(0, 1)
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Agricultural Suitability')}")
        
        return added_count