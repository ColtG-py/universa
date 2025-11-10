"""
World Builder - Pass 14: Natural Resources Napari Visualization
Interactive visualization of harvestable resources - minerals, timber, agriculture, fishing
"""

import numpy as np

from utils.visualizers.base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE
from config import Mineral, TimberType, QuarryType


class Pass14ResourcesNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for natural resource distributions."""
    
    def __init__(self):
        super().__init__(pass_number=14, pass_name="Natural Resources")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add resource layers to napari viewer.
        
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
        
        # Mineral Deposits (concentrated ore veins)
        if sample_chunk and hasattr(sample_chunk, 'mineral_deposits') and sample_chunk.mineral_deposits:
            mineral_colormaps = {
                Mineral.IRON: 'Greys',
                Mineral.COPPER: 'YlOrBr',
                Mineral.GOLD: 'YlOrRd',
                Mineral.SILVER: 'PuBuGn',
                Mineral.COAL: 'Greys',
                Mineral.SALT: 'Blues',
                Mineral.DIAMOND: 'PuRd',
                Mineral.EMERALD: 'Greens',
            }
            
            for mineral in Mineral:
                data = self._stitch_dict_attribute(world_state, 'mineral_deposits', mineral)
                
                if data is not None and data.max() > 0:
                    cmap = mineral_colormaps.get(mineral, 'viridis')
                    
                    viewer.add_image(
                        data,
                        name=self._layer_name(f"Mineral Vein: {mineral.name.title()}"),
                        colormap=cmap,
                        opacity=0.6,
                        visible=False,  # Hidden by default
                        contrast_limits=(0, 1)
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'Mineral Vein: {mineral.name.title()}')}")
        
        # Quarry Quality (building stone)
        quarry_quality = self._stitch_chunks(world_state, 'quarry_quality')
        if quarry_quality is not None and quarry_quality.max() > 0:
            viewer.add_image(
                quarry_quality,
                name=self._layer_name("Quarry Quality"),
                colormap='YlOrBr',
                opacity=0.6,
                visible=default_visible,
                contrast_limits=(0, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Quarry Quality')}")
        
        # Quarry Type
        quarry_type = self._stitch_chunks(world_state, 'quarry_type')
        if quarry_type is not None:
            # Mask zero values (no quarry)
            quarry_masked = np.where(quarry_type > 0, quarry_type, np.nan)
            viewer.add_image(
                quarry_masked,
                name=self._layer_name("Quarry Type"),
                colormap='tab10',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Quarry Type')}")
        
        # Timber Quality
        timber_quality = self._stitch_chunks(world_state, 'timber_quality')
        if timber_quality is not None and timber_quality.max() > 0:
            viewer.add_image(
                timber_quality,
                name=self._layer_name("Timber Quality"),
                colormap='YlGn',
                opacity=0.6,
                visible=default_visible,
                contrast_limits=(0, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Timber Quality')}")
        
        # Timber Type
        timber_type = self._stitch_chunks(world_state, 'timber_type')
        if timber_type is not None:
            # Mask zero values (no timber)
            timber_masked = np.where(timber_type > 0, timber_type, np.nan)
            viewer.add_image(
                timber_masked,
                name=self._layer_name("Timber Type"),
                colormap='tab10',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Timber Type')}")
        
        # Agricultural Yield
        agricultural_yield = self._stitch_chunks(world_state, 'agricultural_yield')
        if agricultural_yield is not None and agricultural_yield.max() > 0:
            viewer.add_image(
                agricultural_yield,
                name=self._layer_name("Agricultural Yield"),
                colormap='RdYlGn',
                opacity=0.6,
                visible=default_visible,
                contrast_limits=(0, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Agricultural Yield')}")
        
        # Fishing Quality
        fishing_quality = self._stitch_chunks(world_state, 'fishing_quality')
        if fishing_quality is not None and fishing_quality.max() > 0:
            viewer.add_image(
                fishing_quality,
                name=self._layer_name("Fishing Quality"),
                colormap='GnBu',
                opacity=0.6,
                visible=default_visible,
                contrast_limits=(0, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Fishing Quality')}")
        
        # Rare Resources (gemstones, magical materials)
        rare_resources = self._stitch_chunks(world_state, 'rare_resources')
        if rare_resources is not None and rare_resources.max() > 0:
            # Mask low values to show only significant rare resource sites
            rare_masked = np.where(rare_resources > 0.5, rare_resources, np.nan)
            viewer.add_image(
                rare_masked,
                name=self._layer_name("Rare Resources"),
                colormap='plasma',
                opacity=0.8,
                visible=default_visible,
                contrast_limits=(0.5, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Rare Resources')}")
        
        # Resource Accessibility (extraction difficulty)
        accessibility = self._stitch_chunks(world_state, 'resource_accessibility')
        if accessibility is not None:
            viewer.add_image(
                accessibility,
                name=self._layer_name("Resource Accessibility"),
                colormap='RdYlGn',
                opacity=0.5,
                visible=False,
                contrast_limits=(0, 1)
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Resource Accessibility')}")
        
        return added_count