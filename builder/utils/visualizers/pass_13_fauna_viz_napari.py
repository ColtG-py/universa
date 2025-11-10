"""
World Builder - Pass 13: Fauna Napari Visualization
Interactive visualization of wildlife distribution, apex predator territories, and migration routes
"""

import numpy as np

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE
from config import FaunaCategory


class Pass13FaunaNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for fauna distribution and wildlife patterns."""
    
    def __init__(self):
        super().__init__(pass_number=13, pass_name="Fauna Distribution")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add fauna layers to napari viewer.
        
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
        
        # Fauna Density Layers (by category)
        if sample_chunk and hasattr(sample_chunk, 'fauna_density') and sample_chunk.fauna_density:
            fauna_colormaps = {
                FaunaCategory.HERBIVORE_GRAZER: 'YlGn',
                FaunaCategory.HERBIVORE_BROWSER: 'Greens',
                FaunaCategory.PREDATOR_APEX: 'Reds',
                FaunaCategory.PREDATOR_MEDIUM: 'Oranges',
                FaunaCategory.PREDATOR_SMALL: 'YlOrRd',
                FaunaCategory.OMNIVORE: 'YlOrBr',
                FaunaCategory.AQUATIC_FISH: 'Blues',
                FaunaCategory.AQUATIC_AMPHIBIAN: 'GnBu',
                FaunaCategory.AVIAN_RAPTOR: 'Purples',
                FaunaCategory.AVIAN_SONGBIRD: 'RdPu',
                FaunaCategory.AVIAN_MIGRATORY: 'PuBuGn',
                FaunaCategory.INSECT: 'Greys',
            }
            
            for fauna_cat in FaunaCategory:
                data = self._stitch_dict_attribute(world_state, 'fauna_density', fauna_cat)
                
                if data is not None and data.max() > 0:
                    cmap = fauna_colormaps.get(fauna_cat, 'viridis')
                    layer_name = fauna_cat.name.replace('_', ' ').title()
                    
                    viewer.add_image(
                        data,
                        name=self._layer_name(f"Fauna: {layer_name}"),
                        colormap=cmap,
                        opacity=0.5,
                        visible=False,  # Hidden by default to avoid clutter
                        contrast_limits=(0, 1)
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'Fauna: {layer_name}')}")
        
        # Apex Predator Territories
        apex_territories = self._stitch_chunks(world_state, 'apex_predator_territories')
        if apex_territories is not None and apex_territories.max() > 0:
            viewer.add_image(
                apex_territories,
                name=self._layer_name("Apex Territories"),
                colormap='tab20',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Apex Territories')}")
        
        # Migration Routes
        migration_routes = self._stitch_chunks(world_state, 'migration_routes')
        if migration_routes is not None and migration_routes.any():
            viewer.add_image(
                migration_routes.astype(np.float32),
                name=self._layer_name("Migration Routes"),
                colormap='autumn',
                opacity=0.7,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Migration Routes')}")
        
        return added_count