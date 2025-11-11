"""
World Builder - Pass 15: Magic & Ley Lines Napari Visualization
Interactive visualization of magical energy networks and enchanted locations
"""

import numpy as np
from typing import Optional

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass15MagicNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for magical energy systems."""
    
    def __init__(self):
        super().__init__(pass_number=15, pass_name="Magic & Ley Lines")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add magic layers to napari viewer.
        
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
        
        # Mana Concentration
        mana_concentration = self._stitch_chunks(world_state, 'mana_concentration')
        if mana_concentration is not None:
            viewer.add_image(
                mana_concentration,
                name=self._layer_name("Mana Concentration"),
                colormap='viridis',
                opacity=0.7,
                visible=default_visible,
                blending='additive',
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Mana Concentration')}")
        
        # Ley Line Network
        ley_line_presence = self._stitch_chunks(world_state, 'ley_line_presence')
        if ley_line_presence is not None:
            viewer.add_image(
                ley_line_presence.astype(np.float32),
                name=self._layer_name("Ley Lines"),
                colormap='cyan',
                opacity=0.8,
                visible=default_visible,
                blending='additive',
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Ley Lines')}")
        
        # Ley Line Nodes
        ley_line_nodes = self._stitch_chunks(world_state, 'ley_line_node')
        if ley_line_nodes is not None:
            viewer.add_image(
                ley_line_nodes.astype(np.float32),
                name=self._layer_name("Ley Line Nodes"),
                colormap='yellow',
                opacity=1.0,
                visible=False,  # Hidden by default
                blending='additive',
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Ley Line Nodes')}")
        
        # Corrupted Zones
        corrupted_zones = self._stitch_chunks(world_state, 'corrupted_zone')
        if corrupted_zones is not None and corrupted_zones.sum() > 0:
            viewer.add_image(
                corrupted_zones.astype(np.float32),
                name=self._layer_name("Corrupted Zones"),
                colormap='magma',
                opacity=0.7,
                visible=False,  # Hidden by default
                blending='additive',
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Corrupted Zones')}")
        
        # Elemental Affinity
        elemental_affinity = self._stitch_chunks(world_state, 'elemental_affinity')
        if elemental_affinity is not None:
            viewer.add_image(
                elemental_affinity,
                name=self._layer_name("Elemental Affinity"),
                colormap='tab10',
                opacity=0.5,
                visible=False,  # Hidden by default
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Elemental Affinity')}")
        
        # Enchanted Locations as Points
        enchanted_points = self._collect_enchanted_locations(world_state)
        if enchanted_points is not None and len(enchanted_points) > 0:
            # Separate by type for color coding
            mana_wells = [p for p in enchanted_points if p[2] == "mana_well"]
            fey_groves = [p for p in enchanted_points if p[2] == "fey_grove"]
            dragon_lairs = [p for p in enchanted_points if p[2] == "dragon_lair"]
            corrupted_sites = [p for p in enchanted_points if p[2] == "corrupted_site"]
            
            if mana_wells:
                coords = np.array([[p[0], p[1]] for p in mana_wells])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Mana Wells"),
                    size=10,
                    face_color='cyan',
                    edge_color='white',
                    edge_width=2,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Mana Wells')} ({len(mana_wells)} points)")
            
            if fey_groves:
                coords = np.array([[p[0], p[1]] for p in fey_groves])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Fey Groves"),
                    size=10,
                    face_color='green',
                    edge_color='white',
                    edge_width=2,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Fey Groves')} ({len(fey_groves)} points)")
            
            if dragon_lairs:
                coords = np.array([[p[0], p[1]] for p in dragon_lairs])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Dragon Lairs"),
                    size=12,
                    face_color='red',
                    edge_color='orange',
                    edge_width=3,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Dragon Lairs')} ({len(dragon_lairs)} points)")
            
            if corrupted_sites:
                coords = np.array([[p[0], p[1]] for p in corrupted_sites])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Corrupted Sites"),
                    size=10,
                    face_color='purple',
                    edge_color='black',
                    edge_width=2,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Corrupted Sites')} ({len(corrupted_sites)} points)")
        
        return added_count
    
    def _collect_enchanted_locations(self, world_state) -> Optional[list]:
        """
        Collect all enchanted locations from chunks.
        
        Returns:
            List of (x, y, type, power) tuples
        """
        locations = []
        
        for chunk in world_state.chunks.values():
            if hasattr(chunk, 'enchanted_locations') and chunk.enchanted_locations:
                for loc in chunk.enchanted_locations:
                    locations.append((
                        loc.location_x,
                        loc.location_y,
                        loc.location_type,
                        loc.power_level
                    ))
        
        return locations if locations else None