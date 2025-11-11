"""
World Builder - Pass 16: Settlement Sites Napari Visualization
Interactive visualization of settlement locations, types, and specializations
"""

import numpy as np
from typing import Optional

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass16SettlementsNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for settlement sites."""
    
    def __init__(self):
        super().__init__(pass_number=16, pass_name="Settlement Sites")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add settlement layers to napari viewer.
        
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
        
        # Settlement presence map (rasterized)
        settlement_map = self._stitch_chunks(world_state, 'settlement_presence')
        if settlement_map is not None:
            # Create labels layer for settlement types
            viewer.add_labels(
                settlement_map,
                name=self._layer_name("Settlement Map"),
                opacity=0.7,
                visible=False,  # Hidden by default in favor of points
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Settlement Map')}")
        
        # Collect all settlements as points
        settlements = self._collect_settlements(world_state)
        
        if settlements and len(settlements) > 0:
            # Separate by type for color coding
            metropolises = [s for s in settlements if s['type'] == 4 and not s['is_ruin']]
            cities = [s for s in settlements if s['type'] == 3 and not s['is_ruin']]
            towns = [s for s in settlements if s['type'] == 2 and not s['is_ruin']]
            villages = [s for s in settlements if s['type'] == 1 and not s['is_ruin']]
            hamlets = [s for s in settlements if s['type'] == 0 and not s['is_ruin']]
            fortresses = [s for s in settlements if s['type'] == 5 and not s['is_ruin']]
            monasteries = [s for s in settlements if s['type'] == 6 and not s['is_ruin']]
            ruins = [s for s in settlements if s['is_ruin']]
            
            # Add metropolises
            if metropolises:
                coords = np.array([[s['x'], s['y']] for s in metropolises])
                sizes = np.array([self._pop_to_size(s['population']) for s in metropolises])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Metropolises"),
                    size=sizes,
                    face_color='gold',
                    edge_color='orange',
                    edge_width=3,
                    visible=default_visible,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Metropolises')} ({len(metropolises)} points)")
            
            # Add cities
            if cities:
                coords = np.array([[s['x'], s['y']] for s in cities])
                sizes = np.array([self._pop_to_size(s['population']) for s in cities])
                
                # Mark capitals with different color
                capitals = [s for s in cities if s['is_capital']]
                regular_cities = [s for s in cities if not s['is_capital']]
                
                if capitals:
                    cap_coords = np.array([[s['x'], s['y']] for s in capitals])
                    cap_sizes = np.array([self._pop_to_size(s['population']) for s in capitals])
                    viewer.add_points(
                        cap_coords,
                        name=self._layer_name("Capital Cities"),
                        size=cap_sizes,
                        face_color='purple',
                        edge_color='white',
                        edge_width=3,
                        visible=default_visible,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name('Capital Cities')} ({len(capitals)} points)")
                
                if regular_cities:
                    reg_coords = np.array([[s['x'], s['y']] for s in regular_cities])
                    reg_sizes = np.array([self._pop_to_size(s['population']) for s in regular_cities])
                    viewer.add_points(
                        reg_coords,
                        name=self._layer_name("Cities"),
                        size=reg_sizes,
                        face_color='red',
                        edge_color='white',
                        edge_width=2,
                        visible=default_visible,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name('Cities')} ({len(regular_cities)} points)")
            
            # Add towns
            if towns:
                coords = np.array([[s['x'], s['y']] for s in towns])
                sizes = np.array([self._pop_to_size(s['population']) for s in towns])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Towns"),
                    size=sizes,
                    face_color='orange',
                    edge_color='white',
                    edge_width=1,
                    visible=default_visible,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Towns')} ({len(towns)} points)")
            
            # Add villages
            if villages:
                coords = np.array([[s['x'], s['y']] for s in villages])
                sizes = np.array([self._pop_to_size(s['population']) for s in villages])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Villages"),
                    size=sizes,
                    face_color='yellow',
                    edge_color='gray',
                    edge_width=1,
                    visible=False,  # Hidden by default to reduce clutter
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Villages')} ({len(villages)} points)")
            
            # Add hamlets
            if hamlets:
                coords = np.array([[s['x'], s['y']] for s in hamlets])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Hamlets"),
                    size=6,
                    face_color='lightgreen',
                    edge_color='gray',
                    edge_width=0.5,
                    visible=False,  # Hidden by default
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Hamlets')} ({len(hamlets)} points)")
            
            # Add fortresses
            if fortresses:
                coords = np.array([[s['x'], s['y']] for s in fortresses])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Fortresses"),
                    size=12,
                    face_color='darkred',
                    edge_color='black',
                    edge_width=2,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Fortresses')} ({len(fortresses)} points)")
            
            # Add monasteries
            if monasteries:
                coords = np.array([[s['x'], s['y']] for s in monasteries])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Monasteries"),
                    size=10,
                    face_color='lightblue',
                    edge_color='white',
                    edge_width=2,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Monasteries')} ({len(monasteries)} points)")
            
            # Add ruins
            if ruins:
                coords = np.array([[s['x'], s['y']] for s in ruins])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Ruins"),
                    size=8,
                    face_color='gray',
                    edge_color='black',
                    edge_width=1,
                    symbol='x',
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Ruins')} ({len(ruins)} points)")
            
            # Also add by specialization (alternative view)
            # Group by specialization
            spec_names = {
                0: "Agricultural",
                1: "Mining",
                2: "Fishing/Port",
                3: "Trade Hub",
                4: "Fortress",
                5: "Religious",
                6: "Magical",
                7: "Manufacturing",
            }
            
            spec_colors = {
                0: 'green',
                1: 'brown',
                2: 'blue',
                3: 'yellow',
                4: 'red',
                5: 'lightblue',
                6: 'purple',
                7: 'orange',
            }
            
            # Create specialization layers (hidden by default)
            for spec_id, spec_name in spec_names.items():
                spec_settlements = [s for s in settlements if s['specialization'] == spec_id and not s['is_ruin']]
                
                if spec_settlements:
                    coords = np.array([[s['x'], s['y']] for s in spec_settlements])
                    viewer.add_points(
                        coords,
                        name=self._layer_name(f"Spec: {spec_name}"),
                        size=8,
                        face_color=spec_colors[spec_id],
                        edge_color='white',
                        edge_width=1,
                        visible=False,  # Hidden by default
                    )
                    added_count += 1
        
        return added_count
    
    def _collect_settlements(self, world_state) -> Optional[list]:
        """
        Collect all settlements from chunks.
        
        Returns:
            List of settlement dictionaries
        """
        settlements = []
        
        for chunk in world_state.chunks.values():
            if hasattr(chunk, 'settlements') and chunk.settlements:
                for settlement in chunk.settlements:
                    settlements.append({
                        'x': settlement.x,
                        'y': settlement.y,
                        'type': settlement.settlement_type,
                        'population': settlement.population,
                        'specialization': settlement.specialization,
                        'is_ruin': settlement.is_ruin,
                        'is_capital': settlement.is_capital,
                    })
        
        return settlements if settlements else None
    
    def _pop_to_size(self, population: int) -> float:
        """Convert population to point size for visualization."""
        # Logarithmic scaling
        import math
        if population <= 0:
            return 5
        
        # Size range: 8-24 pixels
        log_pop = math.log10(population)
        min_log = math.log10(20)    # Hamlet
        max_log = math.log10(100000)  # Metropolis
        
        normalized = (log_pop - min_log) / (max_log - min_log)
        normalized = np.clip(normalized, 0, 1)
        
        size = 8 + normalized * 16  # 8 to 24 pixels
        return size