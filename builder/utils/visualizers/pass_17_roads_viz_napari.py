"""
World Builder - Pass 17: Road Networks Napari Visualization
Interactive visualization of road networks, bridges, and travel times.
"""

import numpy as np
from typing import Optional
from scipy.ndimage import distance_transform_edt

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass17RoadsNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for road networks."""
    
    def __init__(self):
        super().__init__(pass_number=17, pass_name="Road Networks")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add road network layers to napari viewer.
        
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
        
        # =====================================================================
        # ROAD PRESENCE (All Roads)
        # =====================================================================
        
        road_presence = self._stitch_chunks(world_state, 'road_presence')
        if road_presence is not None:
            viewer.add_image(
                road_presence.astype(np.float32),
                name=self._layer_name("Road Network (All)"),
                colormap='gray',
                opacity=0.7,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Road Network (All)')}")
        
        # =====================================================================
        # ROAD TYPE MAP (Color-coded by road quality)
        # =====================================================================
        
        road_type_map = self._stitch_chunks(world_state, 'road_type')
        if road_type_map is not None:
            # Create custom colormap for road types
            # 0 = no road (black)
            # 1 = Imperial Highway (gold)
            # 2 = Main Road (orange)
            # 3 = Rural Road (yellow)
            # 4 = Path (light green)
            # 5 = Trail (pale green)
            
            viewer.add_labels(
                road_type_map,
                name=self._layer_name("Road Types"),
                opacity=0.8,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Road Types')}")
        
        # =====================================================================
        # INDIVIDUAL ROAD LAYERS BY TYPE
        # =====================================================================
        
        if road_type_map is not None:
            road_type_names = {
                1: "Imperial Highways",
                2: "Main Roads",
                3: "Rural Roads",
                4: "Paths",
                5: "Trails",
            }
            
            road_type_colors = {
                1: [1, 0.84, 0, 1],      # Gold
                2: [1, 0.5, 0, 1],       # Orange
                3: [1, 1, 0, 1],         # Yellow
                4: [0.5, 1, 0.5, 1],     # Light green
                5: [0.7, 1, 0.7, 1],     # Pale green
            }
            
            for road_value, road_name in road_type_names.items():
                road_mask = (road_type_map == road_value)
                
                if np.any(road_mask):
                    viewer.add_image(
                        road_mask.astype(np.float32),
                        name=self._layer_name(f"🛣 {road_name}"),
                        colormap='gray',
                        opacity=0.8,
                        visible=False,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'🛣 {road_name}')}")
        
        # =====================================================================
        # BRIDGE POINTS
        # =====================================================================
        
        bridges = self._collect_bridges(world_state)
        
        if bridges and len(bridges) > 0:
            coords = np.array([[b['x'], b['y']] for b in bridges])
            sizes = np.array([max(8, b['length']) for b in bridges])
            
            # Color by road type
            colors = []
            road_type_colors_rgb = {
                0: [1, 0.84, 0],      # Gold
                1: [1, 0.5, 0],       # Orange
                2: [1, 1, 0],         # Yellow
                3: [0.5, 1, 0.5],     # Light green
                4: [0.7, 1, 0.7],     # Pale green
            }
            
            for b in bridges:
                colors.append(road_type_colors_rgb.get(b['road_type'], [1, 1, 1]))
            
            viewer.add_points(
                coords,
                name=self._layer_name("Bridges 🌉"),
                size=sizes,
                face_color=colors,
                edge_color='blue',
                edge_width=2,
                symbol='square',
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Bridges 🌉')} ({len(bridges)} bridges)")
        
        # =====================================================================
        # TRAVEL TIME MAP (Distance to Nearest Road)
        # =====================================================================
        
        if road_presence is not None:
            print(f"      Calculating travel time map...")
            
            # Distance transform: cells to nearest road
            distance_to_road = distance_transform_edt(~road_presence)
            
            # Convert to travel time (assume 5 km/h walking speed, 100m per cell)
            # time (hours) = distance (cells) * 0.1 km / 5 km/h
            travel_time_hours = distance_to_road * 0.02
            
            # Cap at reasonable maximum
            travel_time_hours = np.clip(travel_time_hours, 0, 24)
            
            viewer.add_image(
                travel_time_hours,
                name=self._layer_name("🔍 Travel Time to Road (hours)"),
                colormap='viridis',
                opacity=0.6,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('🔍 Travel Time to Road')}")
        
        # =====================================================================
        # ROAD DENSITY HEATMAP
        # =====================================================================
        
        if road_presence is not None:
            from scipy.ndimage import gaussian_filter
            
            # Create density map by smoothing road presence
            road_density = gaussian_filter(road_presence.astype(np.float32), sigma=20.0)
            
            viewer.add_image(
                road_density,
                name=self._layer_name("🔍 Road Density"),
                colormap='hot',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('🔍 Road Density')}")
        
        # =====================================================================
        # ACCESSIBILITY MAP (Combined metric)
        # =====================================================================
        
        if road_presence is not None and hasattr(world_state, 'settlements'):
            print(f"      Calculating accessibility map...")
            
            # Collect settlements
            settlements = []
            for chunk in world_state.chunks.values():
                if hasattr(chunk, 'settlements') and chunk.settlements:
                    for s in chunk.settlements:
                        if not s.is_ruin:
                            settlements.append((s.x, s.y, s.population))
            
            if settlements:
                # Create settlement influence map (weighted by population)
                size = world_state.size
                settlement_influence = np.zeros((size, size), dtype=np.float32)
                
                for x, y, pop in settlements:
                    # Add Gaussian influence around settlement
                    radius = max(10, int(np.log10(pop + 1) * 10))
                    
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                dist = np.sqrt(dx**2 + dy**2)
                                if dist < radius:
                                    influence = np.exp(-dist / (radius / 2)) * np.log10(pop + 1)
                                    settlement_influence[nx, ny] += influence
                
                # Normalize
                if settlement_influence.max() > 0:
                    settlement_influence /= settlement_influence.max()
                
                # Combine with road access
                road_access = np.exp(-distance_to_road / 50.0)
                accessibility = (settlement_influence + road_access) / 2.0
                
                viewer.add_image(
                    accessibility,
                    name=self._layer_name("🔍 Accessibility"),
                    colormap='RdYlGn',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Accessibility')}")
        
        # =====================================================================
        # ROAD NETWORK GRAPH (Vector layer)
        # =====================================================================
        
        if hasattr(world_state, 'road_network') and world_state.road_network:
            # Separate highways from other roads
            highways = [r for r in world_state.road_network if r.road_type <= 1]
            other_roads = [r for r in world_state.road_network if r.road_type > 1]
            
            # Convert highways to shapes
            if highways:
                highway_shapes = []
                for road in highways:
                    if len(road.path) > 1:
                        # Convert path to numpy array for shapes layer
                        path_array = np.array(road.path)
                        highway_shapes.append(path_array)
                
                if highway_shapes:
                    viewer.add_shapes(
                        highway_shapes,
                        shape_type='path',
                        edge_width=3,
                        edge_color='gold',
                        face_color=[0, 0, 0, 0],  # Transparent
                        name=self._layer_name("Highway Network (Vector)"),
                        visible=False,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name('Highway Network (Vector)')} ({len(highways)} segments)")
            
            # Convert other roads to shapes
            if other_roads:
                other_road_shapes = []
                for road in other_roads:
                    if len(road.path) > 1:
                        path_array = np.array(road.path)
                        other_road_shapes.append(path_array)
                
                if other_road_shapes:
                    viewer.add_shapes(
                        other_road_shapes,
                        shape_type='path',
                        edge_width=1,
                        edge_color='yellow',
                        face_color=[0, 0, 0, 0],  # Transparent
                        name=self._layer_name("Secondary Roads (Vector)"),
                        visible=False,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name('Secondary Roads (Vector)')} ({len(other_roads)} segments)")
        
        return added_count
    
    def _collect_bridges(self, world_state) -> Optional[list]:
        """Collect all bridges from chunks."""
        bridges = []
        
        for chunk in world_state.chunks.values():
            if hasattr(chunk, 'bridges') and chunk.bridges:
                for bridge in chunk.bridges:
                    bridges.append({
                        'x': bridge.x,
                        'y': bridge.y,
                        'length': bridge.length,
                        'road_type': bridge.road_type,
                    })
        
        return bridges if bridges else None