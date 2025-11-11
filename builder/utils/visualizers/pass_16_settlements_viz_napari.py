"""
World Builder - Pass 16: Settlement Sites Napari Visualization (ENHANCED)
Interactive visualization with debug layers for suitability analysis
"""

import numpy as np
from typing import Optional

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Pass16SettlementsNapariVisualizer(BaseNapariVisualizer):
    """Enhanced napari visualizer for settlement sites with debug layers."""
    
    def __init__(self):
        super().__init__(pass_number=16, pass_name="Settlement Sites")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add settlement layers and debug suitability layers to napari viewer.
        
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
        # DEBUG LAYERS - Suitability Scores
        # =====================================================================
        
        if hasattr(world_state, 'settlement_debug_data') and world_state.settlement_debug_data:
            debug_data = world_state.settlement_debug_data
            
            # Total Suitability Score
            if 'total' in debug_data:
                viewer.add_image(
                    debug_data['total'],
                    name=self._layer_name("🔍 Suitability: Total"),
                    colormap='viridis',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Total')}")
            
            # Water Access Score
            if 'water' in debug_data:
                viewer.add_image(
                    debug_data['water'],
                    name=self._layer_name("🔍 Suitability: Water"),
                    colormap='Blues',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Water')}")
            
            # Defense Score
            if 'defense' in debug_data:
                viewer.add_image(
                    debug_data['defense'],
                    name=self._layer_name("🔍 Suitability: Defense"),
                    colormap='YlOrRd',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Defense')}")
            
            # Resource Score
            if 'resource' in debug_data:
                viewer.add_image(
                    debug_data['resource'],
                    name=self._layer_name("🔍 Suitability: Resources"),
                    colormap='YlGn',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Resources')}")
            
            # Climate Score
            if 'climate' in debug_data:
                viewer.add_image(
                    debug_data['climate'],
                    name=self._layer_name("🔍 Suitability: Climate"),
                    colormap='RdYlBu_r',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Climate')}")
            
            # Accessibility Score
            if 'access' in debug_data:
                viewer.add_image(
                    debug_data['access'],
                    name=self._layer_name("🔍 Suitability: Accessibility"),
                    colormap='Purples',
                    opacity=0.6,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Suitability: Accessibility')}")
        
        # =====================================================================
        # SETTLEMENT MAP (Rasterized)
        # =====================================================================
        
        settlement_map = self._stitch_chunks(world_state, 'settlement_presence')
        if settlement_map is not None:
            # Create custom colormap for settlement types
            colors = {
                0: [0, 0, 0, 0],        # Empty (transparent)
                1: [0.5, 1, 0.5, 0.5],  # Hamlet (light green)
                2: [1, 1, 0, 0.6],      # Village (yellow)
                3: [1, 0.5, 0, 0.7],    # Town (orange)
                4: [1, 0, 0, 0.8],      # City (red)
                5: [1, 0.84, 0, 0.9],   # Metropolis (gold)
                6: [0.5, 0, 0, 0.8],    # Fortress (dark red)
                7: [0.53, 0.81, 0.98, 0.7],  # Monastery (light blue)
                8: [0.5, 0.5, 0.5, 0.6],     # Ruin (gray)
            }
            
            viewer.add_labels(
                settlement_map,
                name=self._layer_name("Settlement Map (Raster)"),
                opacity=0.7,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Settlement Map (Raster)')}")
        
        # =====================================================================
        # SETTLEMENT POINTS (Primary visualization)
        # =====================================================================
        
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
            
            # Add cities (separate capitals from regular)
            if cities:
                capitals = [s for s in cities if s['is_capital']]
                regular_cities = [s for s in cities if not s['is_capital']]
                
                if capitals:
                    cap_coords = np.array([[s['x'], s['y']] for s in capitals])
                    cap_sizes = np.array([self._pop_to_size(s['population']) for s in capitals])
                    viewer.add_points(
                        cap_coords,
                        name=self._layer_name("Capital Cities ★"),
                        size=cap_sizes,
                        face_color='purple',
                        edge_color='white',
                        edge_width=3,
                        visible=default_visible,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name('Capital Cities ★')} ({len(capitals)} points)")
                
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
                    name=self._layer_name("Fortresses ⚔"),
                    size=12,
                    face_color='darkred',
                    edge_color='black',
                    edge_width=2,
                    symbol='square',
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Fortresses ⚔')} ({len(fortresses)} points)")
            
            # Add monasteries
            if monasteries:
                coords = np.array([[s['x'], s['y']] for s in monasteries])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Monasteries ✟"),
                    size=10,
                    face_color='lightblue',
                    edge_color='white',
                    edge_width=2,
                    symbol='cross',
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Monasteries ✟')} ({len(monasteries)} points)")
            
            # Add ruins
            if ruins:
                coords = np.array([[s['x'], s['y']] for s in ruins])
                viewer.add_points(
                    coords,
                    name=self._layer_name("Ruins ☠"),
                    size=8,
                    face_color='gray',
                    edge_color='black',
                    edge_width=1,
                    symbol='x',
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Ruins ☠')} ({len(ruins)} points)")
            
            # =====================================================================
            # SPECIALIZATION LAYERS (Alternative view)
            # =====================================================================
            
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
                0: '#90EE90',  # Light green
                1: '#8B4513',  # Brown
                2: '#4682B4',  # Steel blue
                3: '#FFD700',  # Gold
                4: '#DC143C',  # Crimson
                5: '#87CEEB',  # Sky blue
                6: '#9370DB',  # Purple
                7: '#FF8C00',  # Dark orange
            }
            
            for spec_id, spec_name in spec_names.items():
                spec_settlements = [s for s in settlements if s['specialization'] == spec_id and not s['is_ruin']]
                
                if spec_settlements:
                    coords = np.array([[s['x'], s['y']] for s in spec_settlements])
                    sizes = np.array([self._pop_to_size(s['population']) for s in spec_settlements])
                    viewer.add_points(
                        coords,
                        name=self._layer_name(f"Specialization: {spec_name}"),
                        size=sizes,
                        face_color=spec_colors[spec_id],
                        edge_color='white',
                        edge_width=1,
                        visible=False,  # Hidden by default
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'Specialization: {spec_name}')} ({len(spec_settlements)} points)")
            
            # =====================================================================
            # SETTLEMENT DENSITY HEATMAP
            # =====================================================================
            
            density_map = self._create_density_heatmap(settlements, world_state.size)
            if density_map is not None:
                viewer.add_image(
                    density_map,
                    name=self._layer_name("🔍 Settlement Density"),
                    colormap='hot',
                    opacity=0.5,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('🔍 Settlement Density')}")
        
        return added_count
    
    def _collect_settlements(self, world_state) -> Optional[list]:
        """Collect all settlements from chunks."""
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
        import math
        if population <= 0:
            return 5
        
        # Logarithmic scaling: 8-24 pixels
        log_pop = math.log10(population)
        min_log = math.log10(20)       # Hamlet
        max_log = math.log10(100000)   # Metropolis
        
        normalized = (log_pop - min_log) / (max_log - min_log)
        normalized = np.clip(normalized, 0, 1)
        
        size = 8 + normalized * 16  # 8 to 24 pixels
        return size
    
    def _create_density_heatmap(self, settlements: list, size: int) -> Optional[np.ndarray]:
        """
        Create a density heatmap showing settlement concentration.
        Uses Gaussian kernel for smooth density estimation.
        """
        if not settlements:
            return None
        
        from scipy.ndimage import gaussian_filter
        
        # Create accumulation map
        density = np.zeros((size, size), dtype=np.float32)
        
        for s in settlements:
            if not s['is_ruin']:
                # Weight by population (log scale)
                weight = np.log10(s['population'] + 1)
                density[s['x'], s['y']] += weight
        
        # Smooth with Gaussian filter
        density = gaussian_filter(density, sigma=20.0)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        return density