"""
World Builder - Pass 18: Political Boundaries Napari Visualization
Interactive visualization of kingdoms, territories, borders, and contested zones.
"""

import numpy as np
from typing import Optional
import matplotlib.colors as mcolors

from .base_napari_visualizer import BaseNapariVisualizer, NAPARI_AVAILABLE


class Pass18PoliticsNapariVisualizer(BaseNapariVisualizer):
    """Napari visualizer for political boundaries."""
    
    def __init__(self):
        super().__init__(pass_number=18, pass_name="Political Boundaries")
    
    def add_layers(
        self,
        viewer,
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add political boundary layers to napari viewer.
        
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
        # FACTION TERRITORIES (Color-coded by faction)
        # =====================================================================
        
        faction_territory = self._stitch_chunks(world_state, 'faction_territory')
        if faction_territory is not None:
            # Use labels layer for distinct factions
            viewer.add_labels(
                faction_territory,
                name=self._layer_name("Faction Territories"),
                opacity=0.5,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('Faction Territories')}")
        
        # =====================================================================
        # BORDER TYPES (Natural vs Political)
        # =====================================================================
        
        border_type = self._stitch_chunks(world_state, 'border_type')
        if border_type is not None:
            from config import BorderType
            
            # Create separate layers for each border type
            political_borders = (border_type == BorderType.POLITICAL)
            river_borders = (border_type == BorderType.RIVER)
            mountain_borders = (border_type == BorderType.MOUNTAIN)
            
            if np.any(political_borders):
                viewer.add_image(
                    political_borders.astype(np.float32),
                    name=self._layer_name("Political Borders"),
                    colormap='gray',
                    opacity=0.8,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Political Borders')}")
            
            if np.any(river_borders):
                viewer.add_image(
                    river_borders.astype(np.float32),
                    name=self._layer_name("River Borders 🌊"),
                    colormap='blue',
                    opacity=0.8,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('River Borders')}")
            
            if np.any(mountain_borders):
                viewer.add_image(
                    mountain_borders.astype(np.float32),
                    name=self._layer_name("Mountain Borders ⛰️"),
                    colormap='gray',
                    opacity=0.8,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Mountain Borders')}")
            
            # Combined border visualization
            all_borders = (border_type > 0)
            if np.any(all_borders):
                viewer.add_image(
                    all_borders.astype(np.float32),
                    name=self._layer_name("All Borders"),
                    colormap='gray',
                    opacity=0.9,
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('All Borders')}")
        
        # =====================================================================
        # CONTESTED ZONES
        # =====================================================================
        
        contested = self._stitch_chunks(world_state, 'contested_zone')
        if contested is not None and np.any(contested):
            viewer.add_image(
                contested.astype(np.float32),
                name=self._layer_name("⚔️ Contested Zones"),
                colormap='red',
                opacity=0.6,
                visible=default_visible,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('⚔️ Contested Zones')}")
        
        # =====================================================================
        # FACTION CAPITALS (Point layer)
        # =====================================================================
        
        if hasattr(world_state, 'factions') and world_state.factions:
            from config import FactionType
            
            capitals = []
            capital_names = []
            capital_colors = []
            capital_sizes = []
            
            # Color scheme for faction types
            faction_colors = {
                FactionType.KINGDOM: [1, 0.84, 0],      # Gold
                FactionType.DUCHY: [0.8, 0.5, 0.2],     # Bronze
                FactionType.COUNTY: [0.6, 0.6, 0.6],    # Silver
                FactionType.FREE_CITY: [0, 1, 1],       # Cyan
                FactionType.TRIBAL: [0.6, 0.3, 0],      # Brown
                FactionType.THEOCRACY: [0.9, 0.9, 1],   # White
            }
            
            for faction in world_state.factions:
                capitals.append([faction.capital_x, faction.capital_y])
                capital_names.append(faction.name)
                capital_colors.append(faction_colors.get(faction.faction_type, [1, 1, 1]))
                
                # Size based on faction type
                if faction.faction_type == FactionType.KINGDOM:
                    capital_sizes.append(20)
                elif faction.faction_type == FactionType.DUCHY:
                    capital_sizes.append(15)
                else:
                    capital_sizes.append(10)
            
            if capitals:
                coords = np.array(capitals)
                
                viewer.add_points(
                    coords,
                    name=self._layer_name("👑 Faction Capitals"),
                    size=capital_sizes,
                    face_color=capital_colors,
                    edge_color='black',
                    edge_width=2,
                    symbol='star',
                    visible=default_visible,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('👑 Faction Capitals')} ({len(capitals)} capitals)")
        
        # =====================================================================
        # FACTION INFLUENCE HEATMAP
        # =====================================================================
        
        if faction_territory is not None and hasattr(world_state, 'factions'):
            print(f"      Calculating faction influence heatmap...")
            
            from scipy.ndimage import gaussian_filter
            
            influence = np.zeros_like(faction_territory, dtype=np.float32)
            
            # For each faction, create influence based on distance from capital
            for faction in world_state.factions:
                # Create distance field from capital
                from scipy.ndimage import distance_transform_edt
                
                capital_mask = np.zeros_like(faction_territory, dtype=bool)
                if (0 <= faction.capital_x < faction_territory.shape[0] and
                    0 <= faction.capital_y < faction_territory.shape[1]):
                    capital_mask[faction.capital_x, faction.capital_y] = True
                
                distance_from_capital = distance_transform_edt(~capital_mask)
                
                # Faction cells get influence based on distance from capital
                faction_mask = (faction_territory == faction.faction_id)
                
                # Exponential decay influence
                max_distance = distance_from_capital[faction_mask].max() if np.any(faction_mask) else 1
                if max_distance > 0:
                    faction_influence = np.exp(-distance_from_capital / (max_distance * 0.3))
                    influence[faction_mask] += faction_influence[faction_mask]
            
            # Normalize
            if influence.max() > 0:
                influence /= influence.max()
            
            # Smooth
            influence = gaussian_filter(influence, sigma=5.0)
            
            viewer.add_image(
                influence,
                name=self._layer_name("🔍 Faction Influence"),
                colormap='viridis',
                opacity=0.5,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('🔍 Faction Influence')}")
        
        # =====================================================================
        # INDIVIDUAL FACTION LAYERS (By Type)
        # =====================================================================
        
        if faction_territory is not None and hasattr(world_state, 'factions'):
            from config import FactionType
            
            # Group factions by type
            factions_by_type = {}
            for faction in world_state.factions:
                if faction.faction_type not in factions_by_type:
                    factions_by_type[faction.faction_type] = []
                factions_by_type[faction.faction_type].append(faction)
            
            # Create layer for each faction type
            faction_type_names = {
                FactionType.KINGDOM: "Kingdoms",
                FactionType.DUCHY: "Duchies",
                FactionType.COUNTY: "Counties",
                FactionType.FREE_CITY: "Free Cities",
                FactionType.TRIBAL: "Tribal Lands",
                FactionType.THEOCRACY: "Theocracies",
            }
            
            for faction_type, type_factions in factions_by_type.items():
                # Create mask for this faction type
                type_mask = np.zeros_like(faction_territory, dtype=bool)
                
                for faction in type_factions:
                    type_mask |= (faction_territory == faction.faction_id)
                
                if np.any(type_mask):
                    type_name = faction_type_names.get(faction_type, f"Type {faction_type}")
                    
                    viewer.add_image(
                        type_mask.astype(np.float32),
                        name=self._layer_name(f"🏰 {type_name}"),
                        colormap='gray',
                        opacity=0.6,
                        visible=False,
                    )
                    added_count += 1
                    print(f"   ✓ Added layer: {self._layer_name(f'🏰 {type_name}')} ({len(type_factions)} factions)")
        
        # =====================================================================
        # VASSALAGE NETWORK (Shapes/Lines)
        # =====================================================================
        
        if hasattr(world_state, 'factions') and world_state.factions:
            vassalage_lines = []
            
            for faction in world_state.factions:
                if faction.liege_faction_id is not None:
                    # Find liege faction
                    liege = next((f for f in world_state.factions if f.faction_id == faction.liege_faction_id), None)
                    
                    if liege:
                        # Create line from vassal capital to liege capital
                        line = np.array([
                            [faction.capital_x, faction.capital_y],
                            [liege.capital_x, liege.capital_y]
                        ])
                        vassalage_lines.append(line)
            
            if vassalage_lines:
                viewer.add_shapes(
                    vassalage_lines,
                    shape_type='line',
                    edge_width=2,
                    edge_color='yellow',
                    name=self._layer_name("Vassalage Bonds"),
                    visible=False,
                )
                added_count += 1
                print(f"   ✓ Added layer: {self._layer_name('Vassalage Bonds')} ({len(vassalage_lines)} bonds)")
        
        # =====================================================================
        # GEOPOLITICAL STABILITY MAP
        # =====================================================================
        
        if faction_territory is not None and contested is not None:
            print(f"      Calculating geopolitical stability...")
            
            from scipy.ndimage import distance_transform_edt, gaussian_filter
            
            # Stability factors:
            # 1. Distance from borders (higher = more stable)
            # 2. Not contested (stable)
            # 3. Natural borders (more stable)
            
            # Calculate distance from any border
            from scipy.ndimage import sobel
            edges_x = sobel(faction_territory.astype(float), axis=0)
            edges_y = sobel(faction_territory.astype(float), axis=1)
            border_mask = (np.abs(edges_x) + np.abs(edges_y)) > 0
            
            distance_from_border = distance_transform_edt(~border_mask)
            
            # Normalize to 0-1
            if distance_from_border.max() > 0:
                border_stability = np.clip(distance_from_border / 50.0, 0, 1)
            else:
                border_stability = np.ones_like(distance_from_border)
            
            # Contested zones are unstable
            contested_penalty = np.where(contested, 0.0, 1.0)
            
            # Natural borders are more stable
            natural_border_bonus = np.ones_like(faction_territory, dtype=np.float32)
            if border_type is not None:
                from config import BorderType
                natural_borders = (border_type == BorderType.RIVER) | (border_type == BorderType.MOUNTAIN)
                
                # Expand natural borders slightly
                from scipy.ndimage import binary_dilation
                natural_border_zones = binary_dilation(natural_borders, iterations=3)
                natural_border_bonus[natural_border_zones] = 1.3
            
            # Combine factors
            stability = border_stability * contested_penalty * natural_border_bonus
            
            # Only apply to faction territories
            stability[faction_territory == 0] = 0
            
            # Smooth
            stability = gaussian_filter(stability, sigma=3.0)
            
            # Normalize
            if stability.max() > 0:
                stability = stability / stability.max()
            
            viewer.add_image(
                stability,
                name=self._layer_name("🔍 Geopolitical Stability"),
                colormap='RdYlGn',  # Red (unstable) to Green (stable)
                opacity=0.6,
                visible=False,
            )
            added_count += 1
            print(f"   ✓ Added layer: {self._layer_name('🔍 Geopolitical Stability')}")
        
        return added_count