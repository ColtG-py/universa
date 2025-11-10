"""
World Builder - Pass 04: Geology Visualization
Visualizes bedrock types and mineral deposits
"""

import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from .base_visualizer import BaseVisualizer


class Pass04GeologyVisualizer(BaseVisualizer):
    """
    Visualizer for geological features - bedrock and minerals.
    """
    
    def visualize_bedrock(
        self,
        bedrock_type: np.ndarray,
        filename: str = "pass_04_bedrock.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize bedrock types with labeled legend.
        
        Args:
            bedrock_type: Array of rock type IDs
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        from config import RockType
        
        # Rotate data for display
        bedrock_type = self._rotate_for_display(bedrock_type)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create custom discrete colormap for rock types
        rock_type_colors = {
            RockType.IGNEOUS: '#8B4513',      # Saddle Brown
            RockType.SEDIMENTARY: '#DEB887',  # Burlywood
            RockType.METAMORPHIC: '#696969',  # Dim Gray
            RockType.LIMESTONE: '#F5DEB3',    # Wheat
        }
        
        # Get unique rock types in the data
        unique_types = np.unique(bedrock_type)
        
        # Create color map
        colors = []
        labels = []
        for rock_type_id in unique_types:
            try:
                rock_type = RockType(rock_type_id)
                colors.append(rock_type_colors.get(rock_type, '#CCCCCC'))
                labels.append(rock_type.name.title())
            except ValueError:
                colors.append('#CCCCCC')
                labels.append(f'Unknown ({rock_type_id})')
        
        # Create discrete colormap
        cmap = mcolors.ListedColormap(colors)
        bounds = list(unique_types) + [unique_types[-1] + 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.imshow(bedrock_type, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create legend with labeled rock types
        legend_elements = []
        for i, (rock_type_id, label) in enumerate(zip(unique_types, labels)):
            legend_elements.append(
                Rectangle((0, 0), 1, 1, fc=colors[i], label=label)
            )
        
        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=11,
            title='Rock Types',
            title_fontsize=12
        )
        
        ax.set_title('Bedrock Geology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_minerals(
        self,
        mineral_richness: Dict[Any, np.ndarray],
        filename: str = "pass_04_minerals.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize mineral distribution across the world.
        
        Args:
            mineral_richness: Dictionary mapping Mineral enum to richness arrays
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        from config import Mineral
        
        # Filter out minerals with no deposits
        active_minerals = {}
        for mineral, richness in mineral_richness.items():
            if richness is not None and richness.max() > 0:
                active_minerals[mineral] = richness
        
        if not active_minerals:
            print("⚠ No mineral deposits to visualize")
            return
        
        n_minerals = len(active_minerals)
        
        # Calculate grid dimensions
        n_cols = min(3, n_minerals)
        n_rows = (n_minerals + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Flatten axes array for easier iteration
        if n_minerals == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        # Color maps for different minerals
        mineral_colormaps = {
            'IRON': 'Reds',
            'COPPER': 'copper',
            'GOLD': 'YlOrBr',
            'SILVER': 'Greys',
            'COAL': 'gray',
            'SALT': 'Blues',
            'DIAMOND': 'Blues',
            'EMERALD': 'Greens',
        }
        
        for idx, (mineral, richness) in enumerate(sorted(active_minerals.items(), 
                                                          key=lambda x: x[1].max(), 
                                                          reverse=True)):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Rotate data for display
            richness = self._rotate_for_display(richness)
            
            # Get mineral name
            try:
                mineral_name = mineral.name if hasattr(mineral, 'name') else str(mineral)
            except:
                mineral_name = str(mineral)
            
            # Select colormap
            cmap = mineral_colormaps.get(mineral_name, 'viridis')
            
            # Mask zero values
            masked_richness = np.ma.masked_where(richness <= 0.01, richness)
            
            im = ax.imshow(masked_richness, cmap=cmap, interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Richness', shrink=0.8)
            
            # Title with mineral name
            ax.set_title(f'{mineral_name.title()} Deposits', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_minerals, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Mineral Distribution', fontsize=18, fontweight='bold', y=0.995)
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        bedrock_filename: str = "pass_04_bedrock.png",
        minerals_filename: str = "pass_04_minerals.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize geology data collected from world chunks.
        
        Args:
            world_state: WorldState object
            bedrock_filename: Output filename for bedrock
            minerals_filename: Output filename for minerals
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE, Mineral
        
        size = world_state.size
        bedrock_type = np.zeros((size, size), dtype=np.uint8)
        
        # Initialize mineral richness dictionary
        sample_chunk = next(iter(world_state.chunks.values()))
        mineral_richness = None
        if sample_chunk.mineral_richness is not None:
            mineral_richness = {}
            for mineral in Mineral:
                mineral_richness[mineral] = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if chunk.bedrock_type is not None:
                bedrock_type[x_start:x_end, y_start:y_end] = chunk.bedrock_type[:chunk_width, :chunk_height]
            
            if mineral_richness is not None and chunk.mineral_richness is not None:
                for mineral in Mineral:
                    if mineral in chunk.mineral_richness:
                        mineral_richness[mineral][x_start:x_end, y_start:y_end] = \
                            chunk.mineral_richness[mineral][:chunk_width, :chunk_height]
        
        # Visualize bedrock
        self.visualize_bedrock(bedrock_type, bedrock_filename, dpi)
        
        # Visualize minerals
        if mineral_richness is not None:
            self.visualize_minerals(mineral_richness, minerals_filename, dpi)