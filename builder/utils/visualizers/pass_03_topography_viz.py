"""
World Builder - Pass 03: Topography Visualization
Visualizes elevation and terrain
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .base_visualizer import BaseVisualizer


class Pass03TopographyVisualizer(BaseVisualizer):
    """
    Visualizer for elevation and topography.
    """
    
    def visualize(
        self,
        elevation: np.ndarray,
        filename: str = "pass_03_topography.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize elevation with land/ocean distinction.
        
        Args:
            elevation: Elevation array in meters
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap for elevation
        # Ocean: blues, Land: greens to browns to white
        ocean_colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, 128))
        land_colors = plt.cm.terrain(np.linspace(0.3, 1.0, 128))
        
        # Combine colormaps
        all_colors = np.vstack((ocean_colors, land_colors))
        terrain_cmap = mcolors.LinearSegmentedColormap.from_list('terrain_ocean', all_colors)
        
        im = ax.imshow(elevation, cmap=terrain_cmap, interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
        
        ax.set_title('Elevation Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_03_topography.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize elevation data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Collect data from chunks
        size = world_state.size
        elevation = np.zeros((size, size), dtype=np.float32)
        
        from config import CHUNK_SIZE
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if chunk.elevation is not None:
                elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(elevation, filename, dpi)