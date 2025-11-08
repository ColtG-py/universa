"""
World Builder - Pass 09: Groundwater Visualization
Visualizes water table depth and aquifer systems
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass09GroundwaterVisualizer(BaseVisualizer):
    """
    Visualizer for groundwater systems and water table.
    """
    
    def visualize(
        self,
        water_table_depth: np.ndarray,
        elevation: Optional[np.ndarray] = None,
        filename: str = "pass_09_groundwater.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize groundwater table depth.
        
        Args:
            water_table_depth: Array of water table depths in meters
            elevation: Optional elevation for context
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Show elevation as faint background if available
        if elevation is not None:
            land_mask = elevation > 0
            land_elevation = np.where(land_mask, elevation, np.nan)
            ax.imshow(land_elevation, cmap='terrain', interpolation='bilinear', alpha=0.2)
        
        # Visualize water table depth
        im = ax.imshow(water_table_depth, cmap='Blues_r', interpolation='bilinear', alpha=0.85)
        cbar = plt.colorbar(im, ax=ax, label='Water Table Depth (m)', shrink=0.8)
        
        ax.set_title('Groundwater Table', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_09_groundwater.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize groundwater data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        water_table_depth = np.zeros((size, size), dtype=np.float32)
        elevation = None
        
        sample_chunk = next(iter(world_state.chunks.values()))
        if sample_chunk.elevation is not None:
            elevation = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if chunk.water_table_depth is not None:
                water_table_depth[x_start:x_end, y_start:y_end] = chunk.water_table_depth[:chunk_width, :chunk_height]
            
            if elevation is not None and chunk.elevation is not None:
                elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(water_table_depth, elevation, filename, dpi)