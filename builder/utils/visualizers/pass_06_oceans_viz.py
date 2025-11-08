"""
World Builder - Pass 06: Oceans Visualization
Visualizes ocean currents and marine dynamics
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass06OceansVisualizer(BaseVisualizer):
    """
    Visualizer for ocean currents and circulation patterns.
    """
    
    def visualize(
        self,
        ocean_current_speed: np.ndarray,
        ocean_current_direction: np.ndarray,
        elevation: np.ndarray,
        filename: str = "pass_06_oceans.png",
        dpi: int = 150,
        subsample: int = 1
    ) -> None:
        """
        Visualize ocean currents using streamlines.
        
        Args:
            ocean_current_speed: Array of current speeds in m/s
            ocean_current_direction: Array of current directions in degrees
            elevation: Elevation array (for masking land)
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Downsampling factor for performance
        """
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Create ocean mask
        ocean_mask = elevation < 0
        
        # Show ocean depth as background
        ocean_depth = np.where(ocean_mask, -elevation, np.nan)
        im = ax.imshow(ocean_depth, cmap='Blues', interpolation='bilinear', alpha=0.3, origin='lower')
        
        size = ocean_current_speed.shape[0]
        
        # Optionally subsample for performance
        if subsample > 1:
            ocean_current_speed = ocean_current_speed[::subsample, ::subsample]
            ocean_current_direction = ocean_current_direction[::subsample, ::subsample]
            ocean_mask = ocean_mask[::subsample, ::subsample]
            size = ocean_current_speed.shape[0]
        
        # Create coordinate grid
        Y, X = np.mgrid[0:size, 0:size]
        
        # Convert current direction to components
        current_dir_rad = np.deg2rad(ocean_current_direction)
        current_u = ocean_current_speed * np.cos(current_dir_rad)
        current_v = ocean_current_speed * np.sin(current_dir_rad)
        
        # Mask out land areas
        current_u = np.where(ocean_mask, current_u, 0)
        current_v = np.where(ocean_mask, current_v, 0)
        current_speed_masked = np.where(ocean_mask, ocean_current_speed, 0)
        
        # Create streamplot for ocean currents
        stream = ax.streamplot(
            X, Y, current_u, current_v,
            color=current_speed_masked,
            cmap='YlOrRd',
            density=[2.0, 1.8],
            linewidth=2.0,
            arrowsize=1.5,
            arrowstyle='->',
            minlength=0.1,
            integration_direction='both'
        )
        
        # Add colorbar for current speed
        cbar = plt.colorbar(stream.lines, ax=ax, label='Current Speed (m/s)', 
                           shrink=0.75, pad=0.02, aspect=30)
        
        ax.set_title('Ocean Surface Currents', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_06_oceans.png",
        dpi: int = 150,
        subsample: int = 1
    ) -> None:
        """
        Visualize ocean current data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Downsampling factor for performance
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        ocean_current_speed = np.zeros((size, size), dtype=np.float32)
        ocean_current_direction = np.zeros((size, size), dtype=np.float32)
        elevation = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if hasattr(chunk, 'ocean_current_speed') and chunk.ocean_current_speed is not None:
                ocean_current_speed[x_start:x_end, y_start:y_end] = chunk.ocean_current_speed[:chunk_width, :chunk_height]
            
            if hasattr(chunk, 'ocean_current_direction') and chunk.ocean_current_direction is not None:
                ocean_current_direction[x_start:x_end, y_start:y_end] = chunk.ocean_current_direction[:chunk_width, :chunk_height]
            
            if chunk.elevation is not None:
                elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(ocean_current_speed, ocean_current_direction, elevation, filename, dpi, subsample)