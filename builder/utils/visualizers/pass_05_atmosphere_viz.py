"""
World Builder - Pass 05: Atmosphere Visualization
Visualizes wind patterns and atmospheric circulation
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass05AtmosphereVisualizer(BaseVisualizer):
    """
    Visualizer for atmospheric dynamics and wind patterns.
    """
    
    def visualize(
        self,
        wind_speed: np.ndarray,
        wind_direction: np.ndarray,
        elevation: Optional[np.ndarray] = None,
        filename: str = "pass_05_atmosphere.png",
        dpi: int = 150,
        subsample: int = 1
    ) -> None:
        """
        Visualize wind patterns using streamlines.
        
        Args:
            wind_speed: Array of wind speeds in m/s
            wind_direction: Array of wind directions in degrees
            elevation: Optional elevation for context
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Downsampling factor for performance
        """
        fig, ax = plt.subplots(figsize=(18, 14))
        
        size = wind_speed.shape[0]
        
        # Show elevation as background if available
        if elevation is not None:
            land_mask = elevation > 0
            land_elevation = np.where(land_mask, elevation, np.nan)
            ax.imshow(land_elevation, cmap='terrain', interpolation='bilinear', alpha=0.2, origin='lower')
        
        # Optionally subsample for performance
        if subsample > 1:
            wind_speed = wind_speed[::subsample, ::subsample]
            wind_direction = wind_direction[::subsample, ::subsample]
            size = wind_speed.shape[0]
        
        # Create coordinate grid
        Y, X = np.mgrid[0:size, 0:size]
        
        # Convert wind direction to U,V components
        wind_dir_rad = np.deg2rad(wind_direction)
        wind_u = wind_speed * np.cos(wind_dir_rad)
        wind_v = wind_speed * np.sin(wind_dir_rad)
        
        # Create streamplot with curved flow lines
        stream = ax.streamplot(
            X, Y, wind_u, wind_v,
            color=wind_speed,
            cmap='cool',
            density=[2.5, 2.0],
            linewidth=2.0,
            arrowsize=1.5,
            arrowstyle='->',
            minlength=0.1,
            integration_direction='both'
        )
        
        # Add colorbar
        cbar = plt.colorbar(stream.lines, ax=ax, label='Wind Speed (m/s)', 
                           shrink=0.75, pad=0.02, aspect=30)
        
        ax.set_title('Atmospheric Wind Patterns', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_05_atmosphere.png",
        dpi: int = 150,
        subsample: int = 1
    ) -> None:
        """
        Visualize atmospheric data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Downsampling factor for performance
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        wind_speed = np.zeros((size, size), dtype=np.float32)
        wind_direction = np.zeros((size, size), dtype=np.float32)
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
            
            if chunk.wind_speed is not None:
                wind_speed[x_start:x_end, y_start:y_end] = chunk.wind_speed[:chunk_width, :chunk_height]
            
            if chunk.wind_direction is not None:
                wind_direction[x_start:x_end, y_start:y_end] = chunk.wind_direction[:chunk_width, :chunk_height]
            
            if elevation is not None and chunk.elevation is not None:
                elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(wind_speed, wind_direction, elevation, filename, dpi, subsample)