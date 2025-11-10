"""
World Builder - Pass 11: Soil Visualization
Visualizes soil types, pH, and drainage properties
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass11SoilVisualizer(BaseVisualizer):
    """
    Visualizer for soil properties and characteristics.
    """
    
    def visualize_soil_type(
        self,
        soil_type: np.ndarray,
        filename: str = "pass_11_soil_type.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize soil type distribution.
        
        Args:
            soil_type: Array of soil type IDs
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        soil_type = self._rotate_for_display(soil_type)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(soil_type, cmap='YlOrBr', interpolation='nearest')
        ax.set_title('Soil Types', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_soil_ph(
        self,
        soil_ph: np.ndarray,
        filename: str = "pass_11_soil_ph.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize soil pH distribution.
        
        Args:
            soil_ph: Array of pH values
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        soil_ph = self._rotate_for_display(soil_ph)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(soil_ph, cmap='RdYlGn', interpolation='bilinear', vmin=4, vmax=10)
        cbar = plt.colorbar(im, ax=ax, label='Soil pH', shrink=0.8)
        ax.set_title('Soil pH', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_combined(
        self,
        soil_type: Optional[np.ndarray] = None,
        soil_ph: Optional[np.ndarray] = None,
        filename: str = "pass_11_soil.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize soil properties side by side.
        
        Args:
            soil_type: Array of soil type IDs
            soil_ph: Array of pH values
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        if soil_type is not None:
            soil_type = self._rotate_for_display(soil_type)
        if soil_ph is not None:
            soil_ph = self._rotate_for_display(soil_ph)
        
        n_plots = sum([soil_type is not None, soil_ph is not None])
        
        if n_plots == 0:
            print("⚠ No soil data to visualize")
            return
        
        if n_plots == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 10))
        
        if soil_type is not None:
            ax = ax1 if n_plots == 2 else ax1
            im1 = ax.imshow(soil_type, cmap='YlOrBr', interpolation='nearest')
            ax.set_title('Soil Types', fontsize=16, fontweight='bold')
            ax.axis('off')
        
        if soil_ph is not None:
            ax = ax2 if n_plots == 2 else ax1
            im2 = ax.imshow(soil_ph, cmap='RdYlGn', interpolation='bilinear', vmin=4, vmax=10)
            cbar2 = plt.colorbar(im2, ax=ax, label='Soil pH', shrink=0.8)
            ax.set_title('Soil pH', fontsize=16, fontweight='bold')
            ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        type_filename: str = "pass_11_soil_type.png",
        ph_filename: str = "pass_11_soil_ph.png",
        combined_filename: str = "pass_11_soil.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize soil data collected from world chunks.
        
        Args:
            world_state: WorldState object
            type_filename: Output filename for soil type
            ph_filename: Output filename for soil pH
            combined_filename: Output filename for combined view
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        soil_type = None
        soil_ph = None
        
        sample_chunk = next(iter(world_state.chunks.values()))
        if sample_chunk.soil_type is not None:
            soil_type = np.zeros((size, size), dtype=np.uint8)
        if sample_chunk.soil_ph is not None:
            soil_ph = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if soil_type is not None and chunk.soil_type is not None:
                soil_type[x_start:x_end, y_start:y_end] = chunk.soil_type[:chunk_width, :chunk_height]
            
            if soil_ph is not None and chunk.soil_ph is not None:
                soil_ph[x_start:x_end, y_start:y_end] = chunk.soil_ph[:chunk_width, :chunk_height]
        
        # Visualize
        if soil_type is not None:
            self.visualize_soil_type(soil_type, type_filename, dpi)
        if soil_ph is not None:
            self.visualize_soil_ph(soil_ph, ph_filename, dpi)
        
        self.visualize_combined(soil_type, soil_ph, combined_filename, dpi)