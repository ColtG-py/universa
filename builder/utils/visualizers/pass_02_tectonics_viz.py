"""
World Builder - Pass 02: Tectonics Visualization
Visualizes tectonic plates and stress patterns
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass02TectonicsVisualizer(BaseVisualizer):
    """
    Visualizer for tectonic plate generation and dynamics.
    """
    
    def visualize(
        self,
        plate_id: np.ndarray,
        tectonic_stress: Optional[np.ndarray] = None,
        filename: str = "pass_02_tectonics.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize tectonic plates and stress patterns.
        
        Args:
            plate_id: Array of plate IDs
            tectonic_stress: Optional stress values
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        if tectonic_stress is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 10))
        
        # Plot plate IDs
        im1 = ax1.imshow(plate_id, cmap='tab20', interpolation='nearest')
        ax1.set_title('Tectonic Plates', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Plot stress if available
        if tectonic_stress is not None:
            im2 = ax2.imshow(tectonic_stress, cmap='YlOrRd', interpolation='bilinear')
            cbar2 = plt.colorbar(im2, ax=ax2, label='Tectonic Stress', shrink=0.8)
            ax2.set_title('Tectonic Stress', fontsize=16, fontweight='bold')
            ax2.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_02_tectonics.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize tectonic data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Collect data from chunks
        size = world_state.size
        plate_id = np.zeros((size, size), dtype=np.uint8)
        tectonic_stress = None
        
        from config import CHUNK_SIZE
        
        sample_chunk = next(iter(world_state.chunks.values()))
        if sample_chunk.tectonic_stress is not None:
            tectonic_stress = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if chunk.plate_id is not None:
                plate_id[x_start:x_end, y_start:y_end] = chunk.plate_id[:chunk_width, :chunk_height]
            
            if tectonic_stress is not None and chunk.tectonic_stress is not None:
                tectonic_stress[x_start:x_end, y_start:y_end] = chunk.tectonic_stress[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(plate_id, tectonic_stress, filename, dpi)