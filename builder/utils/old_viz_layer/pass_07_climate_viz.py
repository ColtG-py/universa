"""
World Builder - Pass 07: Climate Visualization
Visualizes temperature and precipitation patterns
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer


class Pass07ClimateVisualizer(BaseVisualizer):
    """
    Visualizer for climate patterns - temperature and precipitation.
    """
    
    def visualize_temperature(
        self,
        temperature: np.ndarray,
        filename: str = "pass_07_temperature.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize temperature distribution.
        
        Args:
            temperature: Temperature array in Celsius
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        temperature = self._rotate_for_display(temperature)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(temperature, cmap='RdYlBu_r', interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
        
        ax.set_title('Temperature Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_precipitation(
        self,
        precipitation: np.ndarray,
        filename: str = "pass_07_precipitation.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize precipitation patterns.
        
        Args:
            precipitation: Precipitation array in mm/year
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        precipitation = self._rotate_for_display(precipitation)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(precipitation, cmap='Blues', interpolation='bilinear')
        cbar = plt.colorbar(im, ax=ax, label='Precipitation (mm/year)', shrink=0.8)
        
        ax.set_title('Precipitation Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_combined(
        self,
        temperature: np.ndarray,
        precipitation: np.ndarray,
        filename: str = "pass_07_climate.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize temperature and precipitation side by side.
        
        Args:
            temperature: Temperature array in Celsius
            precipitation: Precipitation array in mm/year
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Rotate data for display
        temperature = self._rotate_for_display(temperature)
        precipitation = self._rotate_for_display(precipitation)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        
        # Temperature
        im1 = ax1.imshow(temperature, cmap='RdYlBu_r', interpolation='bilinear')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Temperature (°C)', shrink=0.8)
        ax1.set_title('Temperature', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Precipitation
        im2 = ax2.imshow(precipitation, cmap='Blues', interpolation='bilinear')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Precipitation (mm/year)', shrink=0.8)
        ax2.set_title('Precipitation', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle('Climate Patterns', fontsize=18, fontweight='bold')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        temperature_filename: str = "pass_07_temperature.png",
        precipitation_filename: str = "pass_07_precipitation.png",
        combined_filename: str = "pass_07_climate.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize climate data collected from world chunks.
        
        Args:
            world_state: WorldState object
            temperature_filename: Output filename for temperature
            precipitation_filename: Output filename for precipitation
            combined_filename: Output filename for combined view
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        temperature = np.zeros((size, size), dtype=np.float32)
        precipitation = np.zeros((size, size), dtype=np.uint16)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if chunk.temperature_c is not None:
                temperature[x_start:x_end, y_start:y_end] = chunk.temperature_c[:chunk_width, :chunk_height]
            
            if chunk.precipitation_mm is not None:
                precipitation[x_start:x_end, y_start:y_end] = chunk.precipitation_mm[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize_temperature(temperature, temperature_filename, dpi)
        self.visualize_precipitation(precipitation, precipitation_filename, dpi)
        self.visualize_combined(temperature, precipitation, combined_filename, dpi)