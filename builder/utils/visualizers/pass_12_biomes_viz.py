"""
World Builder - Pass 12: Biomes Visualization
Visualizes biome classification, vegetation density, and agricultural suitability
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .base_visualizer import BaseVisualizer
from config import BiomeType


class Pass12BiomesVisualizer(BaseVisualizer):
    """
    Visualizer for biome classification and vegetation properties.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        super().__init__(output_dir)
        
        # Define biome colors (based on real-world biome colors)
        self.biome_colors = {
            BiomeType.OCEAN: '#0077be',
            BiomeType.ICE: '#ffffff',
            BiomeType.TUNDRA: '#c4d6d6',
            BiomeType.COLD_DESERT: '#e8d4b0',
            BiomeType.BOREAL_FOREST: '#2d5016',
            BiomeType.TEMPERATE_RAINFOREST: '#1a472a',
            BiomeType.TEMPERATE_DECIDUOUS_FOREST: '#5f8c4a',
            BiomeType.TEMPERATE_GRASSLAND: '#b8c776',
            BiomeType.MEDITERRANEAN: '#c9b070',
            BiomeType.HOT_DESERT: '#f4e7c7',
            BiomeType.SAVANNA: '#d4ba7c',
            BiomeType.TROPICAL_SEASONAL_FOREST: '#6ba048',
            BiomeType.TROPICAL_RAINFOREST: '#0f6e32',
        }
    
    def visualize_biomes(
        self,
        biome_map: np.ndarray,
        filename: str = "pass_12_biomes.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize biome classification with color-coded map.
        
        Args:
            biome_map: Array of biome type IDs
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create custom colormap
        biome_ids = sorted(set(biome_map.flatten()))
        colors = [self.biome_colors.get(BiomeType(bid), '#cccccc') for bid in biome_ids]
        
        # Create discrete colormap
        cmap = mcolors.ListedColormap(colors)
        bounds = [bid - 0.5 for bid in biome_ids] + [max(biome_ids) + 0.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.imshow(biome_map, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title('World Biomes (Whittaker Classification)', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Create legend
        legend_elements = []
        for bid in biome_ids:
            biome_name = BiomeType(bid).name.replace('_', ' ').title()
            color = self.biome_colors.get(BiomeType(bid), '#cccccc')
            legend_elements.append(Patch(facecolor=color, label=biome_name))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                 frameon=True, fontsize=10, title='Biome Types', title_fontsize=12)
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_vegetation_density(
        self,
        vegetation_density: np.ndarray,
        filename: str = "pass_12_vegetation_density.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize vegetation density.
        
        Args:
            vegetation_density: Array of density values (0-1)
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use greens colormap for vegetation
        im = ax.imshow(vegetation_density, cmap='YlGn', interpolation='bilinear', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, label='Vegetation Density', shrink=0.8)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Barren', 'Sparse', 'Moderate', 'Dense', 'Very Dense'])
        
        ax.set_title('Vegetation Density', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_canopy_height(
        self,
        canopy_height: np.ndarray,
        filename: str = "pass_12_canopy_height.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize forest canopy height.
        
        Args:
            canopy_height: Array of canopy heights in meters
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Mask zero values (non-forest areas)
        canopy_masked = np.ma.masked_where(canopy_height < 1.0, canopy_height)
        
        im = ax.imshow(canopy_masked, cmap='BuGn', interpolation='bilinear', vmin=0, vmax=50)
        cbar = plt.colorbar(im, ax=ax, label='Canopy Height (meters)', shrink=0.8)
        
        ax.set_title('Forest Canopy Height', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_agricultural_suitability(
        self,
        agricultural_suitability: np.ndarray,
        filename: str = "pass_12_agriculture.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize agricultural suitability.
        
        Args:
            agricultural_suitability: Array of suitability values (0-1)
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(agricultural_suitability, cmap='RdYlGn', interpolation='bilinear', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, label='Agricultural Suitability', shrink=0.8)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Unsuitable', 'Poor', 'Moderate', 'Good', 'Excellent'])
        
        ax.set_title('Agricultural Suitability', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_combined(
        self,
        biome_map: Optional[np.ndarray] = None,
        vegetation_density: Optional[np.ndarray] = None,
        canopy_height: Optional[np.ndarray] = None,
        agricultural_suitability: Optional[np.ndarray] = None,
        filename: str = "pass_12_biomes_combined.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize all biome properties in a 2x2 grid.
        
        Args:
            biome_map: Array of biome type IDs
            vegetation_density: Array of density values
            canopy_height: Array of canopy heights
            agricultural_suitability: Array of suitability values
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Count how many visualizations we have
        vis_list = [biome_map, vegetation_density, canopy_height, agricultural_suitability]
        n_plots = sum([v is not None for v in vis_list])
        
        if n_plots == 0:
            print("⚠ No biome data to visualize")
            return
        
        # Create figure with appropriate layout
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 10))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        elif n_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 18))
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Biome map
        if biome_map is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            biome_ids = sorted(set(biome_map.flatten()))
            colors = [self.biome_colors.get(BiomeType(bid), '#cccccc') for bid in biome_ids]
            cmap = mcolors.ListedColormap(colors)
            bounds = [bid - 0.5 for bid in biome_ids] + [max(biome_ids) + 0.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            
            im = ax.imshow(biome_map, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title('Biome Classification', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Vegetation density
        if vegetation_density is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            im = ax.imshow(vegetation_density, cmap='YlGn', interpolation='bilinear', vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Density', rotation=270, labelpad=15)
            ax.set_title('Vegetation Density', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Canopy height
        if canopy_height is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            canopy_masked = np.ma.masked_where(canopy_height < 1.0, canopy_height)
            im = ax.imshow(canopy_masked, cmap='BuGn', interpolation='bilinear', vmin=0, vmax=50)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Height (m)', rotation=270, labelpad=15)
            ax.set_title('Canopy Height', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Agricultural suitability
        if agricultural_suitability is not None:
            ax = axes[plot_idx]
            plot_idx += 1
            
            im = ax.imshow(agricultural_suitability, cmap='RdYlGn', interpolation='bilinear', vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Suitability', rotation=270, labelpad=15)
            ax.set_title('Agricultural Suitability', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Biomes & Vegetation (Pass 12)', fontsize=18, fontweight='bold', y=0.98)
        
        self.save_figure(fig, filename, dpi)
    
    def visualize_from_chunks(
        self,
        world_state,
        biome_filename: str = "pass_12_biomes.png",
        vegetation_filename: str = "pass_12_vegetation.png",
        canopy_filename: str = "pass_12_canopy.png",
        agriculture_filename: str = "pass_12_agriculture.png",
        combined_filename: str = "pass_12_biomes_combined.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize biome data collected from world chunks.
        
        Args:
            world_state: WorldState object
            biome_filename: Output filename for biome map
            vegetation_filename: Output filename for vegetation density
            canopy_filename: Output filename for canopy height
            agriculture_filename: Output filename for agricultural suitability
            combined_filename: Output filename for combined view
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        biome_map = None
        vegetation_density = None
        canopy_height = None
        agricultural_suitability = None
        
        # Check what data is available
        sample_chunk = next(iter(world_state.chunks.values()))
        if hasattr(sample_chunk, 'biome_type') and sample_chunk.biome_type is not None:
            biome_map = np.zeros((size, size), dtype=np.uint8)
        if hasattr(sample_chunk, 'vegetation_density') and sample_chunk.vegetation_density is not None:
            vegetation_density = np.zeros((size, size), dtype=np.float32)
        if hasattr(sample_chunk, 'forest_canopy_height') and sample_chunk.forest_canopy_height is not None:
            canopy_height = np.zeros((size, size), dtype=np.float32)
        if hasattr(sample_chunk, 'agricultural_suitability') and sample_chunk.agricultural_suitability is not None:
            agricultural_suitability = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            if biome_map is not None and hasattr(chunk, 'biome_type') and chunk.biome_type is not None:
                biome_map[x_start:x_end, y_start:y_end] = chunk.biome_type[:chunk_width, :chunk_height]
            
            if vegetation_density is not None and hasattr(chunk, 'vegetation_density') and chunk.vegetation_density is not None:
                vegetation_density[x_start:x_end, y_start:y_end] = chunk.vegetation_density[:chunk_width, :chunk_height]
            
            if canopy_height is not None and hasattr(chunk, 'forest_canopy_height') and chunk.forest_canopy_height is not None:
                canopy_height[x_start:x_end, y_start:y_end] = chunk.forest_canopy_height[:chunk_width, :chunk_height]
            
            if agricultural_suitability is not None and hasattr(chunk, 'agricultural_suitability') and chunk.agricultural_suitability is not None:
                agricultural_suitability[x_start:x_end, y_start:y_end] = chunk.agricultural_suitability[:chunk_width, :chunk_height]
        
        # Generate individual visualizations
        if biome_map is not None:
            self.visualize_biomes(biome_map, biome_filename, dpi)
        
        if vegetation_density is not None:
            self.visualize_vegetation_density(vegetation_density, vegetation_filename, dpi)
        
        if canopy_height is not None:
            self.visualize_canopy_height(canopy_height, canopy_filename, dpi)
        
        if agricultural_suitability is not None:
            self.visualize_agricultural_suitability(agricultural_suitability, agriculture_filename, dpi)
        
        # Generate combined view
        self.visualize_combined(
            biome_map,
            vegetation_density,
            canopy_height,
            agricultural_suitability,
            combined_filename,
            dpi
        )