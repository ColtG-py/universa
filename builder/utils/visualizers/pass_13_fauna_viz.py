"""
World Builder - Pass 13 Fauna Visualization
Visualizes wildlife distribution, apex predator territories, and migration routes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from config import CHUNK_SIZE, FaunaCategory
from utils.visualizers.base_visualizer import BaseVisualizer


class Pass13FaunaVisualizer(BaseVisualizer):
    """Visualizer for Pass 13: Fauna Distribution"""
    
    def visualize_from_chunks(
        self,
        world_state,
        output_fauna_density: str,
        output_territories: str,
        output_migration: str,
        output_combined: str,
        dpi: int = 150
    ) -> None:
        """
        Generate fauna visualizations from world chunks.
        
        Args:
            world_state: WorldState object
            output_fauna_density: Filename for fauna density map
            output_territories: Filename for apex territories map
            output_migration: Filename for migration routes
            output_combined: Filename for combined visualization
            dpi: Resolution
        """
        size = world_state.size
        num_chunks = size // CHUNK_SIZE
        
        print(f"    - Assembling fauna data from {num_chunks}x{num_chunks} chunks...")
        
        # Collect data from chunks
        # We'll visualize a few key fauna categories
        herbivore_grazer = np.zeros((size, size), dtype=np.float32)
        herbivore_browser = np.zeros((size, size), dtype=np.float32)
        predator_apex = np.zeros((size, size), dtype=np.float32)
        predator_medium = np.zeros((size, size), dtype=np.float32)
        aquatic_fish = np.zeros((size, size), dtype=np.float32)
        avian_migratory = np.zeros((size, size), dtype=np.float32)
        
        apex_territories = np.zeros((size, size), dtype=np.uint32)
        migration_routes = np.zeros((size, size), dtype=bool)
        elevation = np.zeros((size, size), dtype=np.float32)
        
        for chunk_y in range(num_chunks):
            for chunk_x in range(num_chunks):
                chunk = world_state.get_chunk(chunk_x, chunk_y)
                if chunk is None:
                    continue
                
                x_start = chunk_x * CHUNK_SIZE
                y_start = chunk_y * CHUNK_SIZE
                x_end = min(x_start + CHUNK_SIZE, size)
                y_end = min(y_start + CHUNK_SIZE, size)
                
                chunk_w = x_end - x_start
                chunk_h = y_end - y_start
                
                if chunk.fauna_density is not None:
                    if FaunaCategory.HERBIVORE_GRAZER in chunk.fauna_density:
                        herbivore_grazer[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.HERBIVORE_GRAZER][:chunk_w, :chunk_h]
                    
                    if FaunaCategory.HERBIVORE_BROWSER in chunk.fauna_density:
                        herbivore_browser[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.HERBIVORE_BROWSER][:chunk_w, :chunk_h]
                    
                    if FaunaCategory.PREDATOR_APEX in chunk.fauna_density:
                        predator_apex[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.PREDATOR_APEX][:chunk_w, :chunk_h]
                    
                    if FaunaCategory.PREDATOR_MEDIUM in chunk.fauna_density:
                        predator_medium[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.PREDATOR_MEDIUM][:chunk_w, :chunk_h]
                    
                    if FaunaCategory.AQUATIC_FISH in chunk.fauna_density:
                        aquatic_fish[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.AQUATIC_FISH][:chunk_w, :chunk_h]
                    
                    if FaunaCategory.AVIAN_MIGRATORY in chunk.fauna_density:
                        avian_migratory[x_start:x_end, y_start:y_end] = \
                            chunk.fauna_density[FaunaCategory.AVIAN_MIGRATORY][:chunk_w, :chunk_h]
                
                if chunk.apex_predator_territories is not None:
                    apex_territories[x_start:x_end, y_start:y_end] = \
                        chunk.apex_predator_territories[:chunk_w, :chunk_h]
                
                if chunk.migration_routes is not None:
                    migration_routes[x_start:x_end, y_start:y_end] = \
                        chunk.migration_routes[:chunk_w, :chunk_h]
                
                if chunk.elevation is not None:
                    elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_w, :chunk_h]
        
        # Rotate for display
        herbivore_grazer = self._rotate_for_display(herbivore_grazer)
        herbivore_browser = self._rotate_for_display(herbivore_browser)
        predator_apex = self._rotate_for_display(predator_apex)
        predator_medium = self._rotate_for_display(predator_medium)
        aquatic_fish = self._rotate_for_display(aquatic_fish)
        avian_migratory = self._rotate_for_display(avian_migratory)
        apex_territories = self._rotate_for_display(apex_territories)
        migration_routes = self._rotate_for_display(migration_routes)
        elevation = self._rotate_for_display(elevation)
        
        # Generate visualizations
        self._visualize_fauna_density(
            herbivore_grazer, herbivore_browser, predator_apex,
            predator_medium, aquatic_fish, avian_migratory,
            output_fauna_density, dpi
        )
        
        self._visualize_territories(
            apex_territories, elevation, output_territories, dpi
        )
        
        self._visualize_migration(
            migration_routes, elevation, output_migration, dpi
        )
        
        self._visualize_combined(
            herbivore_grazer, predator_apex, apex_territories,
            migration_routes, elevation, output_combined, dpi
        )
    
    def _visualize_fauna_density(
        self,
        herbivore_grazer: np.ndarray,
        herbivore_browser: np.ndarray,
        predator_apex: np.ndarray,
        predator_medium: np.ndarray,
        aquatic_fish: np.ndarray,
        avian_migratory: np.ndarray,
        filename: str,
        dpi: int
    ) -> None:
        """Visualize fauna density for major categories"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fauna Distribution by Category', fontsize=16, fontweight='bold')
        
        categories = [
            (herbivore_grazer, "Herbivores - Grazers", "Greens"),
            (herbivore_browser, "Herbivores - Browsers", "YlGn"),
            (predator_apex, "Apex Predators", "Reds"),
            (predator_medium, "Medium Predators", "Oranges"),
            (aquatic_fish, "Aquatic - Fish", "Blues"),
            (avian_migratory, "Migratory Birds", "Purples"),
        ]
        
        for idx, (data, title, cmap) in enumerate(categories):
            ax = axes[idx // 3, idx % 3]
            
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Population Density', fontsize=8)
        
        plt.tight_layout()
        self.save_figure(fig, filename, dpi)
    
    def _visualize_territories(
        self,
        territories: np.ndarray,
        elevation: np.ndarray,
        filename: str,
        dpi: int
    ) -> None:
        """Visualize apex predator territories"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('Apex Predator Territories', fontsize=14, fontweight='bold')
        
        # Create elevation base map
        land_mask = elevation > 0
        elevation_display = elevation.copy()
        elevation_display[~land_mask] = np.nan
        
        ax.imshow(elevation_display, cmap='terrain', alpha=0.3)
        
        # Overlay territories with distinct colors
        num_territories = int(territories.max())
        
        if num_territories > 0:
            # Create colormap with distinct colors
            import matplotlib.colors as mcolors
            colors = plt.cm.Set3(np.linspace(0, 1, num_territories + 1))
            colors[0] = [0, 0, 0, 0]  # Transparent for no territory
            cmap = mcolors.ListedColormap(colors)
            
            # Plot territories
            im = ax.imshow(territories, cmap=cmap, alpha=0.7, vmin=0, vmax=num_territories)
            
            # Add legend
            patches = [mpatches.Patch(color=colors[i], label=f'Territory {i}')
                      for i in range(1, min(num_territories + 1, 11))]  # Show up to 10
            if num_territories > 10:
                patches.append(mpatches.Patch(color='gray', label=f'... and {num_territories - 10} more'))
            
            ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)
            
            ax.set_title(f'{num_territories} Apex Predator Territories', fontweight='bold', pad=10)
        else:
            ax.set_title('No Apex Predator Territories', fontweight='bold', pad=10)
        
        ax.axis('off')
        
        plt.tight_layout()
        self.save_figure(fig, filename, dpi)
    
    def _visualize_migration(
        self,
        migration_routes: np.ndarray,
        elevation: np.ndarray,
        filename: str,
        dpi: int
    ) -> None:
        """Visualize herbivore migration routes"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('Herbivore Migration Routes', fontsize=14, fontweight='bold')
        
        # Create elevation base map
        land_mask = elevation > 0
        elevation_display = elevation.copy()
        elevation_display[~land_mask] = np.nan
        
        ax.imshow(elevation_display, cmap='terrain', alpha=0.5)
        
        # Overlay migration routes
        migration_display = np.zeros_like(migration_routes, dtype=float)
        migration_display[migration_routes] = 1.0
        migration_display[~migration_routes] = np.nan
        
        ax.imshow(migration_display, cmap='autumn', alpha=0.8)
        
        # Calculate coverage
        coverage = migration_routes.sum() / land_mask.sum() * 100 if land_mask.sum() > 0 else 0
        
        ax.set_title(f'Migration Corridors (Coverage: {coverage:.1f}% of land)', 
                    fontweight='bold', pad=10)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='orange', alpha=0.8, label='Migration Corridor'),
            mpatches.Patch(facecolor='gray', alpha=0.5, label='Land (no migration)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        self.save_figure(fig, filename, dpi)
    
    def _visualize_combined(
        self,
        herbivore_density: np.ndarray,
        predator_density: np.ndarray,
        territories: np.ndarray,
        migration_routes: np.ndarray,
        elevation: np.ndarray,
        filename: str,
        dpi: int
    ) -> None:
        """Create combined fauna overview visualization"""
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Fauna Distribution Overview', fontsize=16, fontweight='bold')
        
        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Total herbivore density
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(herbivore_density, cmap='YlGn', vmin=0, vmax=1)
        ax1.set_title('Herbivore Density', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Density')
        
        # 2. Predator density
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(predator_density, cmap='Reds', vmin=0, vmax=1)
        ax2.set_title('Apex Predator Density', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Density')
        
        # 3. Territories
        ax3 = fig.add_subplot(gs[1, 0])
        land_mask = elevation > 0
        elevation_display = elevation.copy()
        elevation_display[~land_mask] = np.nan
        ax3.imshow(elevation_display, cmap='terrain', alpha=0.3)
        
        num_territories = int(territories.max())
        if num_territories > 0:
            import matplotlib.colors as mcolors
            colors = plt.cm.Set3(np.linspace(0, 1, num_territories + 1))
            colors[0] = [0, 0, 0, 0]
            cmap = mcolors.ListedColormap(colors)
            ax3.imshow(territories, cmap=cmap, alpha=0.7, vmin=0, vmax=num_territories)
        
        ax3.set_title(f'Territories ({num_territories} total)', fontweight='bold')
        ax3.axis('off')
        
        # 4. Migration routes
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(elevation_display, cmap='terrain', alpha=0.5)
        
        migration_display = np.zeros_like(migration_routes, dtype=float)
        migration_display[migration_routes] = 1.0
        migration_display[~migration_routes] = np.nan
        ax4.imshow(migration_display, cmap='autumn', alpha=0.8)
        
        coverage = migration_routes.sum() / land_mask.sum() * 100 if land_mask.sum() > 0 else 0
        ax4.set_title(f'Migration Routes ({coverage:.1f}% coverage)', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        self.save_figure(fig, filename, dpi)