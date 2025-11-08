"""
World Builder - Pass 10: Rivers Visualization
Visualizes river networks and drainage patterns
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .base_visualizer import BaseVisualizer


class Pass10RiversVisualizer(BaseVisualizer):
    """
    Visualizer for river networks and hydrological flow.
    """
    
    def visualize(
        self,
        river_presence: np.ndarray,
        river_flow: Optional[np.ndarray] = None,
        elevation: Optional[np.ndarray] = None,
        discharge: Optional[np.ndarray] = None,
        filename: str = "pass_10_rivers.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize river networks with flow information.
        
        Args:
            river_presence: Boolean array of river locations
            river_flow: Optional array of river flow rates
            elevation: Optional elevation for terrain context
            discharge: Optional discharge values for debugging
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Check if we have any rivers
        has_rivers = np.any(river_presence)
        
        if not has_rivers:
            print("⚠ No rivers to visualize")
            if discharge is not None:
                print(f"   Discharge stats: min={discharge.min():.4f}, max={discharge.max():.4f}, mean={discharge.mean():.4f}")
                print(f"   Non-zero discharge cells: {np.sum(discharge > 0)}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Show elevation as base layer if available
        if elevation is not None:
            land_mask = elevation > 0
            land_elevation = np.where(land_mask, elevation, np.nan)
            ax.imshow(land_elevation, cmap='terrain', interpolation='bilinear', alpha=0.4)
        
        # Overlay rivers
        if river_flow is not None and np.any(river_flow > 0):
            # Use flow magnitude to show river intensity
            river_flow_masked = np.ma.masked_where(river_flow <= 0, river_flow)
            
            # Create custom colormap: light cyan to deep blue
            colors_river = [
                (0.7, 1.0, 1.0),    # Very light cyan
                (0.0, 0.8, 1.0),    # Bright cyan
                (0.0, 0.5, 1.0),    # Medium blue
                (0.0, 0.2, 0.8),    # Deep blue
            ]
            n_bins = 256
            cmap_river = mcolors.LinearSegmentedColormap.from_list('rivers', colors_river, N=n_bins)
            
            im = ax.imshow(river_flow_masked, cmap=cmap_river, interpolation='nearest', alpha=0.95)
            cbar = plt.colorbar(im, ax=ax, label='River Flow (m³/s)', shrink=0.8, pad=0.02)
        else:
            # Just show river presence as bright cyan
            river_display = np.zeros((*river_presence.shape, 4))  # RGBA
            river_display[river_presence] = [0, 0.9, 1.0, 1.0]  # Bright cyan, fully opaque
            ax.imshow(river_display, interpolation='nearest')
        
        # Calculate and display statistics
        num_river_cells = np.sum(river_presence)
        
        if elevation is not None:
            land_cells = np.sum(elevation > 0)
            river_pct = (num_river_cells / land_cells * 100) if land_cells > 0 else 0
            title = f'River Networks\n{num_river_cells:,} cells ({river_pct:.2f}% of land)'
        else:
            title = f'River Networks\n{num_river_cells:,} cells'
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        self.save_figure(fig, filename, dpi)
        
        # Print detailed diagnostics
        print(f"   River cells: {num_river_cells:,}")
        if river_flow is not None:
            flow_values = river_flow[river_presence]
            if len(flow_values) > 0:
                print(f"   Flow range: {flow_values.min():.1f} - {flow_values.max():.1f} m³/s")
                print(f"   Mean flow: {flow_values.mean():.1f} m³/s")
                print(f"   Median flow: {np.median(flow_values):.1f} m³/s")
        
        if discharge is not None:
            discharge_values = discharge[river_presence]
            if len(discharge_values) > 0:
                print(f"   Discharge range: {discharge_values.min():.3f} - {discharge_values.max():.3f}")
                print(f"   Mean discharge: {discharge_values.mean():.3f}")
    
    def visualize_from_chunks(
        self,
        world_state,
        filename: str = "pass_10_rivers.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize river data collected from world chunks.
        
        Args:
            world_state: WorldState object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        from config import CHUNK_SIZE
        
        size = world_state.size
        river_presence = np.zeros((size, size), dtype=bool)
        river_flow = None
        discharge = None
        elevation = None
        
        sample_chunk = next(iter(world_state.chunks.values()))
        if hasattr(sample_chunk, 'river_flow') and sample_chunk.river_flow is not None:
            river_flow = np.zeros((size, size), dtype=np.float32)
        if hasattr(sample_chunk, 'discharge') and sample_chunk.discharge is not None:
            discharge = np.zeros((size, size), dtype=np.float32)
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
            
            if chunk.river_presence is not None:
                river_presence[x_start:x_end, y_start:y_end] = chunk.river_presence[:chunk_width, :chunk_height]
            
            if river_flow is not None and hasattr(chunk, 'river_flow') and chunk.river_flow is not None:
                river_flow[x_start:x_end, y_start:y_end] = chunk.river_flow[:chunk_width, :chunk_height]
            
            if discharge is not None and hasattr(chunk, 'discharge') and chunk.discharge is not None:
                discharge[x_start:x_end, y_start:y_end] = chunk.discharge[:chunk_width, :chunk_height]
            
            if elevation is not None and chunk.elevation is not None:
                elevation[x_start:x_end, y_start:y_end] = chunk.elevation[:chunk_width, :chunk_height]
        
        # Visualize
        self.visualize(river_presence, river_flow, elevation, discharge, filename, dpi)