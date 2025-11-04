"""
World Builder - Visualization Utilities
Provides visualization tools for all world generation layers
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

# Try to import PIL for additional export options
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class LayerVisualizer:
    """
    Visualizes different layers of world generation with appropriate colormaps.
    """
    
    # Define colormaps for each layer type
    LAYER_COLORMAPS = {
        'elevation': 'terrain',
        'temperature': 'RdYlBu_r',
        'precipitation': 'Blues',
        'tectonic_stress': 'YlOrRd',
        'plate_id': 'tab20',
        'bedrock_type': 'Set3',
        'soil_type': 'YlOrBr',
        'wind_speed': 'viridis',
        'wind_direction': 'twilight',
        'water_table_depth': 'Blues_r',
        'river_presence': 'Blues',
        'river_flow': 'Blues',
        'soil_ph': 'RdYlGn',
        'soil_drainage': 'BuGn',
        'microclimate_modifier': 'coolwarm',
        'cave_presence': 'binary',
        'mineral_richness': 'copper',
    }
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_elevation(
        self,
        elevation: np.ndarray,
        filename: str = "elevation.png",
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
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved elevation visualization to {self.output_dir / filename}")
    
    def visualize_temperature(
        self,
        temperature: np.ndarray,
        filename: str = "temperature.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize temperature distribution.
        
        Args:
            temperature: Temperature array in Celsius
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(temperature, cmap='RdYlBu_r', interpolation='bilinear')
        
        cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.8)
        
        ax.set_title('Temperature Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved temperature visualization to {self.output_dir / filename}")
    
    def visualize_precipitation(
        self,
        precipitation: np.ndarray,
        filename: str = "precipitation.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize precipitation patterns.
        
        Args:
            precipitation: Precipitation array in mm/year
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(precipitation, cmap='Blues', interpolation='bilinear')
        
        cbar = plt.colorbar(im, ax=ax, label='Precipitation (mm/year)', shrink=0.8)
        
        ax.set_title('Precipitation Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved precipitation visualization to {self.output_dir / filename}")
    
    def visualize_tectonic_plates(
        self,
        plate_id: np.ndarray,
        tectonic_stress: Optional[np.ndarray] = None,
        filename: str = "tectonic_plates.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize tectonic plates and stress.
        
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
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved tectonic visualization to {self.output_dir / filename}")
    
    def visualize_geology(
        self,
        bedrock_type: np.ndarray,
        filename: str = "geology.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize bedrock types.
        
        Args:
            bedrock_type: Array of rock type IDs
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(bedrock_type, cmap='Set3', interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax, label='Rock Type', shrink=0.8)
        
        ax.set_title('Bedrock Geology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved geology visualization to {self.output_dir / filename}")
    
    def visualize_hydrology(
        self,
        river_presence: Optional[np.ndarray] = None,
        water_table_depth: Optional[np.ndarray] = None,
        filename: str = "hydrology.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize hydrological features.
        
        Args:
            river_presence: Boolean array of river locations
            water_table_depth: Array of water table depths
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        n_plots = sum([river_presence is not None, water_table_depth is not None])
        
        if n_plots == 0:
            print("⚠ No hydrology data to visualize")
            return
        
        if n_plots == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 10))
        
        plot_idx = 0
        
        if river_presence is not None:
            ax = ax1 if n_plots == 2 else ax1
            im1 = ax.imshow(river_presence, cmap='Blues', interpolation='nearest')
            ax.set_title('River Networks', fontsize=16, fontweight='bold')
            ax.axis('off')
            plot_idx += 1
        
        if water_table_depth is not None:
            ax = ax2 if n_plots == 2 else ax1
            im2 = ax.imshow(water_table_depth, cmap='Blues_r', interpolation='bilinear')
            cbar2 = plt.colorbar(im2, ax=ax, label='Water Table Depth (m)', shrink=0.8)
            ax.set_title('Groundwater Table', fontsize=16, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved hydrology visualization to {self.output_dir / filename}")
    
    def visualize_soil(
        self,
        soil_type: Optional[np.ndarray] = None,
        soil_ph: Optional[np.ndarray] = None,
        filename: str = "soil.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize soil properties.
        
        Args:
            soil_type: Array of soil type IDs
            soil_ph: Array of pH values
            filename: Output filename
            dpi: Resolution in dots per inch
        """
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
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved soil visualization to {self.output_dir / filename}")
    
    def visualize_all_layers(
        self,
        world_state,
        prefix: str = "world",
        dpi: int = 150
    ) -> None:
        """
        Generate visualizations for all available layers in world state.
        
        Args:
            world_state: WorldState object with generated data
            prefix: Prefix for output filenames
            dpi: Resolution in dots per inch
        """
        print("\n" + "="*60)
        print("GENERATING LAYER VISUALIZATIONS")
        print("="*60 + "\n")
        
        # Collect data from all chunks
        chunk_data = self._collect_chunk_data(world_state)
        
        # Visualize each available layer
        if chunk_data['elevation'] is not None:
            self.visualize_elevation(
                chunk_data['elevation'],
                f"{prefix}_elevation.png",
                dpi
            )
        
        if chunk_data['temperature_c'] is not None:
            self.visualize_temperature(
                chunk_data['temperature_c'],
                f"{prefix}_temperature.png",
                dpi
            )
        
        if chunk_data['precipitation_mm'] is not None:
            self.visualize_precipitation(
                chunk_data['precipitation_mm'],
                f"{prefix}_precipitation.png",
                dpi
            )
        
        if chunk_data['plate_id'] is not None:
            self.visualize_tectonic_plates(
                chunk_data['plate_id'],
                chunk_data['tectonic_stress'],
                f"{prefix}_tectonics.png",
                dpi
            )
        
        if chunk_data['bedrock_type'] is not None:
            self.visualize_geology(
                chunk_data['bedrock_type'],
                f"{prefix}_geology.png",
                dpi
            )
        
        self.visualize_hydrology(
            chunk_data['river_presence'],
            chunk_data['water_table_depth'],
            f"{prefix}_hydrology.png",
            dpi
        )
        
        self.visualize_soil(
            chunk_data['soil_type'],
            chunk_data['soil_ph'],
            f"{prefix}_soil.png",
            dpi
        )
        
        print("\n" + "="*60)
        print(f"✓ All visualizations saved to {self.output_dir}/")
        print("="*60 + "\n")
    
    def _collect_chunk_data(self, world_state) -> Dict[str, Optional[np.ndarray]]:
        """
        Collect and stitch together data from all chunks.
        
        Args:
            world_state: WorldState object
        
        Returns:
            Dictionary of full-world arrays for each layer
        """
        size = world_state.size
        
        # Initialize full-world arrays
        data = {
            'elevation': None,
            'temperature_c': None,
            'precipitation_mm': None,
            'plate_id': None,
            'tectonic_stress': None,
            'bedrock_type': None,
            'river_presence': None,
            'water_table_depth': None,
            'soil_type': None,
            'soil_ph': None,
        }
        
        # Check if we have any chunks
        if not world_state.chunks:
            return data
        
        # Get a sample chunk to see what data is available
        sample_chunk = next(iter(world_state.chunks.values()))
        
        # Initialize arrays for available layers
        from config import CHUNK_SIZE
        
        if sample_chunk.elevation is not None:
            data['elevation'] = np.zeros((size, size), dtype=np.float32)
        if sample_chunk.temperature_c is not None:
            data['temperature_c'] = np.zeros((size, size), dtype=np.float32)
        if sample_chunk.precipitation_mm is not None:
            data['precipitation_mm'] = np.zeros((size, size), dtype=np.uint16)
        if sample_chunk.plate_id is not None:
            data['plate_id'] = np.zeros((size, size), dtype=np.uint8)
        if sample_chunk.tectonic_stress is not None:
            data['tectonic_stress'] = np.zeros((size, size), dtype=np.float32)
        if sample_chunk.bedrock_type is not None:
            data['bedrock_type'] = np.zeros((size, size), dtype=np.uint8)
        if sample_chunk.river_presence is not None:
            data['river_presence'] = np.zeros((size, size), dtype=bool)
        if sample_chunk.water_table_depth is not None:
            data['water_table_depth'] = np.zeros((size, size), dtype=np.float32)
        if sample_chunk.soil_type is not None:
            data['soil_type'] = np.zeros((size, size), dtype=np.uint8)
        if sample_chunk.soil_ph is not None:
            data['soil_ph'] = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            for layer_name, array in data.items():
                if array is not None:
                    chunk_data = getattr(chunk, layer_name)
                    if chunk_data is not None:
                        array[x_start:x_end, y_start:y_end] = chunk_data[:chunk_width, :chunk_height]
        
        return data
    
    def export_to_pil(
        self,
        data: np.ndarray,
        colormap_name: str = 'viridis',
        filename: str = "export.png"
    ) -> None:
        """
        Export data as PIL image with colormap applied.
        
        Args:
            data: 2D numpy array to export
            colormap_name: Name of matplotlib colormap
            filename: Output filename
        """
        if not HAS_PIL:
            print("⚠ PIL not available, using matplotlib instead")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(data, cmap=colormap_name)
            ax.axis('off')
            plt.savefig(self.output_dir / filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            return
        
        # Normalize data to 0-255
        data_norm = data.copy()
        data_min = data_norm.min()
        data_max = data_norm.max()
        
        if data_max > data_min:
            data_norm = ((data_norm - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data_norm = np.zeros_like(data_norm, dtype=np.uint8)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap_name)
        colored = cmap(data_norm / 255.0)
        
        # Convert to RGB (remove alpha channel)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(rgb)
        img.save(self.output_dir / filename)
        
        print(f"✓ Saved PIL image to {self.output_dir / filename}")


def create_visualization_summary(
    world_state,
    output_dir: str = "visualizations",
    dpi: int = 150
) -> None:
    """
    Convenience function to generate all visualizations.
    
    Args:
        world_state: WorldState object
        output_dir: Directory for outputs
        dpi: Resolution
    """
    visualizer = LayerVisualizer(output_dir)
    visualizer.visualize_all_layers(world_state, prefix="world", dpi=dpi)