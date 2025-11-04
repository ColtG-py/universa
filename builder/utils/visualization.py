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
from matplotlib.patches import Rectangle

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
        Visualize bedrock types with labeled legend.
        
        Args:
            bedrock_type: Array of rock type IDs
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        # Import RockType enum
        from config import RockType
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create custom discrete colormap for rock types
        rock_type_colors = {
            RockType.IGNEOUS: '#8B4513',      # Saddle Brown
            RockType.SEDIMENTARY: '#DEB887',  # Burlywood
            RockType.METAMORPHIC: '#696969',  # Dim Gray
            RockType.LIMESTONE: '#F5DEB3',    # Wheat
        }
        
        # Get unique rock types in the data
        unique_types = np.unique(bedrock_type)
        
        # Create color map
        colors = []
        labels = []
        for rock_type_id in unique_types:
            try:
                rock_type = RockType(rock_type_id)
                colors.append(rock_type_colors.get(rock_type, '#CCCCCC'))
                labels.append(rock_type.name.title())
            except ValueError:
                colors.append('#CCCCCC')
                labels.append(f'Unknown ({rock_type_id})')
        
        # Create discrete colormap
        cmap = mcolors.ListedColormap(colors)
        bounds = list(unique_types) + [unique_types[-1] + 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.imshow(bedrock_type, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create legend with labeled rock types
        legend_elements = []
        for i, (rock_type_id, label) in enumerate(zip(unique_types, labels)):
            legend_elements.append(
                Rectangle((0, 0), 1, 1, fc=colors[i], label=label)
            )
        
        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=11,
            title='Rock Types',
            title_fontsize=12
        )
        
        ax.set_title('Bedrock Geology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved geology visualization to {self.output_dir / filename}")
    
    def visualize_minerals(
        self,
        mineral_richness: Dict[Any, np.ndarray],
        filename: str = "minerals.png",
        dpi: int = 150
    ) -> None:
        """
        Visualize mineral distribution across the world.
        
        Args:
            mineral_richness: Dictionary mapping Mineral enum to richness arrays
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        from config import Mineral
        
        # Filter out minerals with no deposits
        active_minerals = {}
        for mineral, richness in mineral_richness.items():
            if richness is not None and richness.max() > 0:
                active_minerals[mineral] = richness
        
        if not active_minerals:
            print("⚠ No mineral deposits to visualize")
            return
        
        n_minerals = len(active_minerals)
        
        # Calculate grid dimensions
        n_cols = min(3, n_minerals)
        n_rows = (n_minerals + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Flatten axes array for easier iteration
        if n_minerals == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        # Color maps for different minerals
        mineral_colormaps = {
            'IRON': 'Reds',
            'COPPER': 'copper',
            'GOLD': 'YlOrBr',
            'SILVER': 'Greys',
            'COAL': 'gray',
            'URANIUM': 'Greens',
            'DIAMOND': 'Blues',
            'EMERALD': 'Greens',
            'RUBY': 'Reds',
            'SAPPHIRE': 'Blues',
        }
        
        for idx, (mineral, richness) in enumerate(sorted(active_minerals.items(), 
                                                          key=lambda x: x[1].max(), 
                                                          reverse=True)):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Get mineral name
            try:
                mineral_name = mineral.name if hasattr(mineral, 'name') else str(mineral)
            except:
                mineral_name = str(mineral)
            
            # Select colormap
            cmap = mineral_colormaps.get(mineral_name, 'viridis')
            
            # Plot with transparency for zero values
            # Mask zero values
            masked_richness = np.ma.masked_where(richness <= 0.01, richness)
            
            im = ax.imshow(masked_richness, cmap=cmap, interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Richness', shrink=0.8)
            
            # Title with mineral name
            ax.set_title(f'{mineral_name.title()} Deposits', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_minerals, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Mineral Distribution', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved mineral visualization to {self.output_dir / filename}")
    
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
    
    def visualize_wind_patterns(
        self,
        wind_speed: np.ndarray,
        wind_direction: np.ndarray,
        elevation: Optional[np.ndarray] = None,
        filename: str = "wind_patterns.png",
        dpi: int = 150,
        subsample: int = 16
    ) -> None:
        """
        Visualize wind patterns using vector field (quiver plot).
        
        Args:
            wind_speed: Array of wind speeds in m/s
            wind_direction: Array of wind directions in degrees
            elevation: Optional elevation for context
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Show every Nth vector (reduces density)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Show elevation as background if available
        if elevation is not None:
            # Create masked elevation (transparent ocean)
            land_mask = elevation > 0
            land_elevation = np.where(land_mask, elevation, np.nan)
            
            ax.imshow(land_elevation, cmap='terrain', interpolation='bilinear', alpha=0.3)
        
        # Subsample for cleaner visualization
        size = wind_speed.shape[0]
        x_indices = np.arange(0, size, subsample)
        y_indices = np.arange(0, size, subsample)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Convert wind direction to components
        wind_dir_rad = np.deg2rad(wind_direction)
        wind_u = wind_speed * np.cos(wind_dir_rad)
        wind_v = wind_speed * np.sin(wind_dir_rad)
        
        # Subsample wind vectors
        U = wind_u[::subsample, ::subsample]
        V = wind_v[::subsample, ::subsample]
        
        # Plot wind vectors
        speed_subsampled = wind_speed[::subsample, ::subsample]
        quiver = ax.quiver(
            X, Y, U, V,
            speed_subsampled,
            cmap='cool',
            scale=100,
            width=0.003,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
        
        ax.set_title('Atmospheric Wind Patterns', fontsize=16, fontweight='bold')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved wind pattern visualization to {self.output_dir / filename}")
    
    def visualize_ocean_currents(
        self,
        ocean_current_speed: np.ndarray,
        ocean_current_direction: np.ndarray,
        elevation: np.ndarray,
        filename: str = "ocean_currents.png",
        dpi: int = 150,
        subsample: int = 16
    ) -> None:
        """
        Visualize ocean currents using vector field (quiver plot).
        
        Args:
            ocean_current_speed: Array of current speeds in m/s
            ocean_current_direction: Array of current directions in degrees
            elevation: Elevation array (for masking land)
            filename: Output filename
            dpi: Resolution in dots per inch
            subsample: Show every Nth vector (reduces density)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create ocean mask
        ocean_mask = elevation < 0
        
        # Show ocean depth as background
        ocean_depth = np.where(ocean_mask, -elevation, np.nan)
        
        im = ax.imshow(ocean_depth, cmap='Blues', interpolation='bilinear', alpha=0.4)
        plt.colorbar(im, ax=ax, label='Ocean Depth (m)', shrink=0.8)
        
        # Subsample for cleaner visualization
        size = ocean_current_speed.shape[0]
        x_indices = np.arange(0, size, subsample)
        y_indices = np.arange(0, size, subsample)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Convert current direction to components
        current_dir_rad = np.deg2rad(ocean_current_direction)
        current_u = ocean_current_speed * np.cos(current_dir_rad)
        current_v = ocean_current_speed * np.sin(current_dir_rad)
        
        # Mask out land areas
        current_u_masked = np.where(ocean_mask, current_u, 0)
        current_v_masked = np.where(ocean_mask, current_v, 0)
        
        # Subsample current vectors
        U = current_u_masked[::subsample, ::subsample]
        V = current_v_masked[::subsample, ::subsample]
        speed_subsampled = ocean_current_speed[::subsample, ::subsample]
        
        # Only plot where there's significant current
        threshold = 0.01
        significant = speed_subsampled > threshold
        
        # Plot ocean current vectors
        quiver = ax.quiver(
            X[significant], Y[significant],
            U[significant], V[significant],
            speed_subsampled[significant],
            cmap='YlOrRd',
            scale=30,
            width=0.004,
            alpha=0.9
        )
        
        # Add colorbar for current speed
        cbar2 = plt.colorbar(quiver, ax=ax, label='Current Speed (m/s)', 
                            shrink=0.8, pad=0.1)
        
        ax.set_title('Ocean Surface Currents', fontsize=16, fontweight='bold')
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved ocean current visualization to {self.output_dir / filename}")
    
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
        
        if chunk_data['mineral_richness'] is not None:
            self.visualize_minerals(
                chunk_data['mineral_richness'],
                f"{prefix}_minerals.png",
                dpi
            )
        
        if chunk_data['wind_speed'] is not None and chunk_data['wind_direction'] is not None:
            self.visualize_wind_patterns(
                chunk_data['wind_speed'],
                chunk_data['wind_direction'],
                chunk_data['elevation'],
                f"{prefix}_wind.png",
                dpi
            )
        
        if (chunk_data['ocean_current_speed'] is not None and 
            chunk_data['ocean_current_direction'] is not None and
            chunk_data['elevation'] is not None):
            self.visualize_ocean_currents(
                chunk_data['ocean_current_speed'],
                chunk_data['ocean_current_direction'],
                chunk_data['elevation'],
                f"{prefix}_ocean_currents.png",
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
            'mineral_richness': None,
            'wind_speed': None,
            'wind_direction': None,
            'ocean_current_speed': None,
            'ocean_current_direction': None,
        }
        
        # Check if we have any chunks
        if not world_state.chunks:
            return data
        
        # Get a sample chunk to see what data is available
        sample_chunk = next(iter(world_state.chunks.values()))
        
        # Initialize arrays for available layers
        from config import CHUNK_SIZE, Mineral
        
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
        if sample_chunk.wind_speed is not None:
            data['wind_speed'] = np.zeros((size, size), dtype=np.float32)
        if sample_chunk.wind_direction is not None:
            data['wind_direction'] = np.zeros((size, size), dtype=np.float32)
        if hasattr(sample_chunk, 'ocean_current_speed') and sample_chunk.ocean_current_speed is not None:
            data['ocean_current_speed'] = np.zeros((size, size), dtype=np.float32)
        if hasattr(sample_chunk, 'ocean_current_direction') and sample_chunk.ocean_current_direction is not None:
            data['ocean_current_direction'] = np.zeros((size, size), dtype=np.float32)
        
        # Initialize mineral richness dictionary
        if sample_chunk.mineral_richness is not None:
            data['mineral_richness'] = {}
            for mineral in Mineral:
                data['mineral_richness'][mineral] = np.zeros((size, size), dtype=np.float32)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            for layer_name, array in data.items():
                if layer_name == 'mineral_richness':
                    # Handle mineral richness dictionary separately
                    if array is not None and chunk.mineral_richness is not None:
                        for mineral in Mineral:
                            if mineral in chunk.mineral_richness:
                                array[mineral][x_start:x_end, y_start:y_end] = \
                                    chunk.mineral_richness[mineral][:chunk_width, :chunk_height]
                elif array is not None:
                    chunk_data = getattr(chunk, layer_name, None)
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