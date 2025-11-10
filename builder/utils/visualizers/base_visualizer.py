"""
World Builder - Base Visualizer
Base class for all pass-specific visualizers
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

# Try to import PIL for additional export options
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class BaseVisualizer:
    """
    Base visualizer class with common functionality for all pass visualizers.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def _rotate_for_display(self, data: np.ndarray) -> np.ndarray:
        """
        Rotate data 90 degrees clockwise for proper map orientation.
        
        This ensures poles are at top/bottom instead of left/right.
        
        Args:
            data: 2D or 3D array to rotate
            
        Returns:
            Rotated array
        """
        # Rotate 90 degrees clockwise (k=-1)
        # This moves poles from left/right to top/bottom
        return np.rot90(data, k=-1)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 150
    ) -> None:
        """
        Save matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved visualization to {self.output_dir / filename}")
    
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
        # Rotate for display
        data = self._rotate_for_display(data)
        
        if not HAS_PIL:
            print("⚠ PIL not available, using matplotlib instead")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(data, cmap=colormap_name)
            ax.axis('off')
            self.save_figure(fig, filename)
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