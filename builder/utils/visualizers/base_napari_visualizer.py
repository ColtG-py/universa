"""
World Builder - Base Napari Visualizer
Base class for all pass-specific Napari visualizers with common utilities
"""

import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

from config import CHUNK_SIZE


class BaseNapariVisualizer:
    """
    Base visualizer class with common functionality for Napari-based visualizers.
    
    Provides chunk stitching, layer naming conventions, and screenshot utilities.
    """
    
    def __init__(self, pass_number: int, pass_name: str):
        """
        Initialize base Napari visualizer.
        
        Args:
            pass_number: Generation pass number (1-20)
            pass_name: Human-readable pass name (e.g., "Topography")
        """
        self.pass_number = pass_number
        self.pass_name = pass_name
        
        if not NAPARI_AVAILABLE:
            print(f"⚠️  napari not installed for {pass_name} visualization")
    
    def _layer_name(self, layer_type: str) -> str:
        """
        Generate standardized layer name.
        
        Args:
            layer_type: Type of layer (e.g., "Elevation", "Temperature")
        
        Returns:
            Formatted layer name: "Pass 03: Elevation"
        """
        return f"Pass {self.pass_number:02d}: {layer_type}"
    
    def _stitch_chunks(
        self, 
        world_state, 
        attribute: str
    ) -> Optional[np.ndarray]:
        """
        Stitch chunk data together into a single array.
        
        Args:
            world_state: WorldState object
            attribute: Name of the chunk attribute to stitch
        
        Returns:
            Stitched numpy array or None if attribute not available
        """
        # Check if any chunks have this attribute
        if not world_state.chunks:
            return None
        
        sample_chunk = next(iter(world_state.chunks.values()))
        if not hasattr(sample_chunk, attribute):
            return None
        
        sample_data = getattr(sample_chunk, attribute)
        if sample_data is None:
            return None
        
        # Initialize output array
        size = world_state.size
        dtype = sample_data.dtype
        stitched = np.zeros((size, size), dtype=dtype)
        
        # Stitch chunks together
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            data = getattr(chunk, attribute, None)
            if data is None:
                continue
            
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            stitched[x_start:x_end, y_start:y_end] = data[:chunk_width, :chunk_height]
        
        return stitched
    
    def _stitch_dict_attribute(
        self, 
        world_state, 
        attribute: str, 
        key: Any
    ) -> Optional[np.ndarray]:
        """
        Stitch dictionary-based chunk data (e.g., mineral_richness).
        
        Args:
            world_state: WorldState object
            attribute: Name of the dictionary attribute
            key: Key within the dictionary
        
        Returns:
            Stitched numpy array or None if not available
        """
        if not world_state.chunks:
            return None
        
        sample_chunk = next(iter(world_state.chunks.values()))
        attr_dict = getattr(sample_chunk, attribute, None)
        
        if attr_dict is None or key not in attr_dict:
            return None
        
        size = world_state.size
        dtype = attr_dict[key].dtype
        stitched = np.zeros((size, size), dtype=dtype)
        
        for (chunk_x, chunk_y), chunk in world_state.chunks.items():
            attr_dict = getattr(chunk, attribute, None)
            if attr_dict is None or key not in attr_dict:
                continue
            
            data = attr_dict[key]
            x_start = chunk_x * CHUNK_SIZE
            y_start = chunk_y * CHUNK_SIZE
            x_end = min(x_start + CHUNK_SIZE, size)
            y_end = min(y_start + CHUNK_SIZE, size)
            
            chunk_width = x_end - x_start
            chunk_height = y_end - y_start
            
            stitched[x_start:x_end, y_start:y_end] = data[:chunk_width, :chunk_height]
        
        return stitched
    
    def add_layers(
        self,
        viewer: 'napari.Viewer',
        world_state,
        default_visible: bool = False
    ) -> int:
        """
        Add this pass's layers to the napari viewer.
        
        Must be implemented by subclasses.
        
        Args:
            viewer: Napari viewer instance
            world_state: WorldState object
            default_visible: Whether layers should be visible by default
        
        Returns:
            Number of layers added
        """
        raise NotImplementedError("Subclasses must implement add_layers()")
    
    def save_screenshot(
        self,
        viewer: 'napari.Viewer',
        output_path: Path,
        layers_to_show: Optional[List[str]] = None
    ) -> None:
        """
        Save a screenshot of the napari viewer with specific layers visible.
        
        Args:
            viewer: Napari viewer instance
            output_path: Path to save screenshot
            layers_to_show: List of layer names to make visible (None = current state)
        """
        if not NAPARI_AVAILABLE:
            print(f"⚠️  Cannot save screenshot - napari not available")
            return
        
        # Store original visibility
        original_visibility = {layer.name: layer.visible for layer in viewer.layers}
        
        try:
            # Set visibility if specified
            if layers_to_show is not None:
                for layer in viewer.layers:
                    layer.visible = layer.name in layers_to_show
            
            # Take screenshot
            screenshot = viewer.screenshot(canvas_only=True, flash=False)
            
            # Save using PIL if available
            try:
                from PIL import Image
                img = Image.fromarray(screenshot)
                img.save(output_path)
                print(f"✓ Saved screenshot to {output_path}")
            except ImportError:
                # Fallback to numpy
                np.save(output_path.with_suffix('.npy'), screenshot)
                print(f"✓ Saved screenshot array to {output_path.with_suffix('.npy')}")
        
        finally:
            # Restore original visibility
            for layer in viewer.layers:
                if layer.name in original_visibility:
                    layer.visible = original_visibility[layer.name]