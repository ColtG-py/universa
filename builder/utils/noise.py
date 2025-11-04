"""
World Builder - Noise Generation Utilities
Provides deterministic noise generation for procedural world generation
"""

import numpy as np
from typing import Optional
from noise import pnoise2, snoise2


class NoiseGenerator:
    """
    Deterministic noise generator using Perlin and Simplex noise.
    All noise is seeded to ensure reproducible generation.
    """
    
    def __init__(
        self,
        seed: int,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 100.0
    ):
        """
        Initialize noise generator with parameters.
        
        Args:
            seed: Random seed for deterministic generation
            octaves: Number of noise layers to combine
            persistence: Amplitude multiplier per octave (0-1)
            lacunarity: Frequency multiplier per octave (>1)
            scale: Overall scale of noise features
        """
        self.seed = seed
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.scale = scale
    
    def generate_perlin_2d(
        self,
        width: int,
        height: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate 2D Perlin noise array.
        
        Args:
            width: Width of output array
            height: Height of output array
            offset_x: X offset for seamless tiling
            offset_y: Y offset for seamless tiling
            normalize: If True, normalize output to [0, 1]
        
        Returns:
            2D numpy array of noise values
        """
        noise_array = np.zeros((width, height), dtype=np.float32)
        
        for x in range(width):
            for y in range(height):
                nx = (x + offset_x) / self.scale
                ny = (y + offset_y) / self.scale
                
                noise_array[x, y] = pnoise2(
                    nx,
                    ny,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=width * 2,
                    repeaty=height * 2,
                    base=self.seed
                )
        
        if normalize:
            # Perlin noise typically ranges from -1 to 1
            # Normalize to 0 to 1
            noise_array = (noise_array + 1.0) / 2.0
        
        return noise_array
    
    def generate_simplex_2d(
        self,
        width: int,
        height: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate 2D Simplex noise array.
        Simplex noise is faster than Perlin and has better visual properties.
        
        Args:
            width: Width of output array
            height: Height of output array
            offset_x: X offset for seamless tiling
            offset_y: Y offset for seamless tiling
            normalize: If True, normalize output to [0, 1]
        
        Returns:
            2D numpy array of noise values
        """
        noise_array = np.zeros((width, height), dtype=np.float32)
        
        for x in range(width):
            for y in range(height):
                nx = (x + offset_x) / self.scale
                ny = (y + offset_y) / self.scale
                
                # Simplex noise with multiple octaves
                value = 0.0
                amplitude = 1.0
                frequency = 1.0
                max_value = 0.0
                
                for octave in range(self.octaves):
                    sample_x = nx * frequency
                    sample_y = ny * frequency
                    
                    noise_val = snoise2(
                        sample_x,
                        sample_y,
                        base=self.seed + octave
                    )
                    
                    value += noise_val * amplitude
                    max_value += amplitude
                    
                    amplitude *= self.persistence
                    frequency *= self.lacunarity
                
                noise_array[x, y] = value / max_value
        
        if normalize:
            # Normalize to 0 to 1
            noise_array = (noise_array + 1.0) / 2.0
        
        return noise_array
    
    def generate_ridged_noise(
        self,
        width: int,
        height: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> np.ndarray:
        """
        Generate ridged multi-fractal noise.
        Useful for mountain ranges and sharp terrain features.
        
        Returns:
            2D numpy array normalized to [0, 1]
        """
        noise = self.generate_perlin_2d(
            width, height, offset_x, offset_y, normalize=False
        )
        
        # Create ridges by taking absolute value and inverting
        ridged = 1.0 - np.abs(noise)
        
        # Square to sharpen ridges
        ridged = ridged ** 2
        
        return ridged
    
    def generate_billow_noise(
        self,
        width: int,
        height: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> np.ndarray:
        """
        Generate billowy noise useful for clouds and rolling hills.
        
        Returns:
            2D numpy array normalized to [0, 1]
        """
        noise = self.generate_perlin_2d(
            width, height, offset_x, offset_y, normalize=False
        )
        
        # Take absolute value to create billows
        billow = np.abs(noise)
        
        # Normalize
        billow = (billow + 1.0) / 2.0
        
        return billow
    
    def generate_domain_warped_noise(
        self,
        width: int,
        height: int,
        warp_strength: float = 10.0
    ) -> np.ndarray:
        """
        Generate domain-warped noise for more organic patterns.
        
        Args:
            width: Width of output array
            height: Height of output array
            warp_strength: How much to warp the domain
        
        Returns:
            2D numpy array normalized to [0, 1]
        """
        # Generate two noise fields for warping
        warp_x = self.generate_perlin_2d(width, height, 0, 0, normalize=False)
        warp_y = self.generate_perlin_2d(width, height, 100, 100, normalize=False)
        
        # Generate base noise
        result = np.zeros((width, height), dtype=np.float32)
        
        for x in range(width):
            for y in range(height):
                # Warp the sampling coordinates
                warped_x = x + warp_x[x, y] * warp_strength
                warped_y = y + warp_y[x, y] * warp_strength
                
                # Sample from warped coordinates
                wx = int(np.clip(warped_x, 0, width - 1))
                wy = int(np.clip(warped_y, 0, height - 1))
                
                result[x, y] = pnoise2(
                    wx / self.scale,
                    wy / self.scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    base=self.seed + 1000
                )
        
        # Normalize
        result = (result + 1.0) / 2.0
        
        return result


def combine_noise_layers(layers: list, weights: Optional[list] = None) -> np.ndarray:
    """
    Combine multiple noise layers with optional weights.
    
    Args:
        layers: List of numpy arrays (must all be same shape)
        weights: Optional list of weights for each layer
    
    Returns:
        Combined noise array normalized to [0, 1]
    """
    if not layers:
        raise ValueError("Must provide at least one layer")
    
    if weights is None:
        weights = [1.0] * len(layers)
    
    if len(layers) != len(weights):
        raise ValueError("Number of layers and weights must match")
    
    # Ensure all layers are same shape
    shape = layers[0].shape
    for layer in layers[1:]:
        if layer.shape != shape:
            raise ValueError("All layers must have same shape")
    
    # Weighted combination
    result = np.zeros(shape, dtype=np.float32)
    total_weight = sum(weights)
    
    for layer, weight in zip(layers, weights):
        result += layer * (weight / total_weight)
    
    return result


def apply_curve(noise: np.ndarray, curve_func) -> np.ndarray:
    """
    Apply a curve function to noise values.
    
    Args:
        noise: Input noise array
        curve_func: Function to apply to each value
    
    Returns:
        Transformed noise array
    """
    return np.vectorize(curve_func)(noise)


def apply_threshold(noise: np.ndarray, threshold: float, above: float = 1.0, below: float = 0.0) -> np.ndarray:
    """
    Apply threshold to noise, creating binary mask.
    
    Args:
        noise: Input noise array
        threshold: Threshold value
        above: Value for cells above threshold
        below: Value for cells below threshold
    
    Returns:
        Thresholded array
    """
    return np.where(noise >= threshold, above, below)


def smooth_noise(noise: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth noise using Gaussian filter.
    
    Args:
        noise: Input noise array
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Smoothed noise array
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(noise, sigma=sigma)
