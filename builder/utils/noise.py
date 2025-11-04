"""
World Builder - Noise Generation Utilities
Provides deterministic noise generation for procedural world generation

UPDATED: Now uses OpenSimplex for properly seeded noise generation
This is a drop-in replacement that maintains backward compatibility.
"""

import numpy as np
from typing import Optional
from opensimplex import OpenSimplex


class NoiseGenerator:
    """
    Deterministic noise generator using OpenSimplex noise.
    
    This replaces the previous implementation that used the 'noise' library,
    which did not properly respect seeds. OpenSimplex ensures:
    - Deterministic generation (same seed = same result)
    - Proper seeding (different seeds = different results)
    - Seamless chunk boundaries
    - Better performance
    
    All noise is seeded to ensure reproducible generation.
    """
    
    def __init__(
        self,
        seed: int,
        octaves: int = 8,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 10.0
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
        
        # Create OpenSimplex generator with proper seeding
        self.simplex = OpenSimplex(seed=seed)
    
    def generate_perlin_2d(
        self,
        width: int,
        height: int,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate 2D noise array (using OpenSimplex instead of Perlin).
        
        Args:
            width: Width of output array
            height: Height of output array
            offset_x: X offset for seamless tiling (in world coordinates)
            offset_y: Y offset for seamless tiling (in world coordinates)
            normalize: If True, normalize output to [0, 1]
        
        Returns:
            2D numpy array of noise values
        """
        noise_array = np.zeros((width, height), dtype=np.float32)
        
        for x in range(width):
            for y in range(height):
                # Calculate global coordinates in noise space
                nx = (x + offset_x) / self.scale
                ny = (y + offset_y) / self.scale
                
                # Multi-octave noise generation
                value = 0.0
                amplitude = 1.0
                frequency = 1.0
                max_value = 0.0
                
                for octave in range(self.octaves):
                    sample_x = nx * frequency
                    sample_y = ny * frequency
                    
                    # OpenSimplex noise - properly seeded
                    noise_val = self.simplex.noise2(sample_x, sample_y)
                    
                    value += noise_val * amplitude
                    max_value += amplitude
                    
                    amplitude *= self.persistence
                    frequency *= self.lacunarity
                
                noise_array[x, y] = value / max_value
        
        if normalize:
            # Normalize to [0, 1]
            # OpenSimplex returns values in roughly [-1, 1]
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
        
        Note: This is actually the same as generate_perlin_2d now since
        we're using OpenSimplex for both. Kept for backward compatibility.
        
        Args:
            width: Width of output array
            height: Height of output array
            offset_x: X offset for seamless tiling
            offset_y: Y offset for seamless tiling
            normalize: If True, normalize output to [0, 1]
        
        Returns:
            2D numpy array of noise values
        """
        # Same implementation as generate_perlin_2d
        return self.generate_perlin_2d(width, height, offset_x, offset_y, normalize)
    
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
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        warp_strength: float = 10.0
    ) -> np.ndarray:
        """
        Generate domain-warped noise for more organic patterns.
        
        Args:
            width: Width of output array
            height: Height of output array
            offset_x: X offset in world coordinates
            offset_y: Y offset in world coordinates
            warp_strength: How much to warp the domain
        
        Returns:
            2D numpy array normalized to [0, 1]
        """
        # Generate two noise fields for warping
        warp_x = self.generate_perlin_2d(width, height, offset_x, offset_y, normalize=False)
        warp_y = self.generate_perlin_2d(width, height, offset_x + 100, offset_y + 100, normalize=False)
        
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
                
                # Calculate noise at warped position
                nx = (wx + offset_x) / self.scale
                ny = (wy + offset_y) / self.scale
                
                # Use a different seed offset for the final sample
                result[x, y] = self.simplex.noise2(nx + 1000, ny + 1000)
        
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


# For backward compatibility - export these at module level
__all__ = [
    'NoiseGenerator',
    'combine_noise_layers',
    'apply_curve',
    'apply_threshold',
    'smooth_noise',
]


# Self-test when run directly
if __name__ == "__main__":
    print("\n" + "="*70)
    print("OpenSimplex-Based Noise Generator - Self Test")
    print("="*70 + "\n")
    
    # Test 1: Same seed produces same results
    print("Test 1: Deterministic generation (same seed = same result)...")
    gen1 = NoiseGenerator(seed=42, scale=10.0)
    noise1 = gen1.generate_perlin_2d(50, 50, 0, 0)
    
    gen2 = NoiseGenerator(seed=42, scale=10.0)
    noise2 = gen2.generate_perlin_2d(50, 50, 0, 0)
    
    same = np.allclose(noise1, noise2)
    print(f"  Result: {'PASS ✓' if same else 'FAIL ✗'}")
    if not same:
        print(f"  Max diff: {np.abs(noise1 - noise2).max()}")
    
    # Test 2: Different seeds produce different results
    print("\nTest 2: Different seeds produce different results...")
    gen3 = NoiseGenerator(seed=123, scale=10.0)
    noise3 = gen3.generate_perlin_2d(50, 50, 0, 0)
    
    different = not np.allclose(noise1, noise3)
    diff_amount = np.abs(noise1 - noise3).mean()
    print(f"  Result: {'PASS ✓' if different else 'FAIL ✗'}")
    print(f"  Mean difference: {diff_amount:.4f}")
    
    # Test 3: Chunk boundary continuity
    print("\nTest 3: Seamless chunk boundaries...")
    gen4 = NoiseGenerator(seed=42, scale=10.0)
    
    chunk0 = gen4.generate_perlin_2d(64, 64, 0, 0)
    chunk1 = gen4.generate_perlin_2d(64, 64, 64, 0)
    
    # Check boundary
    right_edge = chunk0[-1, :]
    left_edge = chunk1[0, :]
    
    boundary_diff = np.abs(right_edge - left_edge)
    continuous = boundary_diff.max() < 0.01
    
    print(f"  Result: {'PASS ✓' if continuous else 'FAIL ✗'}")
    print(f"  Max boundary difference: {boundary_diff.max():.6f}")
    print(f"  Mean boundary difference: {boundary_diff.mean():.6f}")
    
    # Test 4: All noise types work
    print("\nTest 4: All noise generation methods work...")
    try:
        gen5 = NoiseGenerator(seed=99, scale=20.0)
        perlin = gen5.generate_perlin_2d(32, 32)
        simplex = gen5.generate_simplex_2d(32, 32)
        ridged = gen5.generate_ridged_noise(32, 32)
        billow = gen5.generate_billow_noise(32, 32)
        warped = gen5.generate_domain_warped_noise(32, 32)
        
        all_valid = all([
            perlin.shape == (32, 32),
            simplex.shape == (32, 32),
            ridged.shape == (32, 32),
            billow.shape == (32, 32),
            warped.shape == (32, 32),
        ])
        print(f"  Result: {'PASS ✓' if all_valid else 'FAIL ✗'}")
    except Exception as e:
        print(f"  Result: FAIL ✗")
        print(f"  Error: {e}")
    
    # Test 5: Utility functions work
    print("\nTest 5: Utility functions work...")
    try:
        layer1 = gen5.generate_perlin_2d(32, 32)
        layer2 = gen5.generate_perlin_2d(32, 32, offset_x=100)
        
        combined = combine_noise_layers([layer1, layer2], [0.7, 0.3])
        curved = apply_curve(layer1, lambda x: x ** 2)
        thresholded = apply_threshold(layer1, 0.5)
        smoothed = smooth_noise(layer1, sigma=1.0)
        
        all_valid = all([
            combined.shape == (32, 32),
            curved.shape == (32, 32),
            thresholded.shape == (32, 32),
            smoothed.shape == (32, 32),
        ])
        print(f"  Result: {'PASS ✓' if all_valid else 'FAIL ✗'}")
    except Exception as e:
        print(f"  Result: FAIL ✗")
        print(f"  Error: {e}")
    
    print("\n" + "="*70)
    if same and different and continuous:
        print("✓ ALL CRITICAL TESTS PASSED!")
        print("\nThis noise generator:")
        print("  • Produces deterministic results")
        print("  • Respects seed values")
        print("  • Generates seamless chunks")
        print("  • Is ready for world generation!")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Check the implementation above")
    print("="*70 + "\n")