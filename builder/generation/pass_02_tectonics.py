"""
World Builder - Pass 2: Tectonic Plate System (FIXED VERSION)
Generates tectonic plates with more organic, realistic boundaries.
Uses domain-warped Voronoi to create irregular plate shapes.
"""

import numpy as np
from typing import List

from config import WorldGenerationParams
from models.world import WorldState, TectonicSystem, TectonicPlate
from utils.spatial import generate_voronoi_diagram
from utils.noise import NoiseGenerator


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate tectonic plate system with organic boundaries.
    
    CHANGES FROM ORIGINAL:
    - Added domain warping to Voronoi boundaries for irregular shapes
    - Reduced harsh geometric appearance
    - More realistic plate boundaries
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    size = world_state.size
    seed = params.seed
    num_plates = params.num_plates
    
    print(f"  - Generating {num_plates} tectonic plates with organic boundaries...")
    
    # Generate base Voronoi diagram
    plate_map, plate_centers = generate_voronoi_diagram(
        size, size, num_plates, seed
    )
    
    # NEW: Apply domain warping to make boundaries more organic
    print(f"  - Applying domain warping for realistic plate boundaries...")
    plate_map = apply_domain_warping_to_plates(plate_map, size, seed)
    
    # Generate plate velocities
    rng = np.random.default_rng(seed)
    
    plates: List[TectonicPlate] = []
    
    for i in range(num_plates):
        # Random velocity direction
        angle = rng.random() * 2 * np.pi
        speed = params.plate_speed_mm_year * (0.5 + rng.random() * 0.5)
        
        velocity_x = np.cos(angle) * speed
        velocity_y = np.sin(angle) * speed
        
        # Determine if oceanic or continental
        # Roughly 60% oceanic, 40% continental
        is_oceanic = rng.random() < 0.6
        
        plate = TectonicPlate(
            plate_id=i,
            center_x=float(plate_centers[i, 0]),
            center_y=float(plate_centers[i, 1]),
            velocity_x=float(velocity_x),
            velocity_y=float(velocity_y),
            is_oceanic=is_oceanic,
        )
        
        plates.append(plate)
    
    # Create tectonic system
    tectonic_system = TectonicSystem(
        plates=plates,
        num_plates=num_plates,
        plate_speed_mm_year=params.plate_speed_mm_year,
    )
    
    world_state.tectonic_system = tectonic_system
    
    # Calculate plate boundaries and stress
    print(f"  - Calculating plate boundaries...")
    
    num_chunks_x = size // 256
    num_chunks_y = size // 256
    
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            chunk = world_state.get_or_create_chunk(chunk_x, chunk_y)
            
            # Initialize arrays
            chunk.plate_id = np.zeros((256, 256), dtype=np.uint8)
            chunk.tectonic_stress = np.zeros((256, 256), dtype=np.float32)
            
            # Fill in plate IDs for this chunk
            for local_y in range(256):
                for local_x in range(256):
                    global_x = chunk_x * 256 + local_x
                    global_y = chunk_y * 256 + local_y
                    
                    if global_x < size and global_y < size:
                        chunk.plate_id[local_x, local_y] = plate_map[global_x, global_y]
            
            # Calculate tectonic stress (high at plate boundaries)
            from scipy.ndimage import sobel
            
            # Detect plate boundaries using edge detection
            edges_x = sobel(chunk.plate_id.astype(float), axis=0)
            edges_y = sobel(chunk.plate_id.astype(float), axis=1)
            edges = np.sqrt(edges_x**2 + edges_y**2)
            
            # Normalize stress to [0, 1]
            if edges.max() > 0:
                chunk.tectonic_stress = edges / edges.max()
            
            # High stress within a few cells of boundaries
            from scipy.ndimage import maximum_filter
            chunk.tectonic_stress = maximum_filter(chunk.tectonic_stress, size=5)
            chunk.tectonic_stress = np.clip(chunk.tectonic_stress, 0, 1)
    
    oceanic_count = sum(1 for p in plates if p.is_oceanic)
    continental_count = num_plates - oceanic_count
    
    print(f"  - {oceanic_count} oceanic plates, {continental_count} continental plates")
    print(f"  - Average plate speed: {params.plate_speed_mm_year:.1f} mm/year")


def apply_domain_warping_to_plates(
    plate_map: np.ndarray,
    size: int,
    seed: int,
    warp_strength: float = 150.0
) -> np.ndarray:
    """
    Apply domain warping to Voronoi plate boundaries to create organic shapes.
    
    This makes tectonic plates look more realistic by warping their boundaries
    with noise, similar to how real tectonic plates have irregular shapes.
    
    Args:
        plate_map: Original Voronoi plate map
        size: Size of the map
        seed: Random seed for reproducibility
        warp_strength: How much to warp boundaries (in pixels)
        
    Returns:
        Warped plate map with organic boundaries
    """
    # Create two noise fields for X and Y warping
    warp_noise_x = NoiseGenerator(
        seed=seed + 5000,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=size / 4.0  # Large-scale warping
    )
    
    warp_noise_y = NoiseGenerator(
        seed=seed + 6000,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        scale=size / 4.0
    )
    
    # Generate warp fields
    print(f"    - Generating warp fields...")
    warp_x = warp_noise_x.generate_perlin_2d(size, size, 0, 0, normalize=False)
    warp_y = warp_noise_y.generate_perlin_2d(size, size, 0, 0, normalize=False)
    
    # Scale warp fields
    warp_x = warp_x * warp_strength
    warp_y = warp_y * warp_strength
    
    # Apply warping
    print(f"    - Applying domain warp to plate boundaries...")
    warped_plate_map = np.zeros_like(plate_map)
    
    for y in range(size):
        for x in range(size):
            # Calculate warped sampling position
            sample_x = x + warp_x[x, y]
            sample_y = y + warp_y[x, y]
            
            # Clamp to bounds
            sample_x = int(np.clip(sample_x, 0, size - 1))
            sample_y = int(np.clip(sample_y, 0, size - 1))
            
            # Sample from warped position
            warped_plate_map[x, y] = plate_map[sample_x, sample_y]
    
    return warped_plate_map