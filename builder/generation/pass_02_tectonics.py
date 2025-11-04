"""
World Builder - Pass 2: Tectonic Plate System
Generates tectonic plates using Voronoi diagrams.
Plates drive geological activity and mountain formation.
"""

import numpy as np
from typing import List

from config import WorldGenerationParams
from models.world import WorldState, TectonicSystem, TectonicPlate
from utils.spatial import generate_voronoi_diagram


def execute(world_state: WorldState, params: WorldGenerationParams):
    """
    Generate tectonic plate system using Voronoi diagrams.
    
    Args:
        world_state: World state to update
        params: Generation parameters
    """
    size = world_state.size
    seed = params.seed
    num_plates = params.num_plates
    
    print(f"  - Generating {num_plates} tectonic plates...")
    
    # Generate Voronoi diagram for plates
    plate_map, plate_centers = generate_voronoi_diagram(
        size, size, num_plates, seed
    )
    
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
