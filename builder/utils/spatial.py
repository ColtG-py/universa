"""
World Builder - Spatial Utilities
Provides spatial calculations, distance transforms, and geometric operations
"""

import numpy as np
from typing import Tuple, List
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi


def calculate_distance_field(
    binary_mask: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Calculate Euclidean distance transform.
    
    Args:
        binary_mask: Boolean array where True indicates feature locations
        normalize: If True, normalize distances to [0, 1]
    
    Returns:
        Array of distances to nearest True cell
    """
    distances = distance_transform_edt(~binary_mask)
    
    if normalize and distances.max() > 0:
        distances = distances / distances.max()
    
    return distances


def calculate_gradient(elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient (slope) of elevation map.
    
    Args:
        elevation: 2D elevation array
    
    Returns:
        Tuple of (gradient_x, gradient_y) arrays
    """
    gradient_y, gradient_x = np.gradient(elevation)
    return gradient_x, gradient_y


def calculate_slope(elevation: np.ndarray) -> np.ndarray:
    """
    Calculate slope magnitude from elevation.
    
    Args:
        elevation: 2D elevation array
    
    Returns:
        Slope magnitude array
    """
    gradient_x, gradient_y = calculate_gradient(elevation)
    slope = np.sqrt(gradient_x**2 + gradient_y**2)
    return slope


def calculate_aspect(elevation: np.ndarray) -> np.ndarray:
    """
    Calculate aspect (direction of slope) from elevation.
    
    Args:
        elevation: 2D elevation array
    
    Returns:
        Aspect in degrees (0-360, where 0 is North)
    """
    gradient_x, gradient_y = calculate_gradient(elevation)
    aspect = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    # Convert to compass bearing (0 = North, clockwise)
    aspect = (90 - aspect) % 360
    return aspect


def generate_voronoi_diagram(
    width: int,
    height: int,
    num_points: int,
    seed: int
) -> np.ndarray:
    """
    Generate Voronoi diagram for given number of random points.
    
    Args:
        width: Width of output array
        height: Height of output array
        num_points: Number of Voronoi sites
        seed: Random seed
    
    Returns:
        Array where each cell contains ID of nearest site
    """
    rng = np.random.default_rng(seed)
    
    # Generate random point locations
    points = rng.random((num_points, 2))
    points[:, 0] *= width
    points[:, 1] *= height
    
    # Create output array
    voronoi_map = np.zeros((width, height), dtype=np.uint32)
    
    # For each cell, find nearest point
    for x in range(width):
        for y in range(height):
            distances = np.sqrt(
                (points[:, 0] - x)**2 +
                (points[:, 1] - y)**2
            )
            voronoi_map[x, y] = np.argmin(distances)
    
    return voronoi_map, points


def find_local_minima(elevation: np.ndarray, min_depth: float = 0.0) -> List[Tuple[int, int]]:
    """
    Find local minima in elevation map (potential lake locations).
    
    Args:
        elevation: 2D elevation array
        min_depth: Minimum depth below neighbors to count as minimum
    
    Returns:
        List of (x, y) coordinates of local minima
    """
    from scipy.ndimage import minimum_filter
    
    # Use minimum filter to find local minima
    local_min = minimum_filter(elevation, size=3)
    
    # Find where elevation equals local minimum (accounting for depth threshold)
    minima_mask = (elevation <= local_min) & (local_min - elevation >= min_depth)
    
    # Get coordinates
    minima_coords = np.argwhere(minima_mask)
    
    return [(int(x), int(y)) for x, y in minima_coords]


def find_local_maxima(elevation: np.ndarray, min_height: float = 0.0) -> List[Tuple[int, int]]:
    """
    Find local maxima in elevation map (mountain peaks).
    
    Args:
        elevation: 2D elevation array
        min_height: Minimum height above neighbors to count as maximum
    
    Returns:
        List of (x, y) coordinates of local maxima
    """
    from scipy.ndimage import maximum_filter
    
    # Use maximum filter to find local maxima
    local_max = maximum_filter(elevation, size=3)
    
    # Find where elevation equals local maximum
    maxima_mask = (elevation >= local_max) & (elevation - local_max >= min_height)
    
    # Get coordinates
    maxima_coords = np.argwhere(maxima_mask)
    
    return [(int(x), int(y)) for x, y in maxima_coords]


def calculate_flow_direction_d8(elevation: np.ndarray) -> np.ndarray:
    """
    Calculate D8 flow direction for each cell.
    Each cell flows to its steepest downhill neighbor.
    
    Args:
        elevation: 2D elevation array
    
    Returns:
        Array of flow directions (0-7 indicating direction to 8 neighbors)
        Direction encoding:
        7  0  1
        6  X  2
        5  4  3
    """
    height, width = elevation.shape
    flow_dir = np.zeros((height, width), dtype=np.uint8)
    
    # 8 neighbor offsets
    neighbors = [
        (-1, 0),   # 0: North
        (-1, 1),   # 1: Northeast
        (0, 1),    # 2: East
        (1, 1),    # 3: Southeast
        (1, 0),    # 4: South
        (1, -1),   # 5: Southwest
        (0, -1),   # 6: West
        (-1, -1),  # 7: Northwest
    ]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_elev = elevation[y, x]
            steepest_slope = 0.0
            steepest_dir = 0
            
            for direction, (dy, dx) in enumerate(neighbors):
                ny, nx = y + dy, x + dx
                
                # Calculate slope
                neighbor_elev = elevation[ny, nx]
                slope = current_elev - neighbor_elev
                
                # Adjust for diagonal distance
                if dx != 0 and dy != 0:
                    slope /= np.sqrt(2)
                
                if slope > steepest_slope:
                    steepest_slope = slope
                    steepest_dir = direction
            
            flow_dir[y, x] = steepest_dir
    
    return flow_dir


def calculate_flow_accumulation(
    flow_direction: np.ndarray,
    weights: np.ndarray = None
) -> np.ndarray:
    """
    Calculate flow accumulation based on D8 flow direction.
    
    Args:
        flow_direction: D8 flow direction array
        weights: Optional weights for each cell (e.g., precipitation)
    
    Returns:
        Flow accumulation array (number of upstream cells)
    """
    height, width = flow_direction.shape
    accumulation = np.ones((height, width), dtype=np.float32)
    
    if weights is not None:
        accumulation = weights.copy()
    
    # Direction vectors
    dir_vectors = [
        (-1, 0),   # 0: North
        (-1, 1),   # 1: Northeast
        (0, 1),    # 2: East
        (1, 1),    # 3: Southeast
        (1, 0),    # 4: South
        (1, -1),   # 5: Southwest
        (0, -1),   # 6: West
        (-1, -1),  # 7: Northwest
    ]
    
    # Process cells in order from high to low elevation
    # This ensures upstream cells are processed before downstream
    
    # Sort cells by flow direction iteration
    # Multiple passes to propagate flow
    for _ in range(max(height, width)):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                direction = flow_direction[y, x]
                dy, dx = dir_vectors[direction]
                
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < height and 0 <= nx < width:
                    accumulation[ny, nx] += accumulation[y, x]
    
    return accumulation


def get_neighbors_8(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    """
    Get 8-connected neighbors of a cell.
    
    Args:
        x, y: Cell coordinates
        width, height: Grid dimensions
    
    Returns:
        List of valid neighbor coordinates
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    
    return neighbors


def get_neighbors_4(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    """
    Get 4-connected neighbors of a cell.
    
    Args:
        x, y: Cell coordinates
        width, height: Grid dimensions
    
    Returns:
        List of valid neighbor coordinates
    """
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    
    return neighbors


def interpolate_2d(
    data: np.ndarray,
    x: float,
    y: float
) -> float:
    """
    Bilinear interpolation for sampling between grid points.
    
    Args:
        data: 2D array
        x, y: Coordinates to sample (can be non-integer)
    
    Returns:
        Interpolated value
    """
    height, width = data.shape
    
    # Clamp coordinates
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    
    # Get integer parts
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, width - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, height - 1)
    
    # Get fractional parts
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    v00 = data[y0, x0]
    v10 = data[y0, x1]
    v01 = data[y1, x0]
    v11 = data[y1, x1]
    
    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    
    return v0 * (1 - fy) + v1 * fy
