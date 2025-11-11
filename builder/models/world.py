"""
World Builder - World Data Models (UPDATED FOR PASS 14)
Data structures for representing world state and chunks

UPDATED: Added resource attributes for Pass 14 (Natural Resources)
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import numpy as np

from config import (
    WorldStatus,
    WorldGenerationParams,
    FeatureType,
    RockType,
    SoilType,
    DrainageClass,
    Mineral,
    FaunaCategory,
    TimberType,
    QuarryType,
    ElementalAffinity,
    EnchantedLocationType,
    CHUNK_SIZE
)


# =============================================================================
# WORLD METADATA
# =============================================================================

class WorldMetadata(BaseModel):
    """
    Metadata stored in database for each generated world.
    Tracks generation progress and parameters.
    """
    world_id: UUID = Field(default_factory=uuid4)
    seed: int
    size: int
    generation_params: Dict[str, Any]
    status: WorldStatus = WorldStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    current_pass: Optional[str] = None
    progress_percent: float = 0.0
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


# =============================================================================
# PLANETARY DATA
# =============================================================================

class PlanetaryData(BaseModel):
    """
    Global planetary parameters calculated in Pass 1.
    Used by all subsequent passes.
    """
    gravity: float
    erosion_modifier: float
    seasonal_variation: float
    coriolis_parameter: float
    day_length_seconds: float
    solar_input: float = 1361.0  # W/m²
    
    # Derived climate zones
    tropic_latitude: float = 23.5
    arctic_latitude: float = 66.5


# =============================================================================
# TECTONIC DATA
# =============================================================================

class TectonicPlate(BaseModel):
    """Individual tectonic plate information"""
    plate_id: int
    center_x: float
    center_y: float
    velocity_x: float  # mm/year
    velocity_y: float  # mm/year
    is_oceanic: bool


class TectonicSystem(BaseModel):
    """Complete tectonic plate system"""
    plates: List[TectonicPlate]
    num_plates: int
    plate_speed_mm_year: float


# =============================================================================
# GEOLOGICAL FEATURES
# =============================================================================

class GeologicalFeature(BaseModel):
    """Point of interest geological feature"""
    feature_id: UUID = Field(default_factory=uuid4)
    type: FeatureType
    location_x: int
    location_y: int
    chunk_x: int
    chunk_y: int
    properties: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# MAGIC SYSTEM DATA STRUCTURES
# =============================================================================

class LeyLineSegment(BaseModel):
    """
    A segment of a ley line connecting two anchor points.
    Ley lines form a network of magical energy flows.
    """
    segment_id: UUID = Field(default_factory=uuid4)
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    path_points: List[Tuple[int, int]]  # Actual path through world
    strength: float = Field(ge=0.0, le=1.0, description="Magical power of this ley line")
    
    class Config:
        arbitrary_types_allowed = True


class EnchantedLocation(BaseModel):
    """
    Special magical location of interest.
    These are points of concentrated magical power or corruption.
    """
    location_id: UUID = Field(default_factory=uuid4)
    location_type: str  # "mana_well", "fey_grove", "dragon_lair", etc.
    location_x: int
    location_y: int
    chunk_x: int
    chunk_y: int
    power_level: float = Field(ge=0.0, le=1.0, description="Magical power at this location")
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

# =============================================================================
# WORLD CHUNK DATA
# =============================================================================

class WorldChunk:
    """
    A 256x256 section of the world.
    Contains all layer data for this chunk.
    Uses NumPy arrays for efficient storage and computation.
    """
    
    def __init__(self, chunk_x: int, chunk_y: int, world_size: int):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.world_size = world_size
        self.size = CHUNK_SIZE
        
        # Initialize all data layers as None
        # They will be populated during generation passes
        
        # Pass 1-3: Foundation & Topography
        self.elevation: Optional[np.ndarray] = None  # float32[256, 256] - meters
        self.plate_id: Optional[np.ndarray] = None  # uint8[256, 256]
        self.tectonic_stress: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        
        # Pass 4: Geology
        self.bedrock_type: Optional[np.ndarray] = None  # uint8[256, 256] - RockType enum
        self.mineral_richness: Optional[Dict[Mineral, np.ndarray]] = None  # Dict of float32[256, 256] per mineral
        self.soil_depth_cm: Optional[np.ndarray] = None  # uint16[256, 256]
        
        # Pass 5-7: Climate
        self.temperature_c: Optional[np.ndarray] = None  # float32[256, 256] - annual average
        self.precipitation_mm: Optional[np.ndarray] = None  # uint16[256, 256] - annual
        self.wind_direction: Optional[np.ndarray] = None  # uint16[256, 256] - degrees
        self.wind_speed: Optional[np.ndarray] = None  # float32[256, 256] - m/s
        
        # Pass 8-10: Hydrology
        self.water_table_depth: Optional[np.ndarray] = None  # float32[256, 256] - meters
        self.river_presence: Optional[np.ndarray] = None  # bool[256, 256]
        self.river_flow: Optional[np.ndarray] = None  # float32[256, 256] - m³/s
        self.drainage_basin_id: Optional[np.ndarray] = None  # uint32[256, 256]
        
        # Pass 11: Soil
        self.soil_type: Optional[np.ndarray] = None  # uint8[256, 256] - SoilType enum
        self.soil_ph: Optional[np.ndarray] = None  # float32[256, 256]
        self.soil_drainage: Optional[np.ndarray] = None  # uint8[256, 256] - DrainageClass enum
        
        # Pass 12: Biomes & Vegetation
        self.biome_type: Optional[np.ndarray] = None  # uint8[256, 256] - BiomeType enum
        self.vegetation_density: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        self.forest_canopy_height: Optional[np.ndarray] = None  # float32[256, 256] - meters
        self.agricultural_suitability: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale

        # Pass 13: Fauna
        self.fauna_density: Optional[Dict[FaunaCategory, np.ndarray]] = None  # Dict of float32[256, 256] per category
        self.apex_predator_territories: Optional[np.ndarray] = None  # uint32[256, 256] - territory IDs
        self.migration_routes: Optional[np.ndarray] = None  # bool[256, 256] - seasonal migration corridors
        
        # Pass 14: Natural Resources
        self.mineral_deposits: Optional[Dict[Mineral, np.ndarray]] = None  # Dict of float32[256, 256] - concentrated veins
        self.quarry_quality: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        self.quarry_type: Optional[np.ndarray] = None  # uint8[256, 256] - QuarryType enum
        self.timber_quality: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        self.timber_type: Optional[np.ndarray] = None  # uint8[256, 256] - TimberType enum
        self.agricultural_yield: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale (tons/hectare)
        self.fishing_quality: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        self.rare_resources: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale (gemstones, magical)
        self.resource_accessibility: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale (extraction difficulty)
        
        # Pass 15: Magic & Ley Lines
        self.mana_concentration: Optional[np.ndarray] = None  # float32[256, 256] - 0-1 scale
        self.ley_line_presence: Optional[np.ndarray] = None  # bool[256, 256]
        self.ley_line_node: Optional[np.ndarray] = None  # bool[256, 256] - intersection points
        self.corrupted_zone: Optional[np.ndarray] = None  # bool[256, 256]
        self.elemental_affinity: Optional[np.ndarray] = None  # uint8[256, 256] - ElementalAffinity enum
    
        # Enchanted locations in this chunk
        self.enchanted_locations: List[EnchantedLocation] = []

        # Geological features (discrete points)
        self.geological_features: List[GeologicalFeature] = []
        
        # Metadata
        self.generated_at: Optional[datetime] = None
        self.version: str = "1.0"
    
    def get_global_x(self, local_x: int) -> int:
        """Convert chunk-local X coordinate to global world X"""
        return self.chunk_x * CHUNK_SIZE + local_x
    
    def get_global_y(self, local_y: int) -> int:
        """Convert chunk-local Y coordinate to global world Y"""
        return self.chunk_y * CHUNK_SIZE + local_y
    
    def is_valid_coord(self, x: int, y: int) -> bool:
        """Check if local coordinates are within chunk bounds"""
        return 0 <= x < self.size and 0 <= y < self.size
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize chunk to dictionary for storage.
        Arrays are converted to lists for JSON serialization.
        """
        data = {
            "chunk_x": self.chunk_x,
            "chunk_y": self.chunk_y,
            "world_size": self.world_size,
            "size": self.size,
            "version": self.version,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }
        
        # Serialize numpy arrays
        array_fields = [
            "elevation", "plate_id", "tectonic_stress",
            "bedrock_type", "soil_depth_cm",
            "temperature_c", "precipitation_mm", "wind_direction", "wind_speed",
            "water_table_depth", "river_presence", "river_flow", "drainage_basin_id",
            "soil_type", "soil_ph", "soil_drainage",
            "biome_type", "vegetation_density", "forest_canopy_height", "agricultural_suitability",
            "apex_predator_territories", "migration_routes",
            "quarry_quality", "quarry_type", "timber_quality", "timber_type",
            "agricultural_yield", "fishing_quality", "rare_resources", "resource_accessibility",
            "mana_concentration", "ley_line_presence", "ley_line_node",
            "corrupted_zone", "elemental_affinity",
        ]
        
        for field in array_fields:
            arr = getattr(self, field)
            if arr is not None:
                data[field] = arr.tolist()
        
        # Serialize mineral richness dictionary
        if self.mineral_richness is not None:
            data["mineral_richness"] = {
                int(mineral): arr.tolist()
                for mineral, arr in self.mineral_richness.items()
            }
        
        # Serialize mineral deposits dictionary
        if self.mineral_deposits is not None:
            data["mineral_deposits"] = {
                int(mineral): arr.tolist()
                for mineral, arr in self.mineral_deposits.items()
            }
        
        # Serialize fauna density dictionary
        if self.fauna_density is not None:
            data["fauna_density"] = {
                int(fauna_cat): arr.tolist()
                for fauna_cat, arr in self.fauna_density.items()
            }
        
        # Serialize geological features
        data["geological_features"] = [
            feature.dict() for feature in self.geological_features
        ]

        if self.enchanted_locations:
            data["enchanted_locations"] = [
                loc.dict() for loc in self.enchanted_locations
            ]
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldChunk':
        """Deserialize chunk from dictionary"""
        chunk = cls(data["chunk_x"], data["chunk_y"], data["world_size"])
        chunk.version = data["version"]
        
        if data.get("generated_at"):
            chunk.generated_at = datetime.fromisoformat(data["generated_at"])
        
        # Deserialize numpy arrays
        array_fields = {
            "elevation": np.float32,
            "plate_id": np.uint8,
            "tectonic_stress": np.float32,
            "bedrock_type": np.uint8,
            "soil_depth_cm": np.uint16,
            "temperature_c": np.float32,
            "precipitation_mm": np.uint16,
            "wind_direction": np.uint16,
            "wind_speed": np.float32,
            "water_table_depth": np.float32,
            "river_presence": bool,
            "river_flow": np.float32,
            "drainage_basin_id": np.uint32,
            "soil_type": np.uint8,
            "soil_ph": np.float32,
            "soil_drainage": np.uint8,
            "biome_type": np.uint8,
            "vegetation_density": np.float32,
            "forest_canopy_height": np.float32,
            "agricultural_suitability": np.float32,
            "apex_predator_territories": np.uint32,
            "migration_routes": bool,
            "quarry_quality": np.float32,
            "quarry_type": np.uint8,
            "timber_quality": np.float32,
            "timber_type": np.uint8,
            "agricultural_yield": np.float32,
            "fishing_quality": np.float32,
            "rare_resources": np.float32,
            "resource_accessibility": np.float32,
            "mana_concentration": np.float32,
            "ley_line_presence": bool,
            "ley_line_node": bool,
            "corrupted_zone": bool,
            "elemental_affinity": np.uint8,
        }
        
        for field, dtype in array_fields.items():
            if field in data and data[field] is not None:
                setattr(chunk, field, np.array(data[field], dtype=dtype))
        
        # Deserialize mineral richness
        if "mineral_richness" in data and data["mineral_richness"] is not None:
            chunk.mineral_richness = {
                Mineral(int(k)): np.array(v, dtype=np.float32)
                for k, v in data["mineral_richness"].items()
            }
        
        # Deserialize mineral deposits
        if "mineral_deposits" in data and data["mineral_deposits"] is not None:
            chunk.mineral_deposits = {
                Mineral(int(k)): np.array(v, dtype=np.float32)
                for k, v in data["mineral_deposits"].items()
            }
        
        # Deserialize fauna density
        if "fauna_density" in data and data["fauna_density"] is not None:
            chunk.fauna_density = {
                FaunaCategory(int(k)): np.array(v, dtype=np.float32)
                for k, v in data["fauna_density"].items()
            }
        
        # Deserialize geological features
        if "geological_features" in data:
            chunk.geological_features = [
                GeologicalFeature(**feature)
                for feature in data["geological_features"]
            ]

        if "enchanted_locations" in data and data["enchanted_locations"]:
            chunk.enchanted_locations = [
                EnchantedLocation(**loc)
                for loc in data["enchanted_locations"]
            ]
        
        return chunk


# =============================================================================
# WORLD STATE
# =============================================================================

class WorldState:
    """
    Complete world state container.
    Manages chunks and provides query interface.
    """
    
    def __init__(self, metadata: WorldMetadata, params: WorldGenerationParams):
        self.metadata = metadata
        self.params = params
        size_val = params.size
        if isinstance(size_val, str):
            self.size = int(size_val)
        else:
            self.size = size_val.to_int()
        self.num_chunks = self.size // CHUNK_SIZE
        
        # Planetary data (from Pass 1)
        self.planetary_data: Optional[PlanetaryData] = None
        
        # Tectonic system (from Pass 2)
        self.tectonic_system: Optional[TectonicSystem] = None
        
        # Ley line network (from Pass 15)
        self.ley_line_network: Optional[List[LeyLineSegment]] = None

        # Chunks dictionary: (chunk_x, chunk_y) -> WorldChunk
        self.chunks: Dict[tuple, WorldChunk] = {}
    
    def get_chunk(self, chunk_x: int, chunk_y: int) -> Optional[WorldChunk]:
        """Get chunk at specified chunk coordinates"""
        return self.chunks.get((chunk_x, chunk_y))
    
    def get_or_create_chunk(self, chunk_x: int, chunk_y: int) -> WorldChunk:
        """Get existing chunk or create new empty chunk"""
        key = (chunk_x, chunk_y)
        if key not in self.chunks:
            self.chunks[key] = WorldChunk(chunk_x, chunk_y, self.size)
        return self.chunks[key]
    
    def get_chunk_for_location(self, x: int, y: int) -> tuple:
        """Get chunk coordinates for a world location"""
        chunk_x = x // CHUNK_SIZE
        chunk_y = y // CHUNK_SIZE
        return chunk_x, chunk_y
    
    def query_location(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """
        Query all data at a specific world location.
        Returns None if chunk not generated.
        """
        chunk_x, chunk_y = self.get_chunk_for_location(x, y)
        chunk = self.get_chunk(chunk_x, chunk_y)
        
        if chunk is None:
            return None
        
        # Convert to local chunk coordinates
        local_x = x % CHUNK_SIZE
        local_y = y % CHUNK_SIZE
        
        result = {
            "x": x,
            "y": y,
            "chunk_x": chunk_x,
            "chunk_y": chunk_y,
        }
        
        # Extract data from all layers
        if chunk.elevation is not None:
            result["elevation"] = float(chunk.elevation[local_x, local_y])
        
        if chunk.temperature_c is not None:
            result["temperature_c"] = float(chunk.temperature_c[local_x, local_y])
        
        if chunk.precipitation_mm is not None:
            result["precipitation_mm"] = int(chunk.precipitation_mm[local_x, local_y])
        
        if chunk.bedrock_type is not None:
            result["bedrock_type"] = RockType(chunk.bedrock_type[local_x, local_y]).name
        
        if chunk.soil_type is not None:
            result["soil_type"] = SoilType(chunk.soil_type[local_x, local_y]).name
        
        # Add more fields as needed
        
        return result