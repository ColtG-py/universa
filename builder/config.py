"""
World Builder - Configuration and Constants
Contains all global settings, constants, and configuration for world generation
"""

from enum import IntEnum, Enum
from typing import Optional
from pydantic import BaseModel, Field

# =============================================================================
# WORLD SIZES
# =============================================================================

WORLD_SIZE_SMALL = 512
WORLD_SIZE_MEDIUM = 1024
WORLD_SIZE_LARGE = 2048
WORLD_SIZE_HUGE = 4096

# =============================================================================
# CHUNK CONFIGURATION
# =============================================================================

CHUNK_SIZE = 256  # Each chunk is 256x256 tiles

# =============================================================================
# ENUMERATIONS
# =============================================================================

class WorldSize(str, Enum):
    """Available world sizes"""
    SMALL = "512"
    MEDIUM = "1024"
    LARGE = "2048"
    HUGE = "4096"
    
    def to_int(self) -> int:
        return int(self.value)


class WorldStatus(str, Enum):
    """World generation status"""
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    PAUSED = "paused"


class RockType(IntEnum):
    """Types of bedrock"""
    IGNEOUS = 0
    SEDIMENTARY = 1
    METAMORPHIC = 2
    LIMESTONE = 3


class Mineral(IntEnum):
    """Mineral types found in the world"""
    IRON = 0
    COPPER = 1
    GOLD = 2
    SILVER = 3
    COAL = 4
    SALT = 5
    DIAMOND = 6
    EMERALD = 7


class SoilType(IntEnum):
    """Soil classification types"""
    SAND = 0
    SILT = 1
    CLAY = 2
    LOAM = 3
    PEAT = 4


class DrainageClass(IntEnum):
    """Soil drainage capability"""
    VERY_POORLY = 0
    POORLY_DRAINED = 1
    SOMEWHAT_POORLY = 2
    MODERATELY_WELL = 3
    WELL_DRAINED = 4
    EXCESSIVELY = 5


class FeatureType(str, Enum):
    """Geological features"""
    CAVE_SYSTEM = "cave_system"
    HOT_SPRING = "hot_spring"
    CANYON = "canyon"
    NATURAL_BRIDGE = "natural_bridge"
    WATERFALL = "waterfall"
    SINKHOLE = "sinkhole"
    LAVA_TUBE = "lava_tube"
    MINERAL_VEIN = "mineral_vein"


# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

class WorldGenerationParams(BaseModel):
    """
    Input parameters for world generation.
    All generation parameters can be customized.
    """
    seed: int = Field(..., description="Random seed for deterministic generation")
    size: WorldSize = Field(WorldSize.MEDIUM, description="World size")
    
    # Planetary Parameters
    planet_radius_km: float = Field(6371.0, description="Planet radius in kilometers")
    gravity: float = Field(9.8, description="Surface gravity in m/s²")
    axial_tilt: float = Field(23.5, description="Axial tilt in degrees")
    rotation_hours: float = Field(24.0, description="Day length in hours")
    
    # Tectonic Parameters
    num_plates: int = Field(12, ge=4, le=30, description="Number of tectonic plates")
    plate_speed_mm_year: float = Field(50.0, description="Plate movement speed in mm/year")
    
    # Atmospheric Parameters
    base_temperature_c: float = Field(15.0, description="Base temperature in Celsius")
    atmospheric_pressure_atm: float = Field(1.0, description="Atmospheric pressure in atmospheres")
    
    # Hydrological Parameters
    ocean_percentage: float = Field(0.7, ge=0.0, le=1.0, description="Percentage of world covered by ocean")
    
    # Noise Parameters (optional overrides)
    custom_noise_octaves: Optional[int] = Field(6, ge=1, le=12, description="Octaves for Perlin noise")
    custom_noise_persistence: Optional[float] = Field(0.5, ge=0.1, le=0.9, description="Persistence for Perlin noise")
    custom_noise_lacunarity: Optional[float] = Field(2.0, ge=1.5, le=3.0, description="Lacunarity for Perlin noise")
    
    # Erosion Parameters
    erosion_iterations: int = Field(3, ge=1, le=10, description="Number of erosion simulation iterations")
    erosion_strength: float = Field(1.0, ge=0.1, le=5.0, description="Erosion strength multiplier")
    
    # Advanced Generation Options
    enable_caves: bool = Field(True, description="Generate cave systems")
    enable_hot_springs: bool = Field(True, description="Generate hot springs")
    enable_waterfalls: bool = Field(True, description="Generate waterfalls")
    
    class Config:
        use_enum_values = True


# =============================================================================
# PASS CONFIGURATION
# =============================================================================

# List of all generation passes in order
GENERATION_PASSES = [
    "pass_01_planetary",
    "pass_02_tectonics",
    "pass_03_topography",
    "pass_04_geology",
    "pass_05_atmosphere",
    "pass_06_oceans",
    "pass_07_climate",
    "pass_08_erosion",
    "pass_09_groundwater",
    "pass_10_rivers",
    "pass_11_soil",
    "pass_12_microclimate",
    "pass_13_features",
    "pass_14_polish",
]

# Pass weights for progress calculation
PASS_WEIGHTS = {
    "pass_01_planetary": 2,
    "pass_02_tectonics": 8,
    "pass_03_topography": 10,
    "pass_04_geology": 8,
    "pass_05_atmosphere": 6,
    "pass_06_oceans": 7,
    "pass_07_climate": 9,
    "pass_08_erosion": 12,
    "pass_09_groundwater": 7,
    "pass_10_rivers": 10,
    "pass_11_soil": 8,
    "pass_12_microclimate": 5,
    "pass_13_features": 6,
    "pass_14_polish": 2,
}

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

SOLAR_CONSTANT = 1361  # W/m^2
EARTH_RADIUS_KM = 6371.0
EARTH_GRAVITY = 9.8  # m/s^2
WATER_FREEZING_C = 0.0
WATER_BOILING_C = 100.0

# =============================================================================
# LOOKUP TABLES
# =============================================================================

# Rock type permeability (for groundwater calculations)
ROCK_PERMEABILITY = {
    RockType.IGNEOUS: 0.3,
    RockType.SEDIMENTARY: 0.7,
    RockType.METAMORPHIC: 0.4,
    RockType.LIMESTONE: 0.9,
}

# Mineral occurrence probabilities by rock type
MINERAL_PROBABILITIES = {
    RockType.IGNEOUS: {
        Mineral.IRON: 0.4,
        Mineral.COPPER: 0.3,
        Mineral.GOLD: 0.1,
        Mineral.DIAMOND: 0.05,
    },
    RockType.SEDIMENTARY: {
        Mineral.COAL: 0.5,
        Mineral.SALT: 0.3,
        Mineral.IRON: 0.2,
    },
    RockType.METAMORPHIC: {
        Mineral.GOLD: 0.3,
        Mineral.SILVER: 0.25,
        Mineral.EMERALD: 0.1,
        Mineral.COPPER: 0.2,
    },
    RockType.LIMESTONE: {
        Mineral.SALT: 0.4,
        Mineral.IRON: 0.2,
    },
}
