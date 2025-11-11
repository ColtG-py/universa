"""
World Builder - Configuration and Constants
Contains all global settings, constants, and configuration for world generation

UPDATED: Added ElementalAffinity and EnchantedLocationType for Pass 15
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
    SAND = 0
    SANDY_LOAM = 1
    LOAM = 2
    SILT_LOAM = 3
    CLAY_LOAM = 4
    SANDY_CLAY_LOAM = 5
    SILTY_CLAY_LOAM = 6
    CLAY = 7
    SILTY_CLAY = 8
    SANDY_CLAY = 9


class DrainageClass(IntEnum):
    EXCESSIVELY = 0                  # Water removed very rapidly
    SOMEWHAT_EXCESSIVELY = 1         # Water removed rapidly
    WELL = 2                         # Water removed readily but not rapidly
    MODERATELY_WELL = 3              # Water removed somewhat slowly
    SOMEWHAT_POORLY = 4              # Water removed slowly
    POORLY = 5                       # Water removed very slowly
    VERY_POORLY = 6                  # Free water at/near surface


class BiomeType(IntEnum):
    """Biome classifications based on Whittaker diagram + ocean subtypes"""
    # Ocean subtypes (negative elevations)
    OCEAN_TRENCH = 0                 # Very deep ocean (< -4000m), often at convergent boundaries
    OCEAN_DEEP = 1                   # Deep ocean (-4000m to -1000m)
    OCEAN_SHALLOW = 2                # Shallow ocean (-1000m to -200m)
    OCEAN_SHELF = 3                  # Continental shelf (-200m to 0m)
    OCEAN_CORAL_REEF = 4             # Warm, shallow tropical waters suitable for coral
    
    # Land biomes (positive elevations)
    ICE = 10                         # Permanent ice and snow
    TUNDRA = 11                      # Arctic/alpine tundra
    COLD_DESERT = 12                 # Cold, dry regions
    BOREAL_FOREST = 13               # Taiga/boreal forest
    TEMPERATE_RAINFOREST = 14        # Cool, wet forests (Pacific Northwest style)
    TEMPERATE_DECIDUOUS_FOREST = 15  # Classic temperate forest
    TEMPERATE_GRASSLAND = 16         # Prairies, steppes
    MEDITERRANEAN = 17               # Mediterranean climate (dry summer/wet winter)
    HOT_DESERT = 18                  # Hot, arid deserts
    SAVANNA = 19                     # Tropical grasslands with scattered trees
    TROPICAL_SEASONAL_FOREST = 20    # Tropical forests with dry season
    TROPICAL_RAINFOREST = 21         # Tropical rainforest
    ALPINE = 22                      # High elevation, above treeline
    MANGROVE = 23                    # Coastal tropical wetlands
    
class FaunaCategory(IntEnum):
    """Wildlife categories for fauna distribution"""
    HERBIVORE_GRAZER = 0      # Grassland grazers (deer, antelope, cattle)
    HERBIVORE_BROWSER = 1     # Forest browsers (moose, elk, rabbits)
    HERBIVORE_MIXED = 2        # Mixed feeders (bears, boars)
    PREDATOR_SMALL = 3         # Small predators (foxes, wildcats, weasels)
    PREDATOR_MEDIUM = 4        # Medium predators (wolves, lynx, coyotes)
    PREDATOR_APEX = 5          # Apex predators (bears, big cats, eagles)
    OMNIVORE = 6               # Omnivorous species (boars, raccoons)
    AQUATIC_FISH = 7           # Fish populations
    AQUATIC_AMPHIBIAN = 8      # Frogs, salamanders
    AQUATIC_WATERFOWL = 9      # Ducks, geese, herons
    AVIAN_RAPTOR = 10          # Birds of prey
    AVIAN_SONGBIRD = 11        # Small perching birds
    AVIAN_MIGRATORY = 12       # Long-distance migrants
    INSECT = 13                # Pollinators and decomposers (abstracted)

class TimberType(IntEnum):
    """Types of harvestable timber"""
    NONE = 0                   # No commercial timber
    SOFTWOOD = 1               # Pine, spruce, fir (construction lumber)
    HARDWOOD = 2               # Oak, maple, walnut (furniture, tools)
    TROPICAL_HARDWOOD = 3      # Mahogany, teak (high-value exotic wood)


class QuarryType(IntEnum):
    """Types of quarriable building stone"""
    NONE = 0
    SANDSTONE = 1              # Sedimentary, moderate quality
    LIMESTONE = 2              # Good general building stone
    GRANITE = 3                # Igneous, excellent durability
    MARBLE = 4                 # Metamorphic, decorative high-value stone

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


class ElementalAffinity(IntEnum):
    """Elemental magic types based on terrain"""
    NONE = 0
    FIRE = 1      # Deserts, volcanoes, hot areas
    WATER = 2     # Coasts, rivers, wet areas
    EARTH = 3     # Mountains, caves, forests
    AIR = 4       # High peaks, windy areas
    ARCANE = 5    # Mixed/neutral magic


class EnchantedLocationType(str, Enum):
    """Types of enchanted locations"""
    MANA_WELL = "mana_well"              # Ley line convergence points
    FEY_GROVE = "fey_grove"              # Ancient magical forests
    DRAGON_LAIR = "dragon_lair"          # Mountain caves with elemental power
    CRYSTAL_CAVERN = "crystal_cavern"    # Underground magic amplifiers
    CORRUPTED_SITE = "corrupted_site"    # Magically warped zones

class RoadType(IntEnum):
    """Types of roads in the network"""
    IMPERIAL_HIGHWAY = 0  # Major highways connecting metropolises (paved, wide)
    MAIN_ROAD = 1         # Roads connecting cities (gravel/cobblestone)
    RURAL_ROAD = 2        # Roads connecting towns (dirt, well-maintained)
    PATH = 3              # Paths connecting villages (dirt, basic)
    TRAIL = 4             # Trails connecting hamlets (unpaved, narrow)

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
    num_plates: int = Field(12, ge=4, le=1000, description="Number of tectonic plates")
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
    "pass_12_biomes",
    "pass_13_fauna",
    "pass_14_resources",
    "pass_15_magic",
    "pass_16_settlements",
    "pass_17_roads",
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
    "pass_12_biomes": 7,
    "pass_13_fauna": 6,
    "pass_14_resources": 5,
    "pass_15_magic": 6,
    "pass_16_settlements": 7,
    "pass_17_roads": 8,
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

# Road travel speeds (km/hour)
ROAD_SPEEDS = {
    RoadType.IMPERIAL_HIGHWAY: 30,  # Fast travel on paved roads
    RoadType.MAIN_ROAD: 25,          # Good travel on cobblestone
    RoadType.RURAL_ROAD: 20,         # Moderate travel on dirt roads
    RoadType.PATH: 15,               # Slow travel on paths
    RoadType.TRAIL: 12,              # Very slow travel on trails
}

# Road construction costs (abstract units)
ROAD_COSTS = {
    RoadType.IMPERIAL_HIGHWAY: 100,
    RoadType.MAIN_ROAD: 50,
    RoadType.RURAL_ROAD: 20,
    RoadType.PATH: 10,
    RoadType.TRAIL: 5,
}