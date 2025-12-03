"""
World Interface
Provides bridge between agent reasoning and world builder data.
Allows agents to query the procedurally generated world.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from uuid import UUID
import sys
import os

# Add builder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'builder'))

from pydantic import BaseModel, Field


@dataclass
class LocationData:
    """Complete environmental data at a world location"""
    x: int
    y: int
    chunk_x: int
    chunk_y: int

    # Terrain
    elevation: Optional[float] = None  # meters
    biome_type: Optional[str] = None

    # Climate
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[int] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None  # degrees

    # Hydrology
    has_water: bool = False
    water_depth: Optional[float] = None
    river_flow: Optional[float] = None  # m³/s

    # Soil & Vegetation
    soil_type: Optional[str] = None
    vegetation_density: Optional[float] = None
    agricultural_suitability: Optional[float] = None

    # Resources
    timber_quality: Optional[float] = None
    timber_type: Optional[str] = None
    mineral_deposits: Optional[Dict[str, float]] = None
    fishing_quality: Optional[float] = None

    # Magic
    mana_concentration: Optional[float] = None
    ley_line_present: bool = False
    elemental_affinity: Optional[str] = None
    corrupted: bool = False

    # Infrastructure
    has_road: bool = False
    road_type: Optional[str] = None
    settlement_id: Optional[int] = None
    settlement_type: Optional[str] = None

    # Politics
    faction_id: Optional[int] = None
    faction_name: Optional[str] = None
    is_contested: bool = False


@dataclass
class PathResult:
    """Result of pathfinding between two points"""
    success: bool
    path: List[Tuple[int, int]]  # List of (x, y) coordinates
    total_distance: float  # in world units
    estimated_travel_time: float  # in hours
    uses_roads: bool
    terrain_difficulty: float  # 0-1 scale
    crosses_water: bool
    crosses_borders: bool


@dataclass
class AgentSummary:
    """Summary of another agent visible from a location"""
    agent_id: UUID
    name: str
    agent_type: str
    position_x: int
    position_y: int
    distance: float
    is_alive: bool


@dataclass
class ResourceAvailability:
    """Resource availability at a location"""
    resource_type: str
    available: bool
    quantity: float  # 0-1 scale
    quality: float  # 0-1 scale
    extraction_difficulty: float  # 0-1 scale
    required_skill: Optional[str] = None
    required_tools: Optional[List[str]] = None


class WorldInterface:
    """
    Bridge between agent reasoning and world data.
    Provides query methods for agents to perceive and interact with the world.
    """

    def __init__(self, world_state=None):
        """
        Initialize world interface.

        Args:
            world_state: Optional WorldState from builder. If None, queries will
                        need a world_state passed to each method.
        """
        self._world_state = world_state
        self._road_type_names = {
            0: "imperial_highway",
            1: "main_road",
            2: "rural_road",
            3: "path",
            4: "trail"
        }
        self._biome_names = {
            0: "ocean_trench", 1: "ocean_deep", 2: "ocean_shallow",
            3: "ocean_shelf", 4: "ocean_coral_reef",
            10: "ice", 11: "tundra", 12: "cold_desert", 13: "boreal_forest",
            14: "temperate_rainforest", 15: "temperate_deciduous_forest",
            16: "temperate_grassland", 17: "mediterranean", 18: "hot_desert",
            19: "savanna", 20: "tropical_seasonal_forest", 21: "tropical_rainforest",
            22: "alpine", 23: "mangrove"
        }
        self._elemental_names = {
            0: "none", 1: "fire", 2: "water", 3: "earth", 4: "air", 5: "arcane"
        }
        self._timber_names = {
            0: "none", 1: "softwood", 2: "hardwood", 3: "tropical_hardwood"
        }
        self._soil_names = {
            0: "sand", 1: "sandy_loam", 2: "loam", 3: "silt_loam",
            4: "clay_loam", 5: "sandy_clay_loam", 6: "silty_clay_loam",
            7: "clay", 8: "silty_clay", 9: "sandy_clay"
        }

    def set_world_state(self, world_state) -> None:
        """Set or update the world state"""
        self._world_state = world_state

    def query_location(self, x: int, y: int, world_state=None) -> Optional[LocationData]:
        """
        Get full environmental data at coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate
            world_state: Optional world state (uses instance state if not provided)

        Returns:
            LocationData with all available information, or None if location invalid
        """
        ws = world_state or self._world_state
        if ws is None:
            return None

        chunk_x, chunk_y = ws.get_chunk_for_location(x, y)
        chunk = ws.get_chunk(chunk_x, chunk_y)

        if chunk is None:
            return None

        # Convert to local chunk coordinates
        local_x = x % 256
        local_y = y % 256

        # Build location data
        loc = LocationData(
            x=x, y=y,
            chunk_x=chunk_x, chunk_y=chunk_y
        )

        # Terrain
        if chunk.elevation is not None:
            loc.elevation = float(chunk.elevation[local_x, local_y])

        if chunk.biome_type is not None:
            biome_id = int(chunk.biome_type[local_x, local_y])
            loc.biome_type = self._biome_names.get(biome_id, f"biome_{biome_id}")

        # Climate
        if chunk.temperature_c is not None:
            loc.temperature_c = float(chunk.temperature_c[local_x, local_y])

        if chunk.precipitation_mm is not None:
            loc.precipitation_mm = int(chunk.precipitation_mm[local_x, local_y])

        if chunk.wind_speed is not None:
            loc.wind_speed = float(chunk.wind_speed[local_x, local_y])

        if chunk.wind_direction is not None:
            loc.wind_direction = int(chunk.wind_direction[local_x, local_y])

        # Hydrology
        if chunk.river_presence is not None:
            loc.has_water = bool(chunk.river_presence[local_x, local_y])

        if chunk.river_flow is not None:
            loc.river_flow = float(chunk.river_flow[local_x, local_y])

        if chunk.water_table_depth is not None:
            loc.water_depth = float(chunk.water_table_depth[local_x, local_y])

        # Soil & Vegetation
        if chunk.soil_type is not None:
            soil_id = int(chunk.soil_type[local_x, local_y])
            loc.soil_type = self._soil_names.get(soil_id, f"soil_{soil_id}")

        if chunk.vegetation_density is not None:
            loc.vegetation_density = float(chunk.vegetation_density[local_x, local_y])

        if chunk.agricultural_suitability is not None:
            loc.agricultural_suitability = float(chunk.agricultural_suitability[local_x, local_y])

        # Resources
        if chunk.timber_quality is not None:
            loc.timber_quality = float(chunk.timber_quality[local_x, local_y])

        if chunk.timber_type is not None:
            timber_id = int(chunk.timber_type[local_x, local_y])
            loc.timber_type = self._timber_names.get(timber_id, f"timber_{timber_id}")

        if chunk.fishing_quality is not None:
            loc.fishing_quality = float(chunk.fishing_quality[local_x, local_y])

        if chunk.mineral_deposits is not None:
            loc.mineral_deposits = {
                mineral.name: float(arr[local_x, local_y])
                for mineral, arr in chunk.mineral_deposits.items()
            }

        # Magic
        if chunk.mana_concentration is not None:
            loc.mana_concentration = float(chunk.mana_concentration[local_x, local_y])

        if chunk.ley_line_presence is not None:
            loc.ley_line_present = bool(chunk.ley_line_presence[local_x, local_y])

        if chunk.elemental_affinity is not None:
            elem_id = int(chunk.elemental_affinity[local_x, local_y])
            loc.elemental_affinity = self._elemental_names.get(elem_id, f"element_{elem_id}")

        if chunk.corrupted_zone is not None:
            loc.corrupted = bool(chunk.corrupted_zone[local_x, local_y])

        # Infrastructure
        if chunk.road_presence is not None:
            loc.has_road = bool(chunk.road_presence[local_x, local_y])

        if chunk.road_type is not None and loc.has_road:
            road_id = int(chunk.road_type[local_x, local_y])
            loc.road_type = self._road_type_names.get(road_id, f"road_{road_id}")

        if chunk.settlement_presence is not None:
            settlement_val = int(chunk.settlement_presence[local_x, local_y])
            if settlement_val > 0:
                loc.settlement_id = settlement_val

        # Politics
        if chunk.faction_territory is not None:
            faction_val = int(chunk.faction_territory[local_x, local_y])
            if faction_val > 0:
                loc.faction_id = faction_val
                # Look up faction name if factions are loaded
                if ws.factions:
                    for faction in ws.factions:
                        if faction.faction_id == faction_val:
                            loc.faction_name = faction.name
                            break

        if chunk.contested_zone is not None:
            loc.is_contested = bool(chunk.contested_zone[local_x, local_y])

        return loc

    def query_radius(
        self,
        x: int,
        y: int,
        radius: int,
        world_state=None
    ) -> List[LocationData]:
        """
        Get area around agent for observation.

        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Radius in tiles to query
            world_state: Optional world state

        Returns:
            List of LocationData for all tiles in radius
        """
        locations = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Check if within circular radius
                if dx*dx + dy*dy <= radius*radius:
                    loc = self.query_location(x + dx, y + dy, world_state)
                    if loc:
                        locations.append(loc)
        return locations

    def query_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        world_state=None,
        prefer_roads: bool = True
    ) -> PathResult:
        """
        Calculate travel route between two points.
        Uses A* pathfinding with terrain costs.

        Args:
            start: Starting (x, y) coordinates
            end: Destination (x, y) coordinates
            world_state: Optional world state
            prefer_roads: Whether to prefer roads in pathfinding

        Returns:
            PathResult with path details
        """
        ws = world_state or self._world_state

        # For now, return a simple straight-line path
        # Full A* implementation would go in pathfinding.py
        import math

        start_x, start_y = start
        end_x, end_y = end

        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # Simple path interpolation
        steps = max(int(distance), 1)
        path = []
        for i in range(steps + 1):
            t = i / steps
            px = int(start_x + t * (end_x - start_x))
            py = int(start_y + t * (end_y - start_y))
            path.append((px, py))

        # Estimate travel time (assuming ~4 km/h walking speed, 1 tile = 100m)
        travel_time = (distance * 0.1) / 4.0  # hours

        return PathResult(
            success=True,
            path=path,
            total_distance=distance,
            estimated_travel_time=travel_time,
            uses_roads=False,  # TODO: check road presence
            terrain_difficulty=0.5,  # TODO: calculate from terrain
            crosses_water=False,  # TODO: detect
            crosses_borders=False  # TODO: detect
        )

    def query_nearby_agents(
        self,
        x: int,
        y: int,
        radius: int,
        world_id: UUID,
        exclude_agent_id: Optional[UUID] = None,
        supabase_client=None
    ) -> List[AgentSummary]:
        """
        Find other agents in perception range.
        Queries the database for agents near the given location.

        Args:
            x: Center X coordinate
            y: Center Y coordinate
            radius: Search radius in tiles
            world_id: World to search in
            exclude_agent_id: Agent ID to exclude (usually self)
            supabase_client: Supabase client for database queries

        Returns:
            List of AgentSummary for nearby agents
        """
        if supabase_client is None:
            return []

        # Use the get_nearby_agents database function
        result = supabase_client.rpc(
            'get_nearby_agents',
            {
                'p_world_id': str(world_id),
                'p_x': x,
                'p_y': y,
                'p_radius': radius
            }
        ).execute()

        agents = []
        for row in result.data or []:
            if exclude_agent_id and str(row['agent_id']) == str(exclude_agent_id):
                continue
            agents.append(AgentSummary(
                agent_id=UUID(row['agent_id']),
                name=row['name'],
                agent_type=row['agent_type'],
                position_x=row['position_x'],
                position_y=row['position_y'],
                distance=row['distance'],
                is_alive=True
            ))

        return agents

    def query_resources(
        self,
        x: int,
        y: int,
        resource_type: str,
        world_state=None
    ) -> ResourceAvailability:
        """
        Check resource availability for gathering/crafting.

        Args:
            x: World X coordinate
            y: World Y coordinate
            resource_type: Type of resource to query
            world_state: Optional world state

        Returns:
            ResourceAvailability with extraction details
        """
        loc = self.query_location(x, y, world_state)

        if loc is None:
            return ResourceAvailability(
                resource_type=resource_type,
                available=False,
                quantity=0,
                quality=0,
                extraction_difficulty=1.0
            )

        # Check different resource types
        if resource_type == "timber":
            return ResourceAvailability(
                resource_type="timber",
                available=(loc.timber_quality or 0) > 0.1,
                quantity=loc.vegetation_density or 0,
                quality=loc.timber_quality or 0,
                extraction_difficulty=0.3,
                required_skill="crafting.woodworking.logging",
                required_tools=["axe"]
            )

        elif resource_type == "fish":
            return ResourceAvailability(
                resource_type="fish",
                available=(loc.fishing_quality or 0) > 0.1 and loc.has_water,
                quantity=loc.fishing_quality or 0,
                quality=loc.fishing_quality or 0,
                extraction_difficulty=0.4,
                required_skill="survival.fishing",
                required_tools=["fishing_rod"]
            )

        elif resource_type == "crops":
            return ResourceAvailability(
                resource_type="crops",
                available=(loc.agricultural_suitability or 0) > 0.3,
                quantity=loc.agricultural_suitability or 0,
                quality=loc.agricultural_suitability or 0,
                extraction_difficulty=0.5,
                required_skill="labor.agriculture.farming",
                required_tools=["hoe", "seeds"]
            )

        elif resource_type in ["iron", "copper", "gold", "silver", "coal"]:
            mineral_value = 0
            if loc.mineral_deposits:
                mineral_value = loc.mineral_deposits.get(resource_type.upper(), 0)
            return ResourceAvailability(
                resource_type=resource_type,
                available=mineral_value > 0.1,
                quantity=mineral_value,
                quality=mineral_value,
                extraction_difficulty=0.7,
                required_skill="labor.mining",
                required_tools=["pickaxe"]
            )

        elif resource_type == "mana":
            return ResourceAvailability(
                resource_type="mana",
                available=(loc.mana_concentration or 0) > 0.2,
                quantity=loc.mana_concentration or 0,
                quality=loc.mana_concentration or 0,
                extraction_difficulty=0.8,
                required_skill="magic.channeling"
            )

        else:
            return ResourceAvailability(
                resource_type=resource_type,
                available=False,
                quantity=0,
                quality=0,
                extraction_difficulty=1.0
            )

    def get_settlement_info(
        self,
        settlement_id: int,
        world_state=None
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a settlement.

        Args:
            settlement_id: ID of the settlement
            world_state: Optional world state

        Returns:
            Dictionary with settlement details or None
        """
        ws = world_state or self._world_state
        if ws is None or ws.settlements is None:
            return None

        for settlement in ws.settlements:
            if settlement.settlement_id == settlement_id:
                return {
                    "id": settlement.settlement_id,
                    "name": getattr(settlement, 'name', f"Settlement {settlement_id}"),
                    "type": getattr(settlement, 'settlement_type', 'unknown'),
                    "population": getattr(settlement, 'population', 0),
                    "x": getattr(settlement, 'x', 0),
                    "y": getattr(settlement, 'y', 0),
                    "faction_id": getattr(settlement, 'faction_id', None)
                }

        return None

    def get_faction_info(
        self,
        faction_id: int,
        world_state=None
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a faction.

        Args:
            faction_id: ID of the faction
            world_state: Optional world state

        Returns:
            Dictionary with faction details or None
        """
        ws = world_state or self._world_state
        if ws is None or ws.factions is None:
            return None

        for faction in ws.factions:
            if faction.faction_id == faction_id:
                return {
                    "id": faction.faction_id,
                    "name": faction.name,
                    "type": faction.faction_type.name if hasattr(faction.faction_type, 'name') else str(faction.faction_type),
                    "capital_settlement_id": faction.capital_settlement_id,
                    "num_settlements": faction.num_settlements,
                    "total_population": faction.total_population,
                    "territory_size_km2": faction.territory_size_km2,
                    "allies": faction.allied_faction_ids,
                    "enemies": faction.enemy_faction_ids
                }

        return None
