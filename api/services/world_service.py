"""
World Service
Handles world generation, loading, and data access.
Integrates with builder/ for generation and Supabase for persistence.
"""

import logging
import json
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

from api.models.requests import CreateWorldRequest
from api.models.responses import (
    WorldResponse,
    ChunkResponse,
    TileData,
    GenerationProgressResponse
)
from api.services.generation_runner import generation_runner
from api.services.supabase_client import get_supabase_client, SupabaseClient

logger = logging.getLogger(__name__)

# Chunk size for database storage
# NOTE: Existing worlds in DB were generated with CHUNK_SIZE=32
# New worlds should use 256, but we need 32 to read existing data
CHUNK_SIZE = 32


class WorldService:
    """
    Service for world management.

    Integrates with:
    - builder/ for world generation
    - Supabase for persistence
    """

    def __init__(self):
        # Cache of active worlds (world_id -> world data)
        self._worlds: Dict[str, Dict[str, Any]] = {}
        # Supabase client
        self._db: Optional[SupabaseClient] = None

    def _get_db(self) -> Optional[SupabaseClient]:
        """Get Supabase client, initializing if needed."""
        if self._db is None:
            self._db = get_supabase_client()
        return self._db

    def set_supabase_client(self, client):
        """Set the Supabase client for persistence (legacy method)."""
        self._db = client

    async def create_world(self, request: CreateWorldRequest) -> str:
        """
        Create a new world with the given parameters.

        Starts async generation and returns world_id for progress tracking.
        """
        world_id = str(uuid4())
        seed = request.seed or int(datetime.utcnow().timestamp() * 1000) % (2**31)

        # Size to dimensions mapping
        size_map = {
            "SMALL": 512,
            "MEDIUM": 1024,
            "LARGE": 2048,
            "HUGE": 4096
        }
        dimensions = size_map.get(request.size.value, 1024)

        # Initialize world record
        world_data = {
            "id": world_id,
            "name": request.name,
            "seed": seed,
            "size": request.size.value,
            "dimensions": dimensions,
            "created_at": datetime.utcnow(),
            "status": "generating",
            "config": {
                "planet_radius_km": request.planet_radius_km,
                "axial_tilt": request.axial_tilt,
                "ocean_percentage": request.ocean_percentage,
                "num_plates": request.num_plates,
                "enable_magic": request.enable_magic,
                "enable_caves": request.enable_caves,
                "settlement_density": request.settlement_density
            }
        }

        self._worlds[world_id] = world_data

        # Save to Supabase
        db = self._get_db()
        if db:
            try:
                await db.create_world(
                    world_id=world_id,
                    name=request.name,
                    seed=seed,
                    size=request.size.value,
                    generation_params=world_data["config"]
                )
                logger.info(f"World {world_id} saved to database")
            except Exception as e:
                logger.error(f"Failed to save world to DB: {e}")

        # Start async generation
        generation_params = {
            "seed": seed,
            "size": request.size.value,
            "planet_radius_km": request.planet_radius_km,
            "axial_tilt": request.axial_tilt,
            "ocean_percentage": request.ocean_percentage,
            "num_plates": request.num_plates,
            "enable_caves": request.enable_caves,
        }

        # Reference to self for callbacks
        service = self

        def on_complete(wid: str, world_state):
            """Called when generation completes."""
            logger.info(f"World {wid} generation complete")
            service._worlds[wid]["status"] = "ready"
            service._worlds[wid]["world_state"] = world_state

            # Extract stats
            if world_state:
                service._worlds[wid]["num_settlements"] = len(getattr(world_state, 'settlements', []))
                service._worlds[wid]["num_factions"] = len(getattr(world_state, 'factions', []))

            # Schedule async persistence
            asyncio.create_task(service._persist_world_data(wid, world_state))

        def on_error(wid: str, error: Exception):
            """Called when generation fails."""
            logger.error(f"World {wid} generation failed: {error}")
            service._worlds[wid]["status"] = "failed"
            service._worlds[wid]["error"] = str(error)

            # Update status in DB
            db = service._get_db()
            if db:
                asyncio.create_task(db.update_world_status(wid, "failed"))

        await generation_runner.start_generation(
            world_id=world_id,
            params=generation_params,
            on_complete=on_complete,
            on_error=on_error
        )

        logger.info(f"Started world generation: {world_id}")
        return world_id

    async def _persist_world_data(self, world_id: str, world_state):
        """Persist generated world data to database."""
        db = self._get_db()
        if not db or not world_state:
            return

        try:
            # Update world status
            await db.update_world_status(world_id, "ready", progress=100.0)

            # Save settlements
            settlements = getattr(world_state, 'settlements', [])
            if settlements:
                settlement_data = []
                for s in settlements:
                    x = getattr(s, 'x', 0)
                    y = getattr(s, 'y', 0)
                    population = getattr(s, 'population', 100)
                    settlement_data.append({
                        "id": str(getattr(s, 'id', uuid4())),
                        "name": str(getattr(s, 'name', 'Unknown')),
                        "type": str(getattr(s, 'settlement_type', 'village')),
                        "x": int(x) if hasattr(x, 'item') else int(x),
                        "y": int(y) if hasattr(y, 'item') else int(y),
                        "population": int(population) if hasattr(population, 'item') else int(population),
                        "faction_id": str(getattr(s, 'faction_id', '')) if getattr(s, 'faction_id', None) else None,
                    })
                await db.save_settlements_batch(world_id, settlement_data)
                logger.info(f"Saved {len(settlement_data)} settlements for world {world_id}")

            # Save factions
            factions = getattr(world_state, 'factions', [])
            if factions:
                faction_data = []
                for f in factions:
                    faction_data.append({
                        "id": str(getattr(f, 'id', uuid4())),
                        "name": str(getattr(f, 'name', 'Unknown')),
                        "type": str(getattr(f, 'faction_type', 'kingdom')),
                        "color": getattr(f, 'color', None),
                        "center_x": getattr(f, 'territory_center_x', None),
                        "center_y": getattr(f, 'territory_center_y', None),
                        "radius": getattr(f, 'territory_radius', None),
                    })
                await db.save_factions_batch(world_id, faction_data)
                logger.info(f"Saved {len(faction_data)} factions for world {world_id}")

            # Save tile chunks
            dimensions = self._worlds[world_id].get("dimensions", 1024)
            num_chunks = dimensions // CHUNK_SIZE
            chunks_saved = 0

            # Save in batches to avoid overwhelming the database
            batch_size = 100
            batch = []

            for cx in range(num_chunks):
                for cy in range(num_chunks):
                    chunk = world_state.get_chunk(cx, cy)
                    if chunk:
                        tiles = self._extract_chunk_tiles(chunk, cx, cy)
                        batch.append({
                            "chunk_x": cx,
                            "chunk_y": cy,
                            "tiles": tiles
                        })
                        chunks_saved += 1

                        if len(batch) >= batch_size:
                            await db.save_chunks_batch(world_id, batch)
                            batch = []

            # Save remaining batch
            if batch:
                await db.save_chunks_batch(world_id, batch)

            logger.info(f"Saved {chunks_saved} chunks for world {world_id}")

        except Exception as e:
            logger.error(f"Failed to persist world data: {e}")

    def _extract_chunk_tiles(self, chunk, chunk_x: int, chunk_y: int) -> List[Dict[str, Any]]:
        """Extract tile data from a chunk for storage."""
        tiles = []
        biome_types = getattr(chunk, 'biome_type', None)
        elevations = getattr(chunk, 'elevation', None)
        temperatures = getattr(chunk, 'temperature_c', None)
        roads = getattr(chunk, 'road_presence', None)
        rivers = getattr(chunk, 'river_presence', None)

        if biome_types is None:
            return tiles

        for local_x in range(CHUNK_SIZE):
            for local_y in range(CHUNK_SIZE):
                try:
                    tile = {
                        "lx": local_x,
                        "ly": local_y,
                        "b": int(biome_types[local_x, local_y]) if biome_types is not None else 8,
                        "e": float(elevations[local_x, local_y]) if elevations is not None else 0,
                    }
                    # Only include optional fields if non-default
                    if temperatures is not None:
                        tile["t"] = round(float(temperatures[local_x, local_y]), 1)
                    if roads is not None and roads[local_x, local_y]:
                        tile["r"] = True
                    if rivers is not None and rivers[local_x, local_y]:
                        tile["v"] = True
                    tiles.append(tile)
                except (IndexError, TypeError):
                    pass
        return tiles

    async def get_world(self, world_id: str) -> Optional[WorldResponse]:
        """Get world details by ID."""
        # Check cache first
        world = self._worlds.get(world_id)

        if not world:
            # Try to load from database
            db = self._get_db()
            if db:
                db_world = await db.get_world(world_id)
                if db_world:
                    # Convert DB record to our internal format
                    size = db_world.get("size", "MEDIUM")
                    dimensions = {"SMALL": 512, "MEDIUM": 1024, "LARGE": 2048, "HUGE": 4096}.get(size, 1024)
                    world = {
                        "id": db_world.get("world_id"),
                        "name": db_world.get("name") or f"World {world_id[:8]}",
                        "seed": db_world.get("seed", 0),
                        "size": size,
                        "dimensions": dimensions,
                        "created_at": db_world.get("created_at"),
                        "status": db_world.get("status", "ready"),
                        "config": db_world.get("generation_params", {}),
                    }
                    self._worlds[world_id] = world

        if not world:
            return None

        # Calculate dimensions from size
        dimensions = world.get("dimensions", 1024)
        if isinstance(world.get("size"), str):
            dimensions = {"SMALL": 512, "MEDIUM": 1024, "LARGE": 2048, "HUGE": 4096}.get(world["size"], 1024)

        return WorldResponse(
            world_id=world["id"],
            name=world.get("name") or f"World {world['id'][:8]}",
            seed=world.get("seed", 0),
            size=world.get("size", "MEDIUM"),
            created_at=world.get("created_at") or datetime.utcnow(),
            status=world.get("status", "unknown"),
            num_settlements=world.get("num_settlements", 0),
            num_agents=world.get("num_agents", 0),
            num_tiles=dimensions ** 2
        )

    async def list_worlds(self, limit: int = 20, offset: int = 0) -> List[WorldResponse]:
        """List all available worlds from database and cache."""
        worlds_list = []

        # First get from database
        db = self._get_db()
        if db:
            db_worlds = await db.list_worlds(limit=limit)
            for db_world in db_worlds:
                world_id = db_world.get("world_id")
                size = db_world.get("size", "MEDIUM")
                dimensions = {"SMALL": 512, "MEDIUM": 1024, "LARGE": 2048, "HUGE": 4096}.get(size, 1024)

                world = {
                    "id": world_id,
                    "name": db_world.get("name") or f"World {world_id[:8]}",
                    "seed": db_world.get("seed", 0),
                    "size": size,
                    "dimensions": dimensions,
                    "created_at": db_world.get("created_at"),
                    "status": db_world.get("status", "ready"),
                    "num_settlements": 0,  # Would need to query settlements table
                    "num_agents": 0,
                }
                worlds_list.append(world)

                # Update cache with DB data
                if world_id not in self._worlds:
                    self._worlds[world_id] = world

        # Also include any in-memory worlds not in DB (currently generating)
        for world_id, world in self._worlds.items():
            if not any(w["id"] == world_id for w in worlds_list):
                worlds_list.append(world)

        # Apply offset and limit
        worlds_list = worlds_list[offset:offset + limit]

        return [
            WorldResponse(
                world_id=w["id"],
                name=w.get("name") or f"World {w['id'][:8]}",
                seed=w.get("seed", 0),
                size=w.get("size", "MEDIUM"),
                created_at=w.get("created_at") or datetime.utcnow(),
                status=w.get("status", "unknown"),
                num_settlements=w.get("num_settlements", 0),
                num_agents=w.get("num_agents", 0),
                num_tiles=w.get("dimensions", 1024) ** 2
            )
            for w in worlds_list
        ]

    async def delete_world(self, world_id: str) -> bool:
        """Delete a world from cache and database."""
        deleted = False

        # Cancel any ongoing generation
        generation_runner.cancel_generation(world_id)
        generation_runner.cleanup(world_id)

        # Remove from cache
        if world_id in self._worlds:
            del self._worlds[world_id]
            deleted = True

        # Delete from database
        db = self._get_db()
        if db:
            try:
                await db.delete_world(world_id)
                deleted = True
                logger.info(f"Deleted world {world_id} from database")
            except Exception as e:
                logger.error(f"Failed to delete world from DB: {e}")

        return deleted

    async def get_generation_status(self, world_id: str) -> Optional[GenerationProgressResponse]:
        """Get world generation progress."""
        progress = generation_runner.get_progress(world_id)
        if not progress:
            # Check if world exists but generation not tracked
            world = self._worlds.get(world_id)
            if world:
                return GenerationProgressResponse(
                    world_id=world_id,
                    status=world.get("status", "unknown"),
                    current_pass=18 if world.get("status") == "ready" else 0,
                    total_passes=18,
                    pass_name="Complete" if world.get("status") == "ready" else "Unknown",
                    progress_percent=100.0 if world.get("status") == "ready" else 0.0
                )
            return None

        return GenerationProgressResponse(
            world_id=world_id,
            status=progress["status"],
            current_pass=progress["current_pass_number"],
            total_passes=progress["total_passes"],
            pass_name=progress["current_pass"] or "Initializing",
            progress_percent=progress["progress_percent"]
        )

    async def get_generation_progress(self, world_id: str) -> Optional[GenerationProgressResponse]:
        """Alias for get_generation_status."""
        return await self.get_generation_status(world_id)

    async def get_chunk(
        self,
        world_id: str,
        chunk_x: int,
        chunk_y: int
    ) -> Optional[Dict[str, Any]]:
        """Get a specific chunk from the generated world."""
        world = self._worlds.get(world_id)
        if not world:
            return None

        world_state = world.get("world_state")
        if not world_state:
            return None

        # Get chunk from world state
        chunk = world_state.get_chunk(chunk_x, chunk_y)
        if not chunk:
            return None

        return self._chunk_to_dict(chunk, chunk_x, chunk_y)

    def _chunk_to_dict(self, chunk, chunk_x: int, chunk_y: int) -> Dict[str, Any]:
        """Convert a WorldChunk to a serializable dictionary."""
        def array_to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        return {
            "chunk_x": chunk_x,
            "chunk_y": chunk_y,
            "elevation": array_to_list(getattr(chunk, 'elevation', None)),
            "temperature_c": array_to_list(getattr(chunk, 'temperature_c', None)),
            "precipitation_mm": array_to_list(getattr(chunk, 'precipitation_mm', None)),
            "biome_type": array_to_list(getattr(chunk, 'biome_type', None)),
            "vegetation_density": array_to_list(getattr(chunk, 'vegetation_density', None)),
            "river_presence": array_to_list(getattr(chunk, 'river_presence', None)),
            "road_presence": array_to_list(getattr(chunk, 'road_presence', None)),
            "mana_concentration": array_to_list(getattr(chunk, 'mana_concentration', None)),
            "faction_territory": array_to_list(getattr(chunk, 'faction_territory', None)),
        }

    async def get_chunks_in_range(
        self,
        world_id: str,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int
    ) -> List[Dict[str, Any]]:
        """Get tile data for a coordinate range.

        Returns individual tile data for rendering, not chunk metadata.
        Loads from memory if available, otherwise from database.
        """
        world = self._worlds.get(world_id)
        if not world:
            # Try to load world from DB
            await self.get_world(world_id)
            world = self._worlds.get(world_id)
            if not world:
                return []

        world_state = world.get("world_state")

        # If we have world state in memory, use it
        if world_state:
            return self._get_tiles_from_world_state(world_state, x_min, x_max, y_min, y_max)

        # Otherwise, try to load from database
        db = self._get_db()
        if db:
            tiles = await self._get_tiles_from_database(db, world_id, x_min, x_max, y_min, y_max)
            if tiles:
                return tiles

        # Return placeholder tiles if no data available
        return self._get_placeholder_tiles(x_min, x_max, y_min, y_max)

    def _get_tiles_from_world_state(
        self,
        world_state,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int
    ) -> List[Dict[str, Any]]:
        """Extract tile data from in-memory world state."""
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                try:
                    location_data = world_state.query_location(x, y)
                    tiles.append({
                        "x": x,
                        "y": y,
                        "elevation": float(location_data.get("elevation", 0)),
                        "biome_type": self._biome_enum_to_name(location_data.get("biome_type", 8)),
                        "temperature_c": float(location_data.get("temperature_c", 15.0)),
                        "has_road": bool(location_data.get("road_presence", False)),
                        "has_river": bool(location_data.get("river_presence", False)),
                        "settlement_id": location_data.get("settlement_id"),
                        "settlement_type": location_data.get("settlement_type"),
                        "faction_name": location_data.get("faction_name"),
                        "resource_type": location_data.get("resource_type"),
                    })
                except Exception as e:
                    logger.debug(f"Failed to query tile ({x}, {y}): {e}")
                    tiles.append(self._default_tile(x, y))
        return tiles

    async def _get_tiles_from_database(
        self,
        db: SupabaseClient,
        world_id: str,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int
    ) -> List[Dict[str, Any]]:
        """Load tile data from database chunks."""
        try:
            # Calculate which chunks we need
            chunk_x_min = x_min // CHUNK_SIZE
            chunk_x_max = x_max // CHUNK_SIZE
            chunk_y_min = y_min // CHUNK_SIZE
            chunk_y_max = y_max // CHUNK_SIZE

            # Fetch chunks from database
            chunks = await db.get_chunks_in_range(
                world_id, chunk_x_min, chunk_x_max, chunk_y_min, chunk_y_max
            )

            if not chunks:
                return []

            # Build a lookup for chunk data
            chunk_lookup = {}
            for chunk in chunks:
                key = (chunk.get("chunk_x"), chunk.get("chunk_y"))
                chunk_lookup[key] = chunk.get("tiles", [])

            # Extract tiles in the requested range
            tiles = []
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    chunk_x = x // CHUNK_SIZE
                    chunk_y = y // CHUNK_SIZE
                    local_x = x % CHUNK_SIZE
                    local_y = y % CHUNK_SIZE

                    chunk_tiles = chunk_lookup.get((chunk_x, chunk_y), [])

                    # Find the tile in the chunk data
                    tile_data = None
                    for t in chunk_tiles:
                        if t.get("lx") == local_x and t.get("ly") == local_y:
                            tile_data = t
                            break

                    if tile_data:
                        tiles.append({
                            "x": x,
                            "y": y,
                            "elevation": tile_data.get("e", 0),
                            "biome_type": self._biome_enum_to_name(tile_data.get("b", 8)),
                            "temperature_c": tile_data.get("t", 15.0),
                            "has_road": tile_data.get("r", False),
                            "has_river": tile_data.get("v", False),
                            "settlement_id": None,
                            "settlement_type": None,
                            "faction_name": None,
                            "resource_type": None,
                        })
                    else:
                        tiles.append(self._default_tile(x, y))

            return tiles
        except Exception as e:
            logger.error(f"Failed to load tiles from database: {e}")
            return []

    def _get_placeholder_tiles(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int
    ) -> List[Dict[str, Any]]:
        """Generate placeholder tiles when no data is available."""
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append(self._default_tile(x, y))
        return tiles

    def _default_tile(self, x: int, y: int) -> Dict[str, Any]:
        """Create a default tile."""
        return {
            "x": x,
            "y": y,
            "elevation": 0,
            "biome_type": "temperate_grassland",
            "temperature_c": 15.0,
            "has_road": False,
            "has_river": False,
            "settlement_id": None,
            "settlement_type": None,
            "faction_name": None,
            "resource_type": None,
        }

    def _biome_enum_to_name(self, biome_value) -> str:
        """Convert biome enum/int to string name."""
        if isinstance(biome_value, str):
            return biome_value

        biome_names = {
            0: "ocean_trench", 1: "ocean_deep", 2: "ocean_shallow",
            3: "ocean_shelf", 4: "coral_reef", 5: "ice",
            6: "tundra", 7: "boreal_forest", 8: "temperate_forest",
            9: "temperate_rainforest", 10: "tropical_rainforest",
            11: "savanna", 12: "grassland", 13: "desert",
            14: "alpine", 15: "wetland", 16: "mangrove"
        }
        try:
            return biome_names.get(int(biome_value), "temperate_grassland")
        except (ValueError, TypeError):
            return "temperate_grassland"

    def _get_primary_biome(self, chunk) -> str:
        """Get the most common biome in a chunk."""
        biome_types = getattr(chunk, 'biome_type', None)
        if biome_types is None:
            return "unknown"

        # Find most common biome
        unique, counts = np.unique(biome_types, return_counts=True)
        if len(unique) == 0:
            return "unknown"

        most_common = unique[np.argmax(counts)]

        # Map biome enum to string
        biome_names = {
            0: "ocean_trench", 1: "ocean_deep", 2: "ocean_shallow",
            3: "ocean_shelf", 4: "coral_reef", 5: "ice",
            6: "tundra", 7: "boreal_forest", 8: "temperate_forest",
            9: "temperate_rainforest", 10: "tropical_rainforest",
            11: "savanna", 12: "grassland", 13: "desert",
            14: "alpine", 15: "wetland", 16: "mangrove"
        }
        return biome_names.get(int(most_common), f"biome_{most_common}")

    async def get_tile_data(
        self,
        world_id: str,
        x: int,
        y: int
    ) -> Optional[TileData]:
        """Get data for a specific tile."""
        world = self._worlds.get(world_id)
        if not world:
            return None

        world_state = world.get("world_state")
        if not world_state:
            return None

        # Query location from world state
        try:
            location_data = world_state.query_location(x, y)
            return TileData(
                x=x,
                y=y,
                biome=location_data.get("biome_type", "unknown"),
                elevation=location_data.get("elevation", 0.0),
                temperature=location_data.get("temperature_c", 15.0),
                moisture=location_data.get("precipitation_mm", 0.0) / 3000.0,  # Normalize
                is_water=location_data.get("elevation", 0) < 0,
                river=location_data.get("river_presence", False),
                magic_level=location_data.get("mana_concentration", 0.0)
            )
        except Exception as e:
            logger.error(f"Failed to query location ({x}, {y}): {e}")
            return None

    async def get_settlements(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all settlements in a world from memory or database."""
        world = self._worlds.get(world_id)

        # Try from in-memory world state first
        if world:
            world_state = world.get("world_state")
            if world_state:
                settlements = getattr(world_state, 'settlements', [])
                result = []
                for s in settlements:
                    x = getattr(s, 'x', 0)
                    y = getattr(s, 'y', 0)
                    population = getattr(s, 'population', 0)
                    specialization = getattr(s, 'specialization', None)

                    result.append({
                        "id": str(getattr(s, 'id', uuid4())),
                        "name": str(getattr(s, 'name', 'Unknown')),
                        "type": str(getattr(s, 'settlement_type', 'village')),
                        "x": int(x) if hasattr(x, 'item') else int(x),
                        "y": int(y) if hasattr(y, 'item') else int(y),
                        "population": int(population) if hasattr(population, 'item') else int(population),
                        "specialization": str(specialization) if specialization is not None else None,
                    })
                return result

        # Try loading from database
        db = self._get_db()
        if db:
            try:
                db_settlements = await db.get_settlements(world_id)
                return [
                    {
                        "id": s.get("settlement_id"),
                        "name": s.get("name", "Unknown"),
                        "type": s.get("type", "village"),
                        "x": s.get("x", 0),
                        "y": s.get("y", 0),
                        "population": s.get("population", 0),
                        "specialization": None,
                        "faction_id": s.get("faction_id"),
                    }
                    for s in db_settlements
                ]
            except Exception as e:
                logger.error(f"Failed to load settlements from DB: {e}")

        return []

    async def get_factions(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all factions in a world from memory or database."""
        world = self._worlds.get(world_id)

        # Try from in-memory world state first
        if world:
            world_state = world.get("world_state")
            if world_state:
                factions = getattr(world_state, 'factions', [])
                return [
                    {
                        "id": str(getattr(f, 'id', uuid4())),
                        "name": getattr(f, 'name', 'Unknown'),
                        "type": str(getattr(f, 'faction_type', 'kingdom')),
                        "capital_id": str(getattr(f, 'capital_id', '')),
                        "territory_size": getattr(f, 'territory_size', 0),
                        "power_level": getattr(f, 'power_level', 0.5),
                    }
                    for f in factions
                ]

        # Try loading from database
        db = self._get_db()
        if db:
            try:
                db_factions = await db.get_factions(world_id)
                return [
                    {
                        "id": f.get("faction_id"),
                        "name": f.get("name", "Unknown"),
                        "type": f.get("type", "kingdom"),
                        "capital_id": None,
                        "territory_size": 0,
                        "power_level": 0.5,
                        "color": f.get("color"),
                        "center_x": f.get("center_x"),
                        "center_y": f.get("center_y"),
                    }
                    for f in db_factions
                ]
            except Exception as e:
                logger.error(f"Failed to load factions from DB: {e}")

        return []

    async def get_roads(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all roads in a world."""
        world = self._worlds.get(world_id)
        if not world:
            return []

        world_state = world.get("world_state")
        if not world_state:
            return []

        roads = getattr(world_state, 'road_network', [])
        return [
            {
                "id": str(getattr(r, 'id', uuid4())),
                "type": str(getattr(r, 'road_type', 'path')),
                "start_x": getattr(r, 'start_x', 0),
                "start_y": getattr(r, 'start_y', 0),
                "end_x": getattr(r, 'end_x', 0),
                "end_y": getattr(r, 'end_y', 0),
                "path": getattr(r, 'path', []),
            }
            for r in roads
        ]

    async def get_ley_lines(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all ley lines in a world."""
        world = self._worlds.get(world_id)
        if not world:
            return []

        world_state = world.get("world_state")
        if not world_state:
            return []

        ley_lines = getattr(world_state, 'ley_line_network', [])
        return [
            {
                "id": str(getattr(l, 'id', uuid4())),
                "start_x": getattr(l, 'start_x', 0),
                "start_y": getattr(l, 'start_y', 0),
                "end_x": getattr(l, 'end_x', 0),
                "end_y": getattr(l, 'end_y', 0),
                "strength": getattr(l, 'strength', 1.0),
            }
            for l in ley_lines
        ]
