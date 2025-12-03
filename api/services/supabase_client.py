"""
Supabase Client Service
Provides database access for persisting worlds, sessions, and agents.
"""

import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

from supabase import create_client, Client
from api.config import get_settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Singleton Supabase client for database operations.

    Provides methods for:
    - World persistence (metadata, chunks, settlements, factions)
    - Game session persistence
    - Agent persistence
    """

    _instance: Optional["SupabaseClient"] = None
    _client: Optional[Client] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            settings = get_settings()
            # Use service role key if available for full access (bypasses RLS)
            key = settings.supabase_service_role_key or settings.supabase_anon_key

            url = settings.supabase_url
            if url == "http://localhost:54321":
                url = "http://127.0.0.1:54321"

            # For local dev, use service role key to bypass RLS
            if not key or (url == "http://127.0.0.1:54321" and "service_role" not in (key or "")):
                # Default local dev service role key (bypasses RLS)
                key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
                logger.info("Using local dev service role key for database access")

            try:
                self._client = create_client(url, key)
                logger.info(f"Supabase client initialized: {url}")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self._client = None

    @property
    def client(self) -> Optional[Client]:
        return self._client

    def is_connected(self) -> bool:
        return self._client is not None

    # =========================================================================
    # World Operations
    # =========================================================================

    async def create_world(
        self,
        world_id: str,
        name: str,
        seed: int,
        size: str,
        generation_params: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new world record."""
        if not self._client:
            return None
        try:
            result = self._client.table("worlds").insert({
                "world_id": world_id,
                "name": name,
                "seed": seed,
                "size": size,
                "generation_params": generation_params or {},
                "status": "generating",
                "progress": 0
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create world: {e}")
            return None

    async def update_world_status(
        self,
        world_id: str,
        status: str,
        progress: float = None,
        current_pass: str = None
    ) -> bool:
        """Update world generation status."""
        if not self._client:
            return False
        try:
            update_data = {"status": status}
            if progress is not None:
                update_data["progress"] = progress
            if current_pass is not None:
                update_data["current_pass"] = current_pass

            self._client.table("worlds").update(update_data).eq(
                "world_id", world_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update world status: {e}")
            return False

    async def get_world(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get world by ID."""
        if not self._client:
            return None
        try:
            result = self._client.table("worlds").select("*").eq(
                "world_id", world_id
            ).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get world: {e}")
            return None

    async def list_worlds(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all worlds."""
        if not self._client:
            return []
        try:
            result = self._client.table("worlds").select("*").order(
                "created_at", desc=True
            ).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list worlds: {e}")
            return []

    async def delete_world(self, world_id: str) -> bool:
        """Delete a world and all related data."""
        if not self._client:
            return False
        try:
            # Cascading delete will handle related tables
            self._client.table("worlds").delete().eq(
                "world_id", world_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete world: {e}")
            return False

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    async def save_chunk(
        self,
        world_id: str,
        chunk_x: int,
        chunk_y: int,
        tiles: List[Dict[str, Any]]
    ) -> bool:
        """Save or update a chunk of tiles."""
        if not self._client:
            return False
        try:
            self._client.table("world_chunks").upsert({
                "world_id": world_id,
                "chunk_x": chunk_x,
                "chunk_y": chunk_y,
                "tiles": tiles,
                "updated_at": "now()"
            }, on_conflict="world_id,chunk_x,chunk_y").execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save chunk: {e}")
            return False

    async def save_chunks_batch(
        self,
        world_id: str,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple chunks in a batch."""
        if not self._client:
            return False
        try:
            records = []
            for chunk in chunks:
                records.append({
                    "world_id": world_id,
                    "chunk_x": chunk["chunk_x"],
                    "chunk_y": chunk["chunk_y"],
                    "tiles": chunk["tiles"]
                })

            if records:
                self._client.table("world_chunks").upsert(
                    records,
                    on_conflict="world_id,chunk_x,chunk_y"
                ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save chunks batch: {e}")
            return False

    async def get_chunk(
        self,
        world_id: str,
        chunk_x: int,
        chunk_y: int
    ) -> Optional[Dict[str, Any]]:
        """Get a specific chunk."""
        if not self._client:
            return None
        try:
            result = self._client.table("world_chunks").select("*").eq(
                "world_id", world_id
            ).eq("chunk_x", chunk_x).eq("chunk_y", chunk_y).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get chunk: {e}")
            return None

    async def get_chunks_in_range(
        self,
        world_id: str,
        chunk_x_min: int,
        chunk_x_max: int,
        chunk_y_min: int,
        chunk_y_max: int
    ) -> List[Dict[str, Any]]:
        """Get all chunks in a coordinate range."""
        if not self._client:
            return []
        try:
            result = self._client.table("world_chunks").select("*").eq(
                "world_id", world_id
            ).gte("chunk_x", chunk_x_min).lte("chunk_x", chunk_x_max).gte(
                "chunk_y", chunk_y_min
            ).lte("chunk_y", chunk_y_max).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get chunks in range: {e}")
            return []

    # =========================================================================
    # Settlement Operations
    # =========================================================================

    async def save_settlement(
        self,
        world_id: str,
        settlement_id: str,
        name: str,
        settlement_type: str,
        x: int,
        y: int,
        population: int = 100,
        faction_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Save a settlement."""
        if not self._client:
            return False
        try:
            self._client.table("settlements").upsert({
                "settlement_id": settlement_id,
                "world_id": world_id,
                "name": name,
                "settlement_type": settlement_type,
                "x": x,
                "y": y,
                "population": population,
                "faction_id": faction_id,
                "metadata": metadata or {}
            }, on_conflict="settlement_id").execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save settlement: {e}")
            return False

    async def save_settlements_batch(
        self,
        world_id: str,
        settlements: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple settlements in a batch."""
        if not self._client:
            return False
        try:
            records = []
            for s in settlements:
                records.append({
                    "settlement_id": s.get("id") or s.get("settlement_id"),
                    "world_id": world_id,
                    "name": s.get("name"),
                    "settlement_type": s.get("type") or s.get("settlement_type"),
                    "x": s.get("x"),
                    "y": s.get("y"),
                    "population": s.get("population", 100),
                    "faction_id": s.get("faction_id"),
                    "metadata": s.get("metadata", {})
                })

            if records:
                self._client.table("settlements").upsert(
                    records,
                    on_conflict="settlement_id"
                ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save settlements batch: {e}")
            return False

    async def get_settlements(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all settlements for a world."""
        if not self._client:
            return []
        try:
            result = self._client.table("settlements").select("*").eq(
                "world_id", world_id
            ).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get settlements: {e}")
            return []

    # =========================================================================
    # Faction Operations
    # =========================================================================

    async def save_faction(
        self,
        world_id: str,
        faction_id: str,
        name: str,
        faction_type: str = None,
        color: str = None,
        territory_center_x: int = None,
        territory_center_y: int = None,
        territory_radius: int = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Save a faction."""
        if not self._client:
            return False
        try:
            self._client.table("factions").upsert({
                "faction_id": faction_id,
                "world_id": world_id,
                "name": name,
                "faction_type": faction_type,
                "color": color,
                "territory_center_x": territory_center_x,
                "territory_center_y": territory_center_y,
                "territory_radius": territory_radius,
                "metadata": metadata or {}
            }, on_conflict="faction_id").execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save faction: {e}")
            return False

    async def save_factions_batch(
        self,
        world_id: str,
        factions: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple factions in a batch."""
        if not self._client:
            return False
        try:
            records = []
            for f in factions:
                records.append({
                    "faction_id": f.get("id") or f.get("faction_id"),
                    "world_id": world_id,
                    "name": f.get("name"),
                    "faction_type": f.get("type") or f.get("faction_type"),
                    "color": f.get("color"),
                    "territory_center_x": f.get("center_x") or f.get("territory_center_x"),
                    "territory_center_y": f.get("center_y") or f.get("territory_center_y"),
                    "territory_radius": f.get("radius") or f.get("territory_radius"),
                    "metadata": f.get("metadata", {})
                })

            if records:
                self._client.table("factions").upsert(
                    records,
                    on_conflict="faction_id"
                ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save factions batch: {e}")
            return False

    async def get_factions(self, world_id: str) -> List[Dict[str, Any]]:
        """Get all factions for a world."""
        if not self._client:
            return []
        try:
            result = self._client.table("factions").select("*").eq(
                "world_id", world_id
            ).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get factions: {e}")
            return []

    # =========================================================================
    # Game Session Operations
    # =========================================================================

    async def create_session(
        self,
        session_id: str,
        world_id: str,
        settings: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new game session."""
        if not self._client:
            return None
        try:
            result = self._client.table("game_sessions").insert({
                "session_id": session_id,
                "world_id": world_id,
                "status": "active",
                "current_tick": 0,
                "game_time": "Day 1, 08:00",
                "settings": settings or {}
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None

    async def update_session(
        self,
        session_id: str,
        current_tick: int = None,
        game_time: str = None,
        status: str = None,
        settings: Dict[str, Any] = None
    ) -> bool:
        """Update session state."""
        if not self._client:
            return False
        try:
            update_data = {"updated_at": "now()"}
            if current_tick is not None:
                update_data["current_tick"] = current_tick
            if game_time is not None:
                update_data["game_time"] = game_time
            if status is not None:
                update_data["status"] = status
            if settings is not None:
                update_data["settings"] = settings

            self._client.table("game_sessions").update(update_data).eq(
                "session_id", session_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        if not self._client:
            return None
        try:
            result = self._client.table("game_sessions").select("*").eq(
                "session_id", session_id
            ).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    async def list_sessions(
        self,
        world_id: str = None,
        status: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List game sessions."""
        if not self._client:
            return []
        try:
            query = self._client.table("game_sessions").select("*")
            if world_id:
                query = query.eq("world_id", world_id)
            if status:
                query = query.eq("status", status)
            result = query.order("created_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def end_session(self, session_id: str) -> bool:
        """End a game session."""
        if not self._client:
            return False
        try:
            self._client.table("game_sessions").update({
                "status": "ended",
                "ended_at": "now()"
            }).eq("session_id", session_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    # =========================================================================
    # Player Operations
    # =========================================================================

    async def create_player(
        self,
        player_id: str,
        session_id: str,
        name: str,
        x: int,
        y: int,
        stats: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a player for a session."""
        if not self._client:
            return None
        try:
            result = self._client.table("players").insert({
                "player_id": player_id,
                "session_id": session_id,
                "name": name,
                "x": x,
                "y": y,
                "stats": stats or {}
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create player: {e}")
            return None

    async def update_player(
        self,
        player_id: str,
        x: int = None,
        y: int = None,
        health: float = None,
        stamina: float = None,
        stats: Dict[str, Any] = None,
        inventory: List[Dict[str, Any]] = None
    ) -> bool:
        """Update player state."""
        if not self._client:
            return False
        try:
            update_data = {"updated_at": "now()"}
            if x is not None:
                update_data["x"] = x
            if y is not None:
                update_data["y"] = y
            if health is not None:
                update_data["health"] = health
            if stamina is not None:
                update_data["stamina"] = stamina
            if stats is not None:
                update_data["stats"] = stats
            if inventory is not None:
                update_data["inventory"] = inventory

            self._client.table("players").update(update_data).eq(
                "player_id", player_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update player: {e}")
            return False

    async def get_player(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get player for a session."""
        if not self._client:
            return None
        try:
            result = self._client.table("players").select("*").eq(
                "session_id", session_id
            ).single().execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get player: {e}")
            return None

    # =========================================================================
    # Agent Operations
    # =========================================================================

    async def save_agent(
        self,
        agent_id: str,
        session_id: str,
        world_id: str,
        name: str,
        agent_type: str,
        x: int,
        y: int,
        tier: str = "background",
        stats: Dict[str, Any] = None
    ) -> bool:
        """Save an agent."""
        if not self._client:
            return False
        try:
            data = {
                "agent_id": agent_id,
                "session_id": session_id,
                "world_id": world_id,
                "name": name,
                "agent_type": agent_type,
                "position_x": x,
                "position_y": y,
                "tier": tier
            }
            if stats:
                data.update({
                    "strength": stats.get("strength", 10),
                    "dexterity": stats.get("dexterity", 10),
                    "constitution": stats.get("constitution", 10),
                    "intelligence": stats.get("intelligence", 10),
                    "wisdom": stats.get("wisdom", 10),
                    "charisma": stats.get("charisma", 10)
                })

            self._client.table("agents").upsert(
                data, on_conflict="agent_id"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
            return False

    async def save_agents_batch(
        self,
        session_id: str,
        world_id: str,
        agents: List[Dict[str, Any]]
    ) -> bool:
        """Save multiple agents in a batch."""
        if not self._client:
            return False
        try:
            records = []
            for a in agents:
                stats = a.get("stats", {})
                records.append({
                    "agent_id": a.get("id") or a.get("agent_id"),
                    "session_id": session_id,
                    "world_id": world_id,
                    "name": a.get("name"),
                    "agent_type": a.get("agent_type", "npc"),
                    "position_x": a.get("x"),
                    "position_y": a.get("y"),
                    "tier": a.get("tier", "background"),
                    "current_action": a.get("current_action"),
                    "current_thought": a.get("current_thought"),
                    "strength": stats.get("strength", 10),
                    "dexterity": stats.get("dexterity", 10),
                    "constitution": stats.get("constitution", 10),
                    "intelligence": stats.get("intelligence", 10),
                    "wisdom": stats.get("wisdom", 10),
                    "charisma": stats.get("charisma", 10),
                    "health": a.get("health", 1.0),
                    "stamina": a.get("energy", 1.0)
                })

            if records:
                self._client.table("agents").upsert(
                    records, on_conflict="agent_id"
                ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to save agents batch: {e}")
            return False

    async def get_agents_for_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all agents for a session."""
        if not self._client:
            return []
        try:
            result = self._client.table("agents").select("*").eq(
                "session_id", session_id
            ).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            return []

    async def update_agent(
        self,
        agent_id: str,
        x: int = None,
        y: int = None,
        tier: str = None,
        current_action: str = None,
        current_thought: str = None,
        health: float = None,
        stamina: float = None
    ) -> bool:
        """Update agent state."""
        if not self._client:
            return False
        try:
            update_data = {"last_active": "now()"}
            if x is not None:
                update_data["position_x"] = x
            if y is not None:
                update_data["position_y"] = y
            if tier is not None:
                update_data["tier"] = tier
            if current_action is not None:
                update_data["current_action"] = current_action
            if current_thought is not None:
                update_data["current_thought"] = current_thought
            if health is not None:
                update_data["health"] = health
            if stamina is not None:
                update_data["stamina"] = stamina

            self._client.table("agents").update(update_data).eq(
                "agent_id", agent_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update agent: {e}")
            return False

    # =========================================================================
    # Party Operations
    # =========================================================================

    async def create_party(
        self,
        party_id: str,
        session_id: str,
        leader_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create a party."""
        if not self._client:
            return None
        try:
            result = self._client.table("parties").insert({
                "party_id": party_id,
                "session_id": session_id,
                "leader_id": leader_id
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create party: {e}")
            return None

    async def add_party_member(
        self,
        party_id: str,
        agent_id: str,
        role: str = "companion",
        loyalty: float = 0.5
    ) -> bool:
        """Add an agent to a party."""
        if not self._client:
            return False
        try:
            self._client.table("party_members").upsert({
                "party_id": party_id,
                "agent_id": agent_id,
                "role": role,
                "loyalty": loyalty
            }, on_conflict="party_id,agent_id").execute()
            return True
        except Exception as e:
            logger.error(f"Failed to add party member: {e}")
            return False

    async def get_party_members(
        self,
        party_id: str
    ) -> List[Dict[str, Any]]:
        """Get all members of a party."""
        if not self._client:
            return []
        try:
            result = self._client.table("party_members").select(
                "*, agents(*)"
            ).eq("party_id", party_id).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get party members: {e}")
            return []


@lru_cache()
def get_supabase_client() -> SupabaseClient:
    """Get singleton Supabase client."""
    return SupabaseClient()
