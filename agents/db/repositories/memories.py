"""
Memory Repository
CRUD operations for agent memories
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from agents.models.memory import Memory, Observation, Reflection, Plan, MemoryQuery
from agents.config import MemoryType
from agents.db.supabase_client import SupabaseClient


class MemoryRepository:
    """
    Repository for memory database operations.
    Handles the memory_stream table.
    """

    def __init__(self, client: SupabaseClient):
        """
        Initialize repository.

        Args:
            client: Supabase client instance
        """
        self.client = client
        self.table_name = "memory_stream"

    def create(self, memory: Memory) -> Memory:
        """
        Create a new memory in the database.

        Args:
            memory: Memory to create

        Returns:
            Created Memory with database-assigned fields
        """
        data = memory.to_database_dict()
        result = self.client.table(self.table_name).insert(data).execute()

        if result.data:
            return Memory.from_database_row(result.data[0])
        raise Exception("Failed to create memory")

    def record_observation(
        self,
        agent_id: UUID,
        description: str,
        importance: float = 0.5,
        game_time: Optional[datetime] = None,
        location_x: Optional[int] = None,
        location_y: Optional[int] = None,
    ) -> UUID:
        """
        Record an observation using database function.

        Args:
            agent_id: Agent UUID
            description: Observation description
            importance: Importance score (0-1)
            game_time: Game time of observation
            location_x: X coordinate
            location_y: Y coordinate

        Returns:
            UUID of created memory
        """
        result = self.client.rpc(
            'record_memory',
            {
                'p_agent_id': str(agent_id),
                'p_memory_type': MemoryType.OBSERVATION.value,
                'p_description': description,
                'p_importance': importance,
                'p_game_time': game_time.isoformat() if game_time else None,
                'p_location_x': location_x,
                'p_location_y': location_y,
            }
        ).execute()

        return UUID(result.data)

    def record_reflection(
        self,
        agent_id: UUID,
        description: str,
        source_memory_ids: List[UUID],
        importance: float = 0.8,
        game_time: Optional[datetime] = None,
    ) -> UUID:
        """
        Record a reflection.

        Args:
            agent_id: Agent UUID
            description: Reflection text
            source_memory_ids: UUIDs of source memories
            importance: Importance score
            game_time: Game time

        Returns:
            UUID of created memory
        """
        result = self.client.rpc(
            'record_memory',
            {
                'p_agent_id': str(agent_id),
                'p_memory_type': MemoryType.REFLECTION.value,
                'p_description': description,
                'p_importance': importance,
                'p_game_time': game_time.isoformat() if game_time else None,
                'p_source_memories': [str(m) for m in source_memory_ids],
            }
        ).execute()

        return UUID(result.data)

    def record_plan(
        self,
        agent_id: UUID,
        description: str,
        importance: float = 0.5,
        game_time: Optional[datetime] = None,
    ) -> UUID:
        """
        Record a plan.

        Args:
            agent_id: Agent UUID
            description: Plan description
            importance: Importance score
            game_time: Game time

        Returns:
            UUID of created memory
        """
        result = self.client.rpc(
            'record_memory',
            {
                'p_agent_id': str(agent_id),
                'p_memory_type': MemoryType.PLAN.value,
                'p_description': description,
                'p_importance': importance,
                'p_game_time': game_time.isoformat() if game_time else None,
            }
        ).execute()

        return UUID(result.data)

    def get(self, memory_id: UUID) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory UUID

        Returns:
            Memory or None if not found
        """
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("memory_id", str(memory_id))
            .execute()
        )

        if result.data:
            return Memory.from_database_row(result.data[0])
        return None

    def retrieve(
        self,
        agent_id: UUID,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using database scoring function.

        Args:
            agent_id: Agent UUID
            limit: Maximum memories to retrieve
            memory_types: Types to include

        Returns:
            List of memory dictionaries with retrieval scores
        """
        types = memory_types or [MemoryType.OBSERVATION, MemoryType.REFLECTION, MemoryType.PLAN]
        type_values = [t.value for t in types]

        result = self.client.rpc(
            'retrieve_memories',
            {
                'p_agent_id': str(agent_id),
                'p_limit': limit,
                'p_memory_types': type_values
            }
        ).execute()

        return result.data or []

    def get_recent(
        self,
        agent_id: UUID,
        hours: int = 24,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Memory]:
        """
        Get memories from the last N hours.

        Args:
            agent_id: Agent UUID
            hours: Hours to look back
            memory_types: Types to include

        Returns:
            List of Memory objects
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(agent_id))
            .gte("created_at", since.isoformat())
            .order("created_at", desc=True)
        )

        if memory_types:
            type_values = [t.value for t in memory_types]
            query = query.in_("memory_type", type_values)

        result = query.execute()

        return [Memory.from_database_row(row) for row in result.data or []]

    def get_importance_sum(
        self,
        agent_id: UUID,
        hours: int = 24
    ) -> float:
        """
        Get sum of importance scores (for reflection trigger).

        Args:
            agent_id: Agent UUID
            hours: Hours to look back

        Returns:
            Sum of importance * 10 (to match paper's 1-10 scale)
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        result = self.client.rpc(
            'get_importance_sum',
            {
                'p_agent_id': str(agent_id),
                'p_since': since.isoformat()
            }
        ).execute()

        return result.data or 0.0

    def touch(self, memory_id: UUID) -> None:
        """
        Update last_accessed time for a memory.

        Args:
            memory_id: Memory UUID to touch
        """
        self.client.rpc(
            'touch_memory',
            {'p_memory_id': str(memory_id)}
        ).execute()

    def search_by_text(
        self,
        agent_id: UUID,
        query_text: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search memories by text content.

        Args:
            agent_id: Agent UUID
            query_text: Text to search for
            limit: Maximum results

        Returns:
            List of matching memories
        """
        # Simple text search using ILIKE
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(agent_id))
            .ilike("description", f"%{query_text}%")
            .limit(limit)
            .execute()
        )

        return [Memory.from_database_row(row) for row in result.data or []]

    def get_by_location(
        self,
        agent_id: UUID,
        x: int,
        y: int,
        radius: int = 10
    ) -> List[Memory]:
        """
        Get memories from a location.

        Args:
            agent_id: Agent UUID
            x: Center X coordinate
            y: Center Y coordinate
            radius: Search radius

        Returns:
            List of memories from that location
        """
        # Get all memories for agent, then filter by location
        # (Supabase doesn't support complex WHERE clauses easily)
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(agent_id))
            .not_.is_("location_x", "null")
            .execute()
        )

        memories = []
        for row in result.data or []:
            loc_x = row.get("location_x", 0)
            loc_y = row.get("location_y", 0)
            if loc_x and loc_y:
                dist = ((loc_x - x) ** 2 + (loc_y - y) ** 2) ** 0.5
                if dist <= radius:
                    memories.append(Memory.from_database_row(row))

        return memories

    def delete_old(
        self,
        agent_id: UUID,
        days: int = 365,
        min_importance: float = 0.3
    ) -> int:
        """
        Delete old, low-importance memories.

        Args:
            agent_id: Agent UUID
            days: Age threshold in days
            min_importance: Importance threshold (delete below this)

        Returns:
            Number of memories deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        result = (
            self.client.table(self.table_name)
            .delete()
            .eq("agent_id", str(agent_id))
            .lt("created_at", cutoff.isoformat())
            .lt("importance", min_importance)
            .execute()
        )

        return len(result.data or [])

    def count(self, agent_id: UUID) -> int:
        """
        Count memories for an agent.

        Args:
            agent_id: Agent UUID

        Returns:
            Memory count
        """
        result = (
            self.client.table(self.table_name)
            .select("memory_id", count="exact")
            .eq("agent_id", str(agent_id))
            .execute()
        )
        return result.count or 0
