"""
Agent Repository
CRUD operations for agent entities
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from agents.models.agent_state import AgentState, CoreStats, CoreNeeds, Alignment
from agents.db.supabase_client import SupabaseClient


class AgentRepository:
    """
    Repository for agent database operations.
    Provides CRUD and query methods for agents.
    """

    def __init__(self, client: SupabaseClient):
        """
        Initialize repository.

        Args:
            client: Supabase client instance
        """
        self.client = client
        self.table_name = "agents"

    def create(self, agent: AgentState) -> AgentState:
        """
        Create a new agent in the database.

        Args:
            agent: AgentState to create

        Returns:
            Created AgentState with database-assigned fields
        """
        data = agent.to_database_dict()
        result = self.client.table(self.table_name).insert(data).execute()

        if result.data:
            return AgentState.from_database_row(result.data[0])
        raise Exception("Failed to create agent")

    def get(self, agent_id: UUID) -> Optional[AgentState]:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent UUID

        Returns:
            AgentState or None if not found
        """
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(agent_id))
            .execute()
        )

        if result.data:
            return AgentState.from_database_row(result.data[0])
        return None

    def get_by_world(
        self,
        world_id: UUID,
        alive_only: bool = True,
        limit: int = 100
    ) -> List[AgentState]:
        """
        Get all agents in a world.

        Args:
            world_id: World UUID
            alive_only: Only return living agents
            limit: Maximum number to return

        Returns:
            List of AgentState
        """
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("world_id", str(world_id))
            .limit(limit)
        )

        if alive_only:
            query = query.eq("is_alive", True)

        result = query.execute()

        return [AgentState.from_database_row(row) for row in result.data or []]

    def get_nearby(
        self,
        world_id: UUID,
        x: int,
        y: int,
        radius: int,
        exclude_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """
        Get agents near a position using database function.

        Args:
            world_id: World UUID
            x: Center X coordinate
            y: Center Y coordinate
            radius: Search radius
            exclude_id: Agent ID to exclude

        Returns:
            List of agent summaries
        """
        result = self.client.rpc(
            'get_nearby_agents',
            {
                'p_world_id': str(world_id),
                'p_x': x,
                'p_y': y,
                'p_radius': radius
            }
        ).execute()

        agents = result.data or []

        if exclude_id:
            agents = [a for a in agents if a['agent_id'] != str(exclude_id)]

        return agents

    def update(self, agent: AgentState) -> AgentState:
        """
        Update an existing agent.

        Args:
            agent: AgentState with updated values

        Returns:
            Updated AgentState
        """
        data = agent.to_database_dict()
        # Remove agent_id from update data (it's in the WHERE clause)
        agent_id = data.pop("agent_id")

        result = (
            self.client.table(self.table_name)
            .update(data)
            .eq("agent_id", agent_id)
            .execute()
        )

        if result.data:
            return AgentState.from_database_row(result.data[0])
        raise Exception(f"Failed to update agent {agent_id}")

    def update_position(
        self,
        agent_id: UUID,
        x: int,
        y: int
    ) -> None:
        """
        Update only an agent's position.

        Args:
            agent_id: Agent UUID
            x: New X position
            y: New Y position
        """
        chunk_id = f"{x // 256}_{y // 256}"
        self.client.table(self.table_name).update({
            "position_x": x,
            "position_y": y,
            "chunk_id": chunk_id,
            "last_active": datetime.utcnow().isoformat()
        }).eq("agent_id", str(agent_id)).execute()

    def update_needs(
        self,
        agent_id: UUID,
        hours_elapsed: float
    ) -> None:
        """
        Update agent needs using database function.

        Args:
            agent_id: Agent UUID
            hours_elapsed: Game hours elapsed
        """
        self.client.rpc(
            'update_agent_needs',
            {
                'p_agent_id': str(agent_id),
                'p_hours_elapsed': hours_elapsed
            }
        ).execute()

    def update_stats(
        self,
        agent_id: UUID,
        stats: CoreStats
    ) -> None:
        """
        Update agent stats.

        Args:
            agent_id: Agent UUID
            stats: New CoreStats
        """
        self.client.table(self.table_name).update({
            "strength": stats.strength,
            "dexterity": stats.dexterity,
            "constitution": stats.constitution,
            "intelligence": stats.intelligence,
            "wisdom": stats.wisdom,
            "charisma": stats.charisma
        }).eq("agent_id", str(agent_id)).execute()

    def update_health(
        self,
        agent_id: UUID,
        health: float,
        stamina: float,
        is_alive: bool
    ) -> None:
        """
        Update agent health status.

        Args:
            agent_id: Agent UUID
            health: Health value (0-1)
            stamina: Stamina value (0-1)
            is_alive: Whether agent is alive
        """
        self.client.table(self.table_name).update({
            "health": health,
            "stamina": stamina,
            "is_alive": is_alive
        }).eq("agent_id", str(agent_id)).execute()

    def delete(self, agent_id: UUID) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: Agent UUID to delete

        Returns:
            True if deleted, False if not found
        """
        result = (
            self.client.table(self.table_name)
            .delete()
            .eq("agent_id", str(agent_id))
            .execute()
        )
        return len(result.data or []) > 0

    def count_by_world(self, world_id: UUID, alive_only: bool = True) -> int:
        """
        Count agents in a world.

        Args:
            world_id: World UUID
            alive_only: Only count living agents

        Returns:
            Agent count
        """
        query = (
            self.client.table(self.table_name)
            .select("agent_id", count="exact")
            .eq("world_id", str(world_id))
        )

        if alive_only:
            query = query.eq("is_alive", True)

        result = query.execute()
        return result.count or 0

    def get_by_faction(
        self,
        faction_id: UUID,
        alive_only: bool = True
    ) -> List[AgentState]:
        """
        Get all agents in a faction.

        Args:
            faction_id: Faction UUID
            alive_only: Only return living agents

        Returns:
            List of AgentState
        """
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("faction_id", str(faction_id))
        )

        if alive_only:
            query = query.eq("is_alive", True)

        result = query.execute()
        return [AgentState.from_database_row(row) for row in result.data or []]

    def get_by_chunk(
        self,
        chunk_id: str,
        world_id: UUID
    ) -> List[AgentState]:
        """
        Get all agents in a specific chunk.

        Args:
            chunk_id: Chunk identifier (e.g., "4_5")
            world_id: World UUID

        Returns:
            List of AgentState
        """
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("world_id", str(world_id))
            .eq("chunk_id", chunk_id)
            .eq("is_alive", True)
            .execute()
        )
        return [AgentState.from_database_row(row) for row in result.data or []]
