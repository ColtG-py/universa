"""
Episodic Memory
Stores specific autobiographical events with full context.
"What happened" - memories of specific experiences.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field

from agents.db.supabase_client import SupabaseClient


@dataclass
class Episode:
    """
    A single episodic memory - a specific event with context.
    """
    episode_id: UUID
    agent_id: UUID
    summary: str
    context: Dict[str, Any]  # Full environmental context
    actions: List[Dict[str, Any]]  # Actions taken during episode
    outcomes: Dict[str, Any]  # Results of actions
    skills_used: List[str]
    reflection: Optional[str] = None

    importance: float = 0.5
    success: bool = True

    game_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Location
    location_x: Optional[int] = None
    location_y: Optional[int] = None

    # Involved agents
    involved_agents: List[UUID] = field(default_factory=list)

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to database format"""
        return {
            "episode_id": str(self.episode_id),
            "agent_id": str(self.agent_id),
            "summary": self.summary,
            "context": self.context,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "skills_used": self.skills_used,
            "reflection": self.reflection,
            "importance": self.importance,
            "success": self.success,
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_database_row(cls, row: Dict[str, Any]) -> "Episode":
        """Create from database row"""
        return cls(
            episode_id=UUID(row["episode_id"]),
            agent_id=UUID(row["agent_id"]),
            summary=row["summary"],
            context=row.get("context", {}),
            actions=row.get("actions", []),
            outcomes=row.get("outcomes", {}),
            skills_used=row.get("skills_used", []),
            reflection=row.get("reflection"),
            importance=row.get("importance", 0.5),
            success=row.get("success", True),
            game_time=row.get("game_time"),
            created_at=row.get("created_at", datetime.utcnow()),
        )


class EpisodicMemory:
    """
    Episodic memory system for storing autobiographical experiences.

    Episodic memories are:
    - Temporally located (happened at a specific time)
    - Contextually rich (include environmental details)
    - First-person perspective
    - Include emotional/importance coloring
    """

    def __init__(
        self,
        agent_id: UUID,
        supabase_client: Optional[SupabaseClient] = None,
    ):
        """
        Initialize episodic memory.

        Args:
            agent_id: Agent this memory belongs to
            supabase_client: Database client
        """
        self.agent_id = agent_id
        self.client = supabase_client
        self.table_name = "episodic_memories"

        # In-memory cache for recent episodes
        self._recent_cache: List[Episode] = []
        self._cache_max_size = 50

    async def record_episode(
        self,
        summary: str,
        context: Dict[str, Any],
        actions: List[Dict[str, Any]],
        outcomes: Dict[str, Any],
        skills_used: List[str] = None,
        importance: float = 0.5,
        success: bool = True,
        game_time: Optional[datetime] = None,
        involved_agents: List[UUID] = None,
    ) -> Episode:
        """
        Record a new episodic memory.

        Args:
            summary: Brief description of what happened
            context: Environmental/situational context
            actions: List of actions taken
            outcomes: What resulted
            skills_used: Skills employed during episode
            importance: How significant this episode was
            success: Whether the episode was successful
            game_time: In-game time of episode
            involved_agents: Other agents involved

        Returns:
            Created Episode
        """
        from uuid import uuid4

        episode = Episode(
            episode_id=uuid4(),
            agent_id=self.agent_id,
            summary=summary,
            context=context,
            actions=actions,
            outcomes=outcomes,
            skills_used=skills_used or [],
            importance=importance,
            success=success,
            game_time=game_time or datetime.utcnow(),
            involved_agents=involved_agents or [],
        )

        # Persist to database
        if self.client:
            self.client.table(self.table_name).insert(
                episode.to_database_dict()
            ).execute()

        # Add to cache
        self._add_to_cache(episode)

        return episode

    async def get_episode(self, episode_id: UUID) -> Optional[Episode]:
        """Get a specific episode by ID"""
        if not self.client:
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("episode_id", str(episode_id))
            .execute()
        )

        if result.data:
            return Episode.from_database_row(result.data[0])
        return None

    async def get_recent_episodes(
        self,
        hours: int = 24,
        limit: int = 20
    ) -> List[Episode]:
        """
        Get recent episodes.

        Args:
            hours: Look back this many hours
            limit: Maximum episodes to return

        Returns:
            List of recent episodes
        """
        if not self.client:
            return self._recent_cache[:limit]

        since = datetime.utcnow() - timedelta(hours=hours)
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .gte("created_at", since.isoformat())
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [Episode.from_database_row(row) for row in result.data or []]

    async def get_episodes_with_skill(
        self,
        skill_id: str,
        limit: int = 10
    ) -> List[Episode]:
        """
        Get episodes where a specific skill was used.

        Args:
            skill_id: Skill identifier
            limit: Maximum episodes

        Returns:
            Episodes using that skill
        """
        if not self.client:
            return [e for e in self._recent_cache if skill_id in e.skills_used][:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .contains("skills_used", [skill_id])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [Episode.from_database_row(row) for row in result.data or []]

    async def get_episodes_with_agent(
        self,
        other_agent_id: UUID,
        limit: int = 10
    ) -> List[Episode]:
        """
        Get episodes involving another agent.

        Args:
            other_agent_id: ID of the other agent
            limit: Maximum episodes

        Returns:
            Episodes with that agent
        """
        # Search in summary text for now (full implementation would use involved_agents array)
        if not self.client:
            return self._recent_cache[:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .order("created_at", desc=True)
            .limit(limit * 3)  # Get more, then filter
            .execute()
        )

        return [Episode.from_database_row(row) for row in result.data or []][:limit]

    async def get_successful_episodes(
        self,
        skill_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Episode]:
        """
        Get successful episodes, optionally filtered by skill.

        Args:
            skill_id: Optional skill filter
            limit: Maximum episodes

        Returns:
            Successful episodes
        """
        if not self.client:
            episodes = [e for e in self._recent_cache if e.success]
            if skill_id:
                episodes = [e for e in episodes if skill_id in e.skills_used]
            return episodes[:limit]

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .eq("success", True)
            .order("created_at", desc=True)
            .limit(limit)
        )

        if skill_id:
            query = query.contains("skills_used", [skill_id])

        result = query.execute()
        return [Episode.from_database_row(row) for row in result.data or []]

    async def get_important_episodes(
        self,
        min_importance: float = 0.7,
        limit: int = 10
    ) -> List[Episode]:
        """
        Get highly important episodes.

        Args:
            min_importance: Minimum importance threshold
            limit: Maximum episodes

        Returns:
            Important episodes
        """
        if not self.client:
            return [e for e in self._recent_cache if e.importance >= min_importance][:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .gte("importance", min_importance)
            .order("importance", desc=True)
            .limit(limit)
            .execute()
        )

        return [Episode.from_database_row(row) for row in result.data or []]

    async def add_reflection(
        self,
        episode_id: UUID,
        reflection: str
    ) -> None:
        """
        Add a reflection to an existing episode.

        Args:
            episode_id: Episode to update
            reflection: Reflection text
        """
        if not self.client:
            return

        self.client.table(self.table_name).update({
            "reflection": reflection
        }).eq("episode_id", str(episode_id)).execute()

    async def count_episodes(self) -> int:
        """Count total episodes for this agent"""
        if not self.client:
            return len(self._recent_cache)

        result = (
            self.client.table(self.table_name)
            .select("episode_id", count="exact")
            .eq("agent_id", str(self.agent_id))
            .execute()
        )
        return result.count or 0

    async def get_skill_success_rate(self, skill_id: str) -> float:
        """
        Calculate success rate for a skill based on episodic memory.

        Args:
            skill_id: Skill to check

        Returns:
            Success rate (0-1)
        """
        episodes = await self.get_episodes_with_skill(skill_id, limit=50)
        if not episodes:
            return 0.5  # No data, return neutral

        successes = sum(1 for e in episodes if e.success)
        return successes / len(episodes)

    def _add_to_cache(self, episode: Episode) -> None:
        """Add episode to in-memory cache"""
        self._recent_cache.insert(0, episode)
        if len(self._recent_cache) > self._cache_max_size:
            self._recent_cache.pop()
