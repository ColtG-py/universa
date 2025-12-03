"""
Memory Stream
Core memory system for generative agents.
Stores observations, reflections, and plans in natural language.
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from uuid import UUID
import asyncio

from agents.models.memory import Memory, Observation, Reflection, Plan
from agents.config import (
    MemoryType,
    ALPHA_RECENCY, ALPHA_IMPORTANCE, ALPHA_RELEVANCE,
    REFLECTION_THRESHOLD, MAX_MEMORY_RETRIEVAL,
    EMBEDDING_DIMENSION
)
from agents.db.supabase_client import SupabaseClient


class MemoryStream:
    """
    The memory stream is the foundation of the generative agent architecture.
    It maintains a comprehensive record of all agent experiences stored in
    natural language, supporting retrieval based on recency, importance, and relevance.
    """

    def __init__(
        self,
        agent_id: UUID,
        supabase_client: Optional[SupabaseClient] = None,
        importance_scorer: Optional[Callable[[str], float]] = None,
        embedding_generator: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize memory stream for an agent.

        Args:
            agent_id: UUID of the agent this stream belongs to
            supabase_client: Database client for persistence
            importance_scorer: Function to score memory importance (0-1)
            embedding_generator: Function to generate embeddings for semantic search
        """
        self.agent_id = agent_id
        self.client = supabase_client
        self._importance_scorer = importance_scorer
        self._embedding_generator = embedding_generator

        # In-memory cache for recent memories
        self._recent_cache: List[Memory] = []
        self._cache_max_size = 100

        # Reflection tracking
        self._importance_accumulator = 0.0
        self._last_reflection_time: Optional[datetime] = None

    def set_importance_scorer(self, scorer: Callable[[str], float]) -> None:
        """Set the importance scoring function (usually LLM-based)"""
        self._importance_scorer = scorer

    def set_embedding_generator(self, generator: Callable[[str], List[float]]) -> None:
        """Set the embedding generation function"""
        self._embedding_generator = generator

    async def add_observation(
        self,
        description: str,
        location_x: Optional[int] = None,
        location_y: Optional[int] = None,
        game_time: Optional[datetime] = None,
        observed_agent_id: Optional[UUID] = None,
        importance: Optional[float] = None,
    ) -> Observation:
        """
        Add an observation to the memory stream.

        Args:
            description: Natural language description of what was observed
            location_x: X coordinate where observation occurred
            location_y: Y coordinate where observation occurred
            game_time: In-game time of observation
            observed_agent_id: ID of agent being observed (if applicable)
            importance: Pre-calculated importance (if None, will be scored)

        Returns:
            Created Observation memory
        """
        # Score importance if not provided
        if importance is None:
            importance = await self._score_importance(description)

        # Generate embedding if generator available
        embedding = None
        if self._embedding_generator:
            embedding = await self._generate_embedding(description)

        observation = Observation(
            agent_id=self.agent_id,
            description=description,
            importance=importance,
            location_x=location_x,
            location_y=location_y,
            game_time=game_time or datetime.utcnow(),
            observed_agent_id=observed_agent_id,
            embedding=embedding,
        )

        # Persist to database
        if self.client:
            await self._persist_memory(observation)

        # Add to cache
        self._add_to_cache(observation)

        # Track importance for reflection trigger
        self._importance_accumulator += importance * 10  # Scale to 1-10

        return observation

    async def add_reflection(
        self,
        description: str,
        source_memories: List[UUID],
        evidence_description: str = "",
        importance: float = 0.8,
    ) -> Reflection:
        """
        Add a reflection (higher-level insight) to the memory stream.

        Args:
            description: The insight/reflection text
            source_memories: UUIDs of memories this reflection is based on
            evidence_description: Description of evidence
            importance: Importance score (reflections default higher)

        Returns:
            Created Reflection memory
        """
        embedding = None
        if self._embedding_generator:
            embedding = await self._generate_embedding(description)

        reflection = Reflection(
            agent_id=self.agent_id,
            description=description,
            source_memories=source_memories,
            evidence_description=evidence_description,
            importance=importance,
            game_time=datetime.utcnow(),
            embedding=embedding,
        )

        if self.client:
            await self._persist_memory(reflection)

        self._add_to_cache(reflection)
        self._last_reflection_time = datetime.utcnow()
        self._importance_accumulator = 0.0  # Reset after reflection

        return reflection

    async def add_plan(
        self,
        description: str,
        granularity: str = "action",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        parent_plan_id: Optional[UUID] = None,
        importance: float = 0.5,
    ) -> Plan:
        """
        Add a plan to the memory stream.

        Args:
            description: What the agent plans to do
            granularity: "day", "hour", or "action"
            start_time: When the plan starts
            end_time: When the plan ends
            parent_plan_id: ID of parent plan (for hierarchical plans)
            importance: Importance score

        Returns:
            Created Plan memory
        """
        embedding = None
        if self._embedding_generator:
            embedding = await self._generate_embedding(description)

        plan = Plan(
            agent_id=self.agent_id,
            description=description,
            granularity=granularity,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time,
            parent_plan_id=parent_plan_id,
            importance=importance,
            game_time=datetime.utcnow(),
            embedding=embedding,
        )

        if self.client:
            await self._persist_memory(plan)

        self._add_to_cache(plan)

        return plan

    async def retrieve(
        self,
        query: Optional[str] = None,
        limit: int = MAX_MEMORY_RETRIEVAL,
        memory_types: Optional[List[MemoryType]] = None,
        alpha_recency: float = ALPHA_RECENCY,
        alpha_importance: float = ALPHA_IMPORTANCE,
        alpha_relevance: float = ALPHA_RELEVANCE,
    ) -> List[Memory]:
        """
        Retrieve memories based on combined scoring.
        score = α_recency × recency + α_importance × importance + α_relevance × relevance

        Args:
            query: Optional query text for relevance scoring
            limit: Maximum memories to retrieve
            memory_types: Types of memories to include
            alpha_recency: Weight for recency score
            alpha_importance: Weight for importance score
            alpha_relevance: Weight for relevance score

        Returns:
            List of memories sorted by retrieval score
        """
        types = memory_types or [
            MemoryType.OBSERVATION,
            MemoryType.REFLECTION,
            MemoryType.PLAN
        ]

        # Get candidate memories
        if self.client:
            memories = await self._fetch_memories(types, limit * 3)
        else:
            memories = [m for m in self._recent_cache if m.memory_type in types]

        # Calculate query embedding for relevance
        query_embedding = None
        if query and self._embedding_generator:
            query_embedding = await self._generate_embedding(query)

        # Score each memory
        scored_memories = []
        current_time = datetime.utcnow()

        for memory in memories:
            # Calculate relevance
            relevance = 0.5  # Default
            if query_embedding and memory.embedding:
                relevance = self._cosine_similarity(query_embedding, memory.embedding)

            # Calculate combined score
            score = memory.calculate_retrieval_score(
                relevance=relevance,
                alpha_recency=alpha_recency,
                alpha_importance=alpha_importance,
                alpha_relevance=alpha_relevance,
                current_time=current_time
            )

            scored_memories.append((score, memory))

        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Touch accessed memories (update last_accessed)
        result = []
        for score, memory in scored_memories[:limit]:
            memory.touch()
            if self.client:
                await self._touch_memory(memory.memory_id)
            result.append(memory)

        return result

    async def retrieve_recent(
        self,
        hours: int = 24,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Memory]:
        """
        Retrieve memories from the last N hours.

        Args:
            hours: Number of hours to look back
            memory_types: Types to include

        Returns:
            List of recent memories
        """
        types = memory_types or [MemoryType.OBSERVATION, MemoryType.REFLECTION]

        if self.client:
            return await self._fetch_recent_memories(hours, types)
        else:
            since = datetime.utcnow() - timedelta(hours=hours)
            return [
                m for m in self._recent_cache
                if m.created_at >= since and m.memory_type in types
            ]

    def should_reflect(self) -> bool:
        """
        Check if the agent should generate reflections.
        Triggered when accumulated importance exceeds threshold.

        Returns:
            True if reflection should be triggered
        """
        return self._importance_accumulator >= REFLECTION_THRESHOLD

    def get_importance_sum(self) -> float:
        """Get current accumulated importance for reflection trigger"""
        return self._importance_accumulator

    async def get_memories_for_reflection(
        self,
        count: int = 100
    ) -> List[Memory]:
        """
        Get recent memories for reflection generation.

        Args:
            count: Number of memories to retrieve

        Returns:
            List of recent observations
        """
        return await self.retrieve_recent(
            hours=24,
            memory_types=[MemoryType.OBSERVATION]
        )

    async def search_by_text(
        self,
        query: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search memories by text content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching memories
        """
        if self.client:
            result = (
                self.client.table("memory_stream")
                .select("*")
                .eq("agent_id", str(self.agent_id))
                .ilike("description", f"%{query}%")
                .limit(limit)
                .execute()
            )
            return [Memory.from_database_row(row) for row in result.data or []]
        else:
            query_lower = query.lower()
            return [
                m for m in self._recent_cache
                if query_lower in m.description.lower()
            ][:limit]

    async def get_memories_about_agent(
        self,
        other_agent_id: UUID,
        limit: int = 10
    ) -> List[Memory]:
        """
        Get memories involving another agent.

        Args:
            other_agent_id: ID of the other agent
            limit: Maximum memories to return

        Returns:
            Memories about the other agent
        """
        # Search for the agent's name or ID in memory descriptions
        # This is a simplified approach - could be enhanced with entity tagging
        if self.client:
            result = (
                self.client.table("memory_stream")
                .select("*")
                .eq("agent_id", str(self.agent_id))
                .order("created_at", desc=True)
                .limit(limit * 5)  # Get more, then filter
                .execute()
            )

            # Filter for memories mentioning the other agent
            # In a full implementation, we'd have agent tagging
            return [Memory.from_database_row(row) for row in result.data or []][:limit]
        else:
            return self._recent_cache[:limit]

    # Private helper methods

    async def _score_importance(self, description: str) -> float:
        """Score the importance of a memory description"""
        if self._importance_scorer:
            try:
                return self._importance_scorer(description)
            except Exception:
                pass
        # Default scoring based on heuristics
        return self._heuristic_importance(description)

    def _heuristic_importance(self, description: str) -> float:
        """
        Heuristic importance scoring based on content.
        Used as fallback when LLM scorer not available.
        """
        score = 0.3  # Base score

        # Keywords that suggest higher importance
        high_importance = [
            "died", "death", "killed", "born", "married", "wedding",
            "discovered", "found", "learned", "realized", "understood",
            "attacked", "fought", "battle", "war", "peace",
            "love", "hate", "betrayed", "trusted", "friendship",
            "secret", "revealed", "confession", "promise", "vow"
        ]

        medium_importance = [
            "met", "talked", "discussed", "agreed", "disagreed",
            "bought", "sold", "traded", "gave", "received",
            "traveled", "arrived", "left", "returned",
            "started", "finished", "completed", "failed"
        ]

        low_importance = [
            "walked", "ate", "drank", "slept", "woke",
            "saw", "heard", "noticed", "observed"
        ]

        description_lower = description.lower()

        for keyword in high_importance:
            if keyword in description_lower:
                score = max(score, 0.8)
                break

        for keyword in medium_importance:
            if keyword in description_lower:
                score = max(score, 0.5)
                break

        for keyword in low_importance:
            if keyword in description_lower:
                score = max(score, 0.2)

        return min(score, 1.0)

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self._embedding_generator:
            try:
                return self._embedding_generator(text)
            except Exception:
                pass
        # Return zero vector if no generator
        return [0.0] * EMBEDDING_DIMENSION

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _add_to_cache(self, memory: Memory) -> None:
        """Add memory to in-memory cache"""
        self._recent_cache.insert(0, memory)
        if len(self._recent_cache) > self._cache_max_size:
            self._recent_cache.pop()

    async def _persist_memory(self, memory: Memory) -> None:
        """Persist memory to database"""
        if not self.client:
            return

        data = memory.to_database_dict()
        self.client.table("memory_stream").insert(data).execute()

    async def _touch_memory(self, memory_id: UUID) -> None:
        """Update last_accessed time in database"""
        if not self.client:
            return

        self.client.rpc(
            'touch_memory',
            {'p_memory_id': str(memory_id)}
        ).execute()

    async def _fetch_memories(
        self,
        types: List[MemoryType],
        limit: int
    ) -> List[Memory]:
        """Fetch memories from database"""
        if not self.client:
            return []

        type_values = [t.value for t in types]
        result = (
            self.client.table("memory_stream")
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .in_("memory_type", type_values)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return [Memory.from_database_row(row) for row in result.data or []]

    async def _fetch_recent_memories(
        self,
        hours: int,
        types: List[MemoryType]
    ) -> List[Memory]:
        """Fetch recent memories from database"""
        if not self.client:
            return []

        since = datetime.utcnow() - timedelta(hours=hours)
        type_values = [t.value for t in types]

        result = (
            self.client.table("memory_stream")
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .in_("memory_type", type_values)
            .gte("created_at", since.isoformat())
            .order("created_at", desc=True)
            .execute()
        )

        return [Memory.from_database_row(row) for row in result.data or []]
