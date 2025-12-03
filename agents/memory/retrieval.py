"""
Memory Retrieval System
Implements the retrieval function from the Stanford Generative Agents paper.
score = α_recency × recency + α_importance × importance + α_relevance × relevance
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass
import math

from agents.models.memory import Memory
from agents.config import (
    MemoryType,
    ALPHA_RECENCY, ALPHA_IMPORTANCE, ALPHA_RELEVANCE,
    RECENCY_DECAY_FACTOR, MAX_MEMORY_RETRIEVAL
)
from agents.db.supabase_client import SupabaseClient


@dataclass
class ScoredMemory:
    """A memory with its retrieval scores"""
    memory: Memory
    recency_score: float
    importance_score: float
    relevance_score: float
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": str(self.memory.memory_id),
            "description": self.memory.description,
            "memory_type": self.memory.memory_type.value,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "relevance_score": self.relevance_score,
            "combined_score": self.combined_score,
        }


class MemoryRetrieval:
    """
    Memory retrieval system implementing the Stanford paper's approach.

    The retrieval function scores memories based on:
    1. Recency: How recently was this memory accessed?
    2. Importance: How significant is this memory?
    3. Relevance: How related is this memory to the current context?
    """

    def __init__(
        self,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_generator: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize retrieval system.

        Args:
            supabase_client: Database client
            embedding_generator: Function to generate embeddings
        """
        self.client = supabase_client
        self._embedding_generator = embedding_generator

        # Cache for embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 1000

    def set_embedding_generator(
        self,
        generator: Callable[[str], List[float]]
    ) -> None:
        """Set the embedding generation function"""
        self._embedding_generator = generator

    def calculate_recency_score(
        self,
        last_accessed: datetime,
        current_time: Optional[datetime] = None,
        decay_factor: float = RECENCY_DECAY_FACTOR
    ) -> float:
        """
        Calculate recency score using exponential decay.
        recency = decay_factor ^ hours_since_access

        Args:
            last_accessed: When the memory was last accessed
            current_time: Reference time (defaults to now)
            decay_factor: Decay rate (default 0.995)

        Returns:
            Recency score between 0 and 1
        """
        if current_time is None:
            current_time = datetime.utcnow()

        hours_since_access = (current_time - last_accessed).total_seconds() / 3600.0
        hours_since_access = max(0, hours_since_access)  # Ensure non-negative

        return decay_factor ** hours_since_access

    def calculate_importance_score(
        self,
        importance: float,
        normalize: bool = True
    ) -> float:
        """
        Process importance score for retrieval.

        Args:
            importance: Raw importance value (0-1)
            normalize: Whether to normalize (already 0-1)

        Returns:
            Normalized importance score
        """
        # Importance is already 0-1, but we can apply transformations
        # For example, slight boost to higher importance memories
        if normalize:
            # Apply slight sigmoid-like curve to emphasize extremes
            return importance
        return importance

    def calculate_relevance_score(
        self,
        query_embedding: List[float],
        memory_embedding: List[float]
    ) -> float:
        """
        Calculate relevance using cosine similarity.

        Args:
            query_embedding: Embedding of the query
            memory_embedding: Embedding of the memory

        Returns:
            Relevance score between 0 and 1
        """
        if not query_embedding or not memory_embedding:
            return 0.5  # Default to neutral

        if len(query_embedding) != len(memory_embedding):
            return 0.5

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, memory_embedding))
        norm_q = math.sqrt(sum(a * a for a in query_embedding))
        norm_m = math.sqrt(sum(b * b for b in memory_embedding))

        if norm_q == 0 or norm_m == 0:
            return 0.5

        similarity = dot_product / (norm_q * norm_m)

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def calculate_combined_score(
        self,
        recency: float,
        importance: float,
        relevance: float,
        alpha_recency: float = ALPHA_RECENCY,
        alpha_importance: float = ALPHA_IMPORTANCE,
        alpha_relevance: float = ALPHA_RELEVANCE,
    ) -> float:
        """
        Calculate combined retrieval score.
        score = α_recency × recency + α_importance × importance + α_relevance × relevance

        Args:
            recency: Recency score
            importance: Importance score
            relevance: Relevance score
            alpha_recency: Weight for recency
            alpha_importance: Weight for importance
            alpha_relevance: Weight for relevance

        Returns:
            Combined score
        """
        return (
            alpha_recency * recency +
            alpha_importance * importance +
            alpha_relevance * relevance
        )

    async def retrieve(
        self,
        agent_id: UUID,
        query: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = MAX_MEMORY_RETRIEVAL,
        alpha_recency: float = ALPHA_RECENCY,
        alpha_importance: float = ALPHA_IMPORTANCE,
        alpha_relevance: float = ALPHA_RELEVANCE,
        min_score: float = 0.0,
    ) -> List[ScoredMemory]:
        """
        Retrieve memories with full scoring.

        Args:
            agent_id: Agent to retrieve memories for
            query: Optional query for relevance scoring
            memory_types: Types to include
            limit: Maximum memories to return
            alpha_recency: Weight for recency
            alpha_importance: Weight for importance
            alpha_relevance: Weight for relevance
            min_score: Minimum combined score threshold

        Returns:
            List of ScoredMemory objects sorted by combined score
        """
        if not self.client:
            return []

        types = memory_types or [
            MemoryType.OBSERVATION,
            MemoryType.REFLECTION,
            MemoryType.PLAN
        ]

        # Fetch candidate memories (more than limit for scoring)
        candidates = await self._fetch_candidates(agent_id, types, limit * 3)

        if not candidates:
            return []

        # Generate query embedding if needed
        query_embedding = None
        if query and self._embedding_generator:
            query_embedding = await self._get_embedding(query)

        # Score all candidates
        current_time = datetime.utcnow()
        scored = []

        for memory in candidates:
            recency = self.calculate_recency_score(
                memory.last_accessed,
                current_time
            )

            importance = self.calculate_importance_score(memory.importance)

            # Calculate relevance
            if query_embedding and memory.embedding:
                relevance = self.calculate_relevance_score(
                    query_embedding,
                    memory.embedding
                )
            else:
                relevance = 0.5  # Neutral relevance if no embeddings

            combined = self.calculate_combined_score(
                recency, importance, relevance,
                alpha_recency, alpha_importance, alpha_relevance
            )

            if combined >= min_score:
                scored.append(ScoredMemory(
                    memory=memory,
                    recency_score=recency,
                    importance_score=importance,
                    relevance_score=relevance,
                    combined_score=combined
                ))

        # Sort by combined score
        scored.sort(key=lambda x: x.combined_score, reverse=True)

        # Return top results and touch them
        results = scored[:limit]
        for sm in results:
            await self._touch_memory(sm.memory.memory_id)

        return results

    async def retrieve_by_context(
        self,
        agent_id: UUID,
        context_memories: List[Memory],
        limit: int = MAX_MEMORY_RETRIEVAL,
    ) -> List[ScoredMemory]:
        """
        Retrieve memories relevant to a set of context memories.
        Useful for reflection generation.

        Args:
            agent_id: Agent ID
            context_memories: Memories to use as context
            limit: Maximum to return

        Returns:
            Related memories
        """
        if not context_memories:
            return []

        # Combine context memory descriptions for query
        context_text = " ".join(m.description for m in context_memories[:5])

        return await self.retrieve(
            agent_id=agent_id,
            query=context_text,
            limit=limit,
            alpha_relevance=1.5,  # Boost relevance for context-based retrieval
        )

    async def retrieve_about_topic(
        self,
        agent_id: UUID,
        topic: str,
        limit: int = MAX_MEMORY_RETRIEVAL,
    ) -> List[ScoredMemory]:
        """
        Retrieve memories about a specific topic.

        Args:
            agent_id: Agent ID
            topic: Topic to search for
            limit: Maximum to return

        Returns:
            Topic-related memories
        """
        return await self.retrieve(
            agent_id=agent_id,
            query=topic,
            limit=limit,
            alpha_relevance=2.0,  # Heavily weight relevance
            alpha_recency=0.5,    # Reduce recency weight
        )

    async def retrieve_about_agent(
        self,
        agent_id: UUID,
        other_agent_name: str,
        limit: int = MAX_MEMORY_RETRIEVAL,
    ) -> List[ScoredMemory]:
        """
        Retrieve memories about another agent.

        Args:
            agent_id: Agent ID
            other_agent_name: Name of the other agent
            limit: Maximum to return

        Returns:
            Memories about the other agent
        """
        # Combine semantic search with text search
        semantic_results = await self.retrieve(
            agent_id=agent_id,
            query=f"interactions with {other_agent_name}",
            limit=limit,
            alpha_relevance=2.0,
        )

        # Also do text search for the name
        text_results = await self._text_search(
            agent_id,
            other_agent_name,
            limit
        )

        # Merge results, preferring semantic matches
        seen_ids = {sm.memory.memory_id for sm in semantic_results}
        for memory in text_results:
            if memory.memory_id not in seen_ids:
                semantic_results.append(ScoredMemory(
                    memory=memory,
                    recency_score=0.5,
                    importance_score=memory.importance,
                    relevance_score=0.8,  # High relevance for name match
                    combined_score=0.6
                ))

        return sorted(
            semantic_results,
            key=lambda x: x.combined_score,
            reverse=True
        )[:limit]

    async def _fetch_candidates(
        self,
        agent_id: UUID,
        types: List[MemoryType],
        limit: int
    ) -> List[Memory]:
        """Fetch candidate memories from database"""
        if not self.client:
            return []

        type_values = [t.value for t in types]
        result = (
            self.client.table("memory_stream")
            .select("*")
            .eq("agent_id", str(agent_id))
            .in_("memory_type", type_values)
            .order("last_accessed", desc=True)
            .limit(limit)
            .execute()
        )

        return [Memory.from_database_row(row) for row in result.data or []]

    async def _text_search(
        self,
        agent_id: UUID,
        query: str,
        limit: int
    ) -> List[Memory]:
        """Text-based search in memory descriptions"""
        if not self.client:
            return []

        result = (
            self.client.table("memory_stream")
            .select("*")
            .eq("agent_id", str(agent_id))
            .ilike("description", f"%{query}%")
            .limit(limit)
            .execute()
        )

        return [Memory.from_database_row(row) for row in result.data or []]

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if not self._embedding_generator:
            return None

        try:
            embedding = self._embedding_generator(text)

            # Cache management
            if len(self._embedding_cache) >= self._cache_max_size:
                # Remove oldest entries
                keys = list(self._embedding_cache.keys())
                for key in keys[:len(keys)//2]:
                    del self._embedding_cache[key]

            self._embedding_cache[text] = embedding
            return embedding
        except Exception:
            return None

    async def _touch_memory(self, memory_id: UUID) -> None:
        """Update last_accessed time"""
        if not self.client:
            return

        self.client.rpc(
            'touch_memory',
            {'p_memory_id': str(memory_id)}
        ).execute()


class RetrievalCache:
    """
    Cache for frequently accessed retrieval results.
    Reduces database load for common queries.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            max_size: Maximum cached entries
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, Tuple[datetime, List[ScoredMemory]]] = {}

    def get(
        self,
        agent_id: UUID,
        query: Optional[str]
    ) -> Optional[List[ScoredMemory]]:
        """Get cached results if fresh"""
        key = self._make_key(agent_id, query)
        if key not in self._cache:
            return None

        timestamp, results = self._cache[key]
        if datetime.utcnow() - timestamp > self.ttl:
            del self._cache[key]
            return None

        return results

    def set(
        self,
        agent_id: UUID,
        query: Optional[str],
        results: List[ScoredMemory]
    ) -> None:
        """Cache results"""
        if len(self._cache) >= self.max_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][0]
            )
            for key in sorted_keys[:len(sorted_keys)//2]:
                del self._cache[key]

        key = self._make_key(agent_id, query)
        self._cache[key] = (datetime.utcnow(), results)

    def invalidate(self, agent_id: UUID) -> None:
        """Invalidate all cache entries for an agent"""
        prefix = str(agent_id)
        keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]

    def _make_key(self, agent_id: UUID, query: Optional[str]) -> str:
        """Create cache key"""
        return f"{agent_id}:{query or 'none'}"
