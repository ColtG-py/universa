"""
Semantic Memory
Stores facts and knowledge without temporal context.
"What I know" - general knowledge and beliefs.
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from agents.db.supabase_client import SupabaseClient
from agents.config import EMBEDDING_DIMENSION


@dataclass
class Fact:
    """
    A semantic fact - knowledge without temporal context.
    """
    fact_id: UUID
    agent_id: UUID
    fact_text: str
    category: Optional[str] = None
    confidence: float = 1.0  # How confident the agent is
    source: str = "experience"  # 'experience', 'told', 'observed', 'inferred'

    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    # Related facts
    related_facts: List[UUID] = field(default_factory=list)

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to database format"""
        return {
            "fact_id": str(self.fact_id),
            "agent_id": str(self.agent_id),
            "fact_text": self.fact_text,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "embedding": self.embedding,
        }

    @classmethod
    def from_database_row(cls, row: Dict[str, Any]) -> "Fact":
        """Create from database row"""
        return cls(
            fact_id=UUID(row["fact_id"]),
            agent_id=UUID(row["agent_id"]),
            fact_text=row["fact_text"],
            category=row.get("category"),
            confidence=row.get("confidence", 1.0),
            source=row.get("source", "experience"),
            access_count=row.get("access_count", 0),
            created_at=row.get("created_at", datetime.utcnow()),
            embedding=row.get("embedding"),
        )


class SemanticMemory:
    """
    Semantic memory system for storing factual knowledge.

    Semantic memories are:
    - Context-free (not tied to specific episodes)
    - Factual or belief-based
    - Can have varying confidence levels
    - Organized by category
    """

    # Fact categories
    CATEGORIES = [
        "self",           # Facts about self
        "people",         # Facts about other agents
        "places",         # Facts about locations
        "skills",         # Facts about abilities
        "world",          # General world knowledge
        "relationships",  # Relationship facts
        "factions",       # Political knowledge
        "items",          # Knowledge about objects
        "events",         # Historical events
        "customs",        # Social norms and customs
    ]

    def __init__(
        self,
        agent_id: UUID,
        supabase_client: Optional[SupabaseClient] = None,
        embedding_generator: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize semantic memory.

        Args:
            agent_id: Agent this memory belongs to
            supabase_client: Database client
            embedding_generator: Function to generate embeddings
        """
        self.agent_id = agent_id
        self.client = supabase_client
        self._embedding_generator = embedding_generator
        self.table_name = "semantic_facts"

        # In-memory cache
        self._cache: Dict[str, List[Fact]] = {}  # category -> facts
        self._all_facts: List[Fact] = []

    def set_embedding_generator(
        self,
        generator: Callable[[str], List[float]]
    ) -> None:
        """Set embedding generator function"""
        self._embedding_generator = generator

    async def store_fact(
        self,
        fact_text: str,
        category: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "experience",
    ) -> Fact:
        """
        Store a new fact.

        Args:
            fact_text: The fact to store
            category: Category for organization
            confidence: How confident (0-1)
            source: How the fact was learned

        Returns:
            Created Fact
        """
        # Check for duplicate/similar facts
        existing = await self.find_similar_fact(fact_text)
        if existing and existing.confidence >= confidence:
            # Update access count of existing fact
            await self._increment_access(existing.fact_id)
            return existing

        # Generate embedding
        embedding = None
        if self._embedding_generator:
            try:
                embedding = self._embedding_generator(fact_text)
            except Exception:
                pass

        fact = Fact(
            fact_id=uuid4(),
            agent_id=self.agent_id,
            fact_text=fact_text,
            category=category or self._infer_category(fact_text),
            confidence=confidence,
            source=source,
            embedding=embedding,
        )

        # Persist
        if self.client:
            self.client.table(self.table_name).insert(
                fact.to_database_dict()
            ).execute()

        # Cache
        self._add_to_cache(fact)

        return fact

    async def get_fact(self, fact_id: UUID) -> Optional[Fact]:
        """Get a specific fact by ID"""
        if not self.client:
            for fact in self._all_facts:
                if fact.fact_id == fact_id:
                    return fact
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("fact_id", str(fact_id))
            .execute()
        )

        if result.data:
            fact = Fact.from_database_row(result.data[0])
            await self._increment_access(fact_id)
            return fact
        return None

    async def get_facts_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[Fact]:
        """
        Get facts in a category.

        Args:
            category: Category to filter by
            limit: Maximum facts

        Returns:
            Facts in that category
        """
        if not self.client:
            return self._cache.get(category, [])[:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .eq("category", category)
            .order("confidence", desc=True)
            .limit(limit)
            .execute()
        )

        return [Fact.from_database_row(row) for row in result.data or []]

    async def search_facts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Fact]:
        """
        Search facts by text.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching facts
        """
        if not self.client:
            query_lower = query.lower()
            return [
                f for f in self._all_facts
                if query_lower in f.fact_text.lower()
            ][:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .ilike("fact_text", f"%{query}%")
            .order("confidence", desc=True)
            .limit(limit)
            .execute()
        )

        return [Fact.from_database_row(row) for row in result.data or []]

    async def find_similar_fact(
        self,
        fact_text: str,
        threshold: float = 0.9
    ) -> Optional[Fact]:
        """
        Find a similar existing fact.

        Args:
            fact_text: Fact to compare
            threshold: Similarity threshold

        Returns:
            Similar fact if found
        """
        # Simple text matching for now
        # Full implementation would use embedding similarity
        fact_lower = fact_text.lower()

        if not self.client:
            for fact in self._all_facts:
                if self._text_similarity(fact_lower, fact.fact_text.lower()) > threshold:
                    return fact
            return None

        # Search for similar facts
        words = fact_text.split()[:5]  # Use first 5 words
        for word in words:
            if len(word) > 3:
                result = (
                    self.client.table(self.table_name)
                    .select("*")
                    .eq("agent_id", str(self.agent_id))
                    .ilike("fact_text", f"%{word}%")
                    .limit(10)
                    .execute()
                )

                for row in result.data or []:
                    fact = Fact.from_database_row(row)
                    if self._text_similarity(fact_lower, fact.fact_text.lower()) > threshold:
                        return fact

        return None

    async def update_confidence(
        self,
        fact_id: UUID,
        new_confidence: float
    ) -> None:
        """
        Update confidence in a fact.

        Args:
            fact_id: Fact to update
            new_confidence: New confidence level
        """
        if not self.client:
            for fact in self._all_facts:
                if fact.fact_id == fact_id:
                    fact.confidence = new_confidence
            return

        self.client.table(self.table_name).update({
            "confidence": new_confidence
        }).eq("fact_id", str(fact_id)).execute()

    async def get_facts_about_agent(
        self,
        other_agent_name: str,
        limit: int = 10
    ) -> List[Fact]:
        """
        Get facts about another agent.

        Args:
            other_agent_name: Name of the agent
            limit: Maximum facts

        Returns:
            Facts about that agent
        """
        return await self.search_facts(other_agent_name, limit)

    async def get_facts_about_location(
        self,
        location_name: str,
        limit: int = 10
    ) -> List[Fact]:
        """
        Get facts about a location.

        Args:
            location_name: Name of the place
            limit: Maximum facts

        Returns:
            Facts about that location
        """
        facts = await self.get_facts_by_category("places", limit * 2)
        return [f for f in facts if location_name.lower() in f.fact_text.lower()][:limit]

    async def get_most_accessed(self, limit: int = 10) -> List[Fact]:
        """
        Get most frequently accessed facts.

        Args:
            limit: Maximum facts

        Returns:
            Most accessed facts
        """
        if not self.client:
            sorted_facts = sorted(
                self._all_facts,
                key=lambda f: f.access_count,
                reverse=True
            )
            return sorted_facts[:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .order("access_count", desc=True)
            .limit(limit)
            .execute()
        )

        return [Fact.from_database_row(row) for row in result.data or []]

    async def delete_low_confidence(
        self,
        threshold: float = 0.3
    ) -> int:
        """
        Delete facts with very low confidence.

        Args:
            threshold: Confidence threshold

        Returns:
            Number of facts deleted
        """
        if not self.client:
            before = len(self._all_facts)
            self._all_facts = [f for f in self._all_facts if f.confidence >= threshold]
            return before - len(self._all_facts)

        result = (
            self.client.table(self.table_name)
            .delete()
            .eq("agent_id", str(self.agent_id))
            .lt("confidence", threshold)
            .execute()
        )

        return len(result.data or [])

    def _infer_category(self, fact_text: str) -> str:
        """Infer category from fact text"""
        text_lower = fact_text.lower()

        if any(w in text_lower for w in ["i am", "i have", "my ", "myself"]):
            return "self"
        if any(w in text_lower for w in ["is a person", "they are", "he is", "she is"]):
            return "people"
        if any(w in text_lower for w in ["located", "place", "town", "city", "village"]):
            return "places"
        if any(w in text_lower for w in ["can do", "able to", "skill", "know how"]):
            return "skills"
        if any(w in text_lower for w in ["friend", "enemy", "knows", "married"]):
            return "relationships"
        if any(w in text_lower for w in ["faction", "kingdom", "guild", "alliance"]):
            return "factions"

        return "world"

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _add_to_cache(self, fact: Fact) -> None:
        """Add fact to cache"""
        self._all_facts.append(fact)
        if fact.category:
            if fact.category not in self._cache:
                self._cache[fact.category] = []
            self._cache[fact.category].append(fact)

    async def _increment_access(self, fact_id: UUID) -> None:
        """Increment access count for a fact"""
        if not self.client:
            for fact in self._all_facts:
                if fact.fact_id == fact_id:
                    fact.access_count += 1
            return

        # Use RPC or direct update
        self.client.table(self.table_name).update({
            "access_count": self.client.table(self.table_name)
            .select("access_count")
            .eq("fact_id", str(fact_id))
            .execute()
            .data[0].get("access_count", 0) + 1
        }).eq("fact_id", str(fact_id)).execute()
