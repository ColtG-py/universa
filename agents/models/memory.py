"""
Memory Models
Data structures for agent memory stream
Based on Stanford Generative Agents paper
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from agents.config import MemoryType, RECENCY_DECAY_FACTOR


class Memory(BaseModel):
    """
    Base memory class for the memory stream.
    Represents a single memory entry that can be an observation,
    reflection, or plan.
    """
    memory_id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    memory_type: MemoryType
    description: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="Importance score (0-1)")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    game_time: Optional[datetime] = None

    # Location context
    location_x: Optional[int] = None
    location_y: Optional[int] = None

    # For reflections: source memories
    source_memories: List[UUID] = Field(default_factory=list)

    # Embedding for semantic search (stored separately)
    embedding: Optional[List[float]] = None

    def calculate_recency_score(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate recency score based on time since last access.
        Uses exponential decay: 0.995^hours

        Args:
            current_time: Reference time (defaults to now)

        Returns:
            Recency score between 0 and 1
        """
        if current_time is None:
            current_time = datetime.utcnow()

        hours_since_access = (current_time - self.last_accessed).total_seconds() / 3600.0
        return RECENCY_DECAY_FACTOR ** hours_since_access

    def calculate_retrieval_score(
        self,
        relevance: float = 0.5,
        alpha_recency: float = 1.0,
        alpha_importance: float = 1.0,
        alpha_relevance: float = 1.0,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate combined retrieval score.
        score = α_recency × recency + α_importance × importance + α_relevance × relevance

        Args:
            relevance: Relevance score from embedding similarity
            alpha_recency: Weight for recency
            alpha_importance: Weight for importance
            alpha_relevance: Weight for relevance
            current_time: Reference time for recency calculation

        Returns:
            Combined retrieval score
        """
        recency = self.calculate_recency_score(current_time)

        return (
            alpha_recency * recency +
            alpha_importance * self.importance +
            alpha_relevance * relevance
        )

    def touch(self) -> None:
        """Update last accessed time"""
        self.last_accessed = datetime.utcnow()

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "memory_type": self.memory_type.value,
            "description": self.description,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "game_time": self.game_time.isoformat() if self.game_time else None,
            "location_x": self.location_x,
            "location_y": self.location_y,
            "source_memories": [str(m) for m in self.source_memories],
            "embedding": self.embedding,
        }

    @classmethod
    def from_database_row(cls, row: Dict[str, Any]) -> "Memory":
        """Create Memory from database row"""
        return cls(
            memory_id=UUID(row["memory_id"]),
            agent_id=UUID(row["agent_id"]),
            memory_type=MemoryType(row["memory_type"]),
            description=row["description"],
            importance=row.get("importance", 0.5),
            created_at=row.get("created_at", datetime.utcnow()),
            last_accessed=row.get("last_accessed", datetime.utcnow()),
            game_time=row.get("game_time"),
            location_x=row.get("location_x"),
            location_y=row.get("location_y"),
            source_memories=[UUID(m) for m in (row.get("source_memories") or [])],
            embedding=row.get("embedding"),
        )


class Observation(Memory):
    """
    An observation - what the agent perceives in the world.
    Examples:
    - "Klaus is reading a book about gentrification"
    - "The sun is setting over the village"
    - "Maria looks tired today"
    """
    memory_type: MemoryType = MemoryType.OBSERVATION

    # Optional observation context
    observed_agent_id: Optional[UUID] = None
    observed_object: Optional[str] = None
    action_observed: Optional[str] = None

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        description: str,
        importance: float = 0.5,
        location_x: Optional[int] = None,
        location_y: Optional[int] = None,
        game_time: Optional[datetime] = None,
        observed_agent_id: Optional[UUID] = None,
    ) -> "Observation":
        """Factory method to create an observation"""
        return cls(
            agent_id=agent_id,
            description=description,
            importance=importance,
            location_x=location_x,
            location_y=location_y,
            game_time=game_time or datetime.utcnow(),
            observed_agent_id=observed_agent_id,
        )


class Reflection(Memory):
    """
    A reflection - higher-level inference from observations.
    Generated when importance threshold is exceeded.
    Examples:
    - "Klaus is deeply interested in gentrification research"
    - "Maria seems to be going through a difficult time"
    - "The village is preparing for winter"
    """
    memory_type: MemoryType = MemoryType.REFLECTION

    # Evidence: which memories this reflection is based on
    evidence_description: str = ""

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        description: str,
        source_memories: List[UUID],
        evidence_description: str = "",
        importance: float = 0.8,
        location_x: Optional[int] = None,
        location_y: Optional[int] = None,
        game_time: Optional[datetime] = None,
    ) -> "Reflection":
        """Factory method to create a reflection"""
        return cls(
            agent_id=agent_id,
            description=description,
            source_memories=source_memories,
            evidence_description=evidence_description,
            importance=importance,
            location_x=location_x,
            location_y=location_y,
            game_time=game_time or datetime.utcnow(),
        )


class Plan(Memory):
    """
    A plan - intended future action.
    Part of hierarchical planning (day -> hour -> action).
    Examples:
    - "Today I will work at the forge, then visit the tavern"
    - "At 3pm, I will meet Maria at the market"
    - "Right now I am crafting a horseshoe"
    """
    memory_type: MemoryType = MemoryType.PLAN

    # Plan hierarchy
    parent_plan_id: Optional[UUID] = None
    granularity: str = Field(default="action",
                             description="day, hour, or action")

    # Time bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Status
    is_completed: bool = False
    is_cancelled: bool = False
    completion_notes: Optional[str] = None

    @classmethod
    def create_day_plan(
        cls,
        agent_id: UUID,
        description: str,
        date: datetime,
        importance: float = 0.6,
    ) -> "Plan":
        """Create a day-level plan"""
        return cls(
            agent_id=agent_id,
            description=description,
            importance=importance,
            granularity="day",
            start_time=date.replace(hour=0, minute=0, second=0),
            end_time=date.replace(hour=23, minute=59, second=59),
            game_time=date,
        )

    @classmethod
    def create_hour_plan(
        cls,
        agent_id: UUID,
        description: str,
        start_time: datetime,
        parent_plan_id: Optional[UUID] = None,
        importance: float = 0.5,
    ) -> "Plan":
        """Create an hour-level plan"""
        from datetime import timedelta
        return cls(
            agent_id=agent_id,
            description=description,
            importance=importance,
            granularity="hour",
            parent_plan_id=parent_plan_id,
            start_time=start_time,
            end_time=start_time + timedelta(hours=1),
            game_time=start_time,
        )

    @classmethod
    def create_action_plan(
        cls,
        agent_id: UUID,
        description: str,
        start_time: datetime,
        duration_minutes: int = 15,
        parent_plan_id: Optional[UUID] = None,
        importance: float = 0.4,
    ) -> "Plan":
        """Create an action-level plan (5-15 minutes)"""
        from datetime import timedelta
        return cls(
            agent_id=agent_id,
            description=description,
            importance=importance,
            granularity="action",
            parent_plan_id=parent_plan_id,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=duration_minutes),
            game_time=start_time,
        )

    def complete(self, notes: Optional[str] = None) -> None:
        """Mark plan as completed"""
        self.is_completed = True
        self.completion_notes = notes

    def cancel(self, reason: Optional[str] = None) -> None:
        """Mark plan as cancelled"""
        self.is_cancelled = True
        self.completion_notes = reason


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval"""
    agent_id: UUID
    query_text: Optional[str] = None
    memory_types: List[MemoryType] = Field(
        default_factory=lambda: [MemoryType.OBSERVATION, MemoryType.REFLECTION, MemoryType.PLAN]
    )
    limit: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    since: Optional[datetime] = None
    location_radius: Optional[int] = None
    location_x: Optional[int] = None
    location_y: Optional[int] = None
