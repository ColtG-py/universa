"""
Agent Inspector
Provides detailed inspection of agent internal state for debugging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from uuid import UUID
from collections import deque
import logging

from agents.graph.state import AgentGraphState, AgentPhase
from agents.memory.memory_stream import MemoryStream
from agents.models.memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class CognitiveSnapshot:
    """Snapshot of agent's current cognitive state."""
    phase: str
    current_thought: Optional[str] = None
    current_perception: Optional[str] = None
    current_plan: Optional[str] = None
    current_action: Optional[str] = None
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    importance_accumulator: float = 0.0
    should_reflect: bool = False
    nodes_visited: List[str] = field(default_factory=list)


@dataclass
class PlanSnapshot:
    """Snapshot of agent's planning state."""
    day_plan: List[Dict[str, Any]] = field(default_factory=list)
    hour_plan: List[Dict[str, Any]] = field(default_factory=list)
    current_action: Optional[Dict[str, Any]] = None
    day_progress: float = 0.0
    hour_progress: float = 0.0


@dataclass
class RelationshipSnapshot:
    """Snapshot of relationship with another agent."""
    other_id: str
    other_name: str
    familiarity: float = 0.0
    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5
    relationship_type: Optional[str] = None
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    shared_memory_count: int = 0


@dataclass
class NeedsSnapshot:
    """Snapshot of agent's needs."""
    hunger: float = 1.0
    thirst: float = 1.0
    rest: float = 1.0
    warmth: float = 1.0
    safety: float = 1.0
    social: float = 0.5


@dataclass
class AgentSnapshot:
    """Complete snapshot of agent state for debugging."""
    agent_id: str
    agent_name: str
    timestamp: datetime

    # Location
    x: int = 0
    y: int = 0
    location_name: Optional[str] = None
    settlement_id: Optional[str] = None

    # Cognitive state
    cognitive: Optional[CognitiveSnapshot] = None

    # Planning state
    plans: Optional[PlanSnapshot] = None

    # Needs
    needs: Optional[NeedsSnapshot] = None

    # Relationships
    relationships: List[RelationshipSnapshot] = field(default_factory=list)

    # Memory stats
    memory_count: int = 0
    recent_observation_count: int = 0
    reflection_count: int = 0

    # Execution stats
    tier: str = "BACKGROUND"
    last_executed_tick: int = 0
    total_actions: int = 0
    execution_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "location": {
                "x": self.x,
                "y": self.y,
                "name": self.location_name,
                "settlement_id": self.settlement_id,
            },
            "cognitive": {
                "phase": self.cognitive.phase if self.cognitive else "idle",
                "current_thought": self.cognitive.current_thought if self.cognitive else None,
                "current_perception": self.cognitive.current_perception if self.cognitive else None,
                "current_plan": self.cognitive.current_plan if self.cognitive else None,
                "current_action": self.cognitive.current_action if self.cognitive else None,
                "importance_accumulator": self.cognitive.importance_accumulator if self.cognitive else 0,
                "should_reflect": self.cognitive.should_reflect if self.cognitive else False,
                "nodes_visited": self.cognitive.nodes_visited if self.cognitive else [],
                "retrieved_memory_count": len(self.cognitive.retrieved_memories) if self.cognitive else 0,
            } if self.cognitive else None,
            "plans": {
                "day_plan_count": len(self.plans.day_plan) if self.plans else 0,
                "hour_plan_count": len(self.plans.hour_plan) if self.plans else 0,
                "current_action": self.plans.current_action if self.plans else None,
                "day_progress": self.plans.day_progress if self.plans else 0,
                "hour_progress": self.plans.hour_progress if self.plans else 0,
            } if self.plans else None,
            "needs": {
                "hunger": self.needs.hunger,
                "thirst": self.needs.thirst,
                "rest": self.needs.rest,
                "warmth": self.needs.warmth,
                "safety": self.needs.safety,
                "social": self.needs.social,
            } if self.needs else None,
            "relationship_count": len(self.relationships),
            "memory_count": self.memory_count,
            "recent_observation_count": self.recent_observation_count,
            "reflection_count": self.reflection_count,
            "execution": {
                "tier": self.tier,
                "last_executed_tick": self.last_executed_tick,
                "total_actions": self.total_actions,
                "execution_duration_ms": self.execution_duration_ms,
            },
        }


class AgentInspector:
    """
    Provides inspection capabilities for agent debugging.

    Maintains snapshots of agent state and provides APIs
    for examining agent internals.
    """

    def __init__(self, max_snapshots: int = 100):
        self.max_snapshots = max_snapshots

        # Agent snapshots (agent_id -> deque of snapshots)
        self._snapshots: Dict[UUID, deque] = {}

        # Current state references (set by orchestrator)
        self._agent_states: Dict[UUID, AgentGraphState] = {}
        self._memory_streams: Dict[UUID, MemoryStream] = {}
        self._agent_graphs: Dict[UUID, Any] = {}  # AgentGraph references

        # Relationship data
        self._relationships: Dict[UUID, Dict[UUID, RelationshipSnapshot]] = {}

    def register_agent(
        self,
        agent_id: UUID,
        memory_stream: Optional[MemoryStream] = None,
        agent_graph: Optional[Any] = None,
    ) -> None:
        """Register an agent for inspection."""
        if agent_id not in self._snapshots:
            self._snapshots[agent_id] = deque(maxlen=self.max_snapshots)

        if memory_stream:
            self._memory_streams[agent_id] = memory_stream

        if agent_graph:
            self._agent_graphs[agent_id] = agent_graph

    def unregister_agent(self, agent_id: UUID) -> None:
        """Unregister an agent."""
        self._snapshots.pop(agent_id, None)
        self._agent_states.pop(agent_id, None)
        self._memory_streams.pop(agent_id, None)
        self._agent_graphs.pop(agent_id, None)
        self._relationships.pop(agent_id, None)

    def update_state(
        self,
        agent_id: UUID,
        state: AgentGraphState,
    ) -> None:
        """Update tracked state for an agent."""
        self._agent_states[agent_id] = state

    def capture_snapshot(
        self,
        agent_id: UUID,
        agent_name: str,
        x: int = 0,
        y: int = 0,
        tier: str = "BACKGROUND",
        last_tick: int = 0,
    ) -> Optional[AgentSnapshot]:
        """
        Capture a snapshot of agent state.

        Args:
            agent_id: Agent to snapshot
            agent_name: Agent's name
            x: X position
            y: Y position
            tier: Current execution tier
            last_tick: Last executed tick

        Returns:
            AgentSnapshot or None if agent not found
        """
        state = self._agent_states.get(agent_id)
        memory_stream = self._memory_streams.get(agent_id)

        # Build cognitive snapshot from state
        cognitive = None
        if state:
            cognitive = CognitiveSnapshot(
                phase=state.phase.value if state.phase else "idle",
                current_thought=state.plan_context.get("thought") if state.plan_context else None,
                current_perception=state.perception.observations[0] if state.perception and state.perception.observations else None,
                current_plan=state.current_plan.description if state.current_plan else None,
                current_action=state.current_action,
                retrieved_memories=[
                    {"description": m.description, "importance": m.importance}
                    for m in state.retrieved_memories[:10]
                ] if state.retrieved_memories else [],
                importance_accumulator=state.importance_accumulator,
                should_reflect=state.should_reflect,
                nodes_visited=state.nodes_visited or [],
            )

        # Build plan snapshot
        plans = None
        graph = self._agent_graphs.get(agent_id)
        if graph and hasattr(graph, 'planning_system'):
            ps = graph.planning_system
            if hasattr(ps, 'get_plan_context'):
                ctx = ps.get_plan_context()
                plans = PlanSnapshot(
                    day_plan=ctx.get("day_plan", []),
                    hour_plan=ctx.get("hour_plan", []),
                    current_action=ctx.get("current_action"),
                    day_progress=ctx.get("day_progress", 0),
                    hour_progress=ctx.get("hour_progress", 0),
                )

        # Build needs snapshot (mock for now, would come from agent state)
        needs = NeedsSnapshot()

        # Get memory stats
        memory_count = 0
        observation_count = 0
        reflection_count = 0
        if memory_stream:
            memory_count = len(memory_stream._recent_cache)
            observation_count = sum(1 for m in memory_stream._recent_cache if hasattr(m, 'memory_type') and str(m.memory_type) == 'observation')
            reflection_count = sum(1 for m in memory_stream._recent_cache if hasattr(m, 'memory_type') and str(m.memory_type) == 'reflection')

        # Get relationships
        relationships = list(self._relationships.get(agent_id, {}).values())

        snapshot = AgentSnapshot(
            agent_id=str(agent_id),
            agent_name=agent_name,
            timestamp=datetime.utcnow(),
            x=x,
            y=y,
            cognitive=cognitive,
            plans=plans,
            needs=needs,
            relationships=relationships,
            memory_count=memory_count,
            recent_observation_count=observation_count,
            reflection_count=reflection_count,
            tier=tier,
            last_executed_tick=last_tick,
            execution_duration_ms=state.execution_duration_ms if state else 0,
        )

        # Store snapshot
        if agent_id not in self._snapshots:
            self._snapshots[agent_id] = deque(maxlen=self.max_snapshots)
        self._snapshots[agent_id].append(snapshot)

        return snapshot

    def get_latest_snapshot(self, agent_id: UUID) -> Optional[AgentSnapshot]:
        """Get the most recent snapshot for an agent."""
        snapshots = self._snapshots.get(agent_id)
        if snapshots:
            return snapshots[-1]
        return None

    def get_snapshot_history(
        self,
        agent_id: UUID,
        limit: int = 20
    ) -> List[AgentSnapshot]:
        """Get snapshot history for an agent."""
        snapshots = self._snapshots.get(agent_id, deque())
        return list(reversed(list(snapshots)))[:limit]

    def get_cognitive_state(self, agent_id: UUID) -> Optional[CognitiveSnapshot]:
        """Get current cognitive state for an agent."""
        snapshot = self.get_latest_snapshot(agent_id)
        return snapshot.cognitive if snapshot else None

    def get_plans(self, agent_id: UUID) -> Optional[PlanSnapshot]:
        """Get current planning state for an agent."""
        snapshot = self.get_latest_snapshot(agent_id)
        return snapshot.plans if snapshot else None

    def get_needs(self, agent_id: UUID) -> Optional[NeedsSnapshot]:
        """Get current needs state for an agent."""
        snapshot = self.get_latest_snapshot(agent_id)
        return snapshot.needs if snapshot else None

    def update_relationship(
        self,
        agent_id: UUID,
        other_id: UUID,
        other_name: str,
        familiarity: float = 0.0,
        trust: float = 0.5,
        affection: float = 0.5,
        respect: float = 0.5,
        relationship_type: Optional[str] = None,
    ) -> None:
        """Update relationship data for an agent."""
        if agent_id not in self._relationships:
            self._relationships[agent_id] = {}

        self._relationships[agent_id][other_id] = RelationshipSnapshot(
            other_id=str(other_id),
            other_name=other_name,
            familiarity=familiarity,
            trust=trust,
            affection=affection,
            respect=respect,
            relationship_type=relationship_type,
            last_interaction=datetime.utcnow(),
            interaction_count=self._relationships[agent_id].get(other_id, RelationshipSnapshot(
                other_id=str(other_id), other_name=other_name
            )).interaction_count + 1,
        )

    def get_relationships(
        self,
        agent_id: UUID,
        min_familiarity: float = 0.0
    ) -> List[RelationshipSnapshot]:
        """Get relationships for an agent."""
        rels = self._relationships.get(agent_id, {})
        return [r for r in rels.values() if r.familiarity >= min_familiarity]

    def get_relationship(
        self,
        agent_id: UUID,
        other_id: UUID
    ) -> Optional[RelationshipSnapshot]:
        """Get specific relationship."""
        return self._relationships.get(agent_id, {}).get(other_id)

    async def get_memories(
        self,
        agent_id: UUID,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get memories for an agent."""
        memory_stream = self._memory_streams.get(agent_id)
        if not memory_stream:
            return []

        memories = memory_stream._recent_cache

        # Filter by type
        if memory_type:
            memories = [m for m in memories if hasattr(m, 'memory_type') and str(m.memory_type) == memory_type]

        # Filter by importance
        memories = [m for m in memories if m.importance >= min_importance]

        # Sort by recency
        memories = sorted(memories, key=lambda m: m.created_at, reverse=True)

        return [
            {
                "id": str(m.memory_id) if hasattr(m, 'memory_id') else str(id(m)),
                "description": m.description,
                "type": str(m.memory_type) if hasattr(m, 'memory_type') else "unknown",
                "importance": m.importance,
                "created_at": m.created_at.isoformat() if hasattr(m, 'created_at') else None,
                "access_count": m.access_count if hasattr(m, 'access_count') else 0,
            }
            for m in memories[:limit]
        ]

    async def search_memories(
        self,
        agent_id: UUID,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search agent memories."""
        memory_stream = self._memory_streams.get(agent_id)
        if not memory_stream:
            return []

        try:
            matches = await memory_stream.search_by_text(query, limit=limit)
            return [
                {
                    "id": str(m.memory_id) if hasattr(m, 'memory_id') else str(id(m)),
                    "description": m.description,
                    "type": str(m.memory_type) if hasattr(m, 'memory_type') else "unknown",
                    "importance": m.importance,
                    "relevance_score": 0.8,  # Placeholder
                }
                for m in matches
            ]
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return []


# Global inspector instance
_inspector: Optional[AgentInspector] = None


def get_inspector() -> AgentInspector:
    """Get the global agent inspector."""
    global _inspector
    if _inspector is None:
        _inspector = AgentInspector()
    return _inspector
