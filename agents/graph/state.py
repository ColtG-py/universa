"""
Agent Graph State
Defines the state that flows through the agent execution graph.
"""

from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum

from agents.models.memory import Memory, Observation, Reflection, Plan


class AgentPhase(str, Enum):
    """Current phase in the agent cycle"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    RETRIEVING = "retrieving"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    ACTING = "acting"
    REACTING = "reacting"


class ActionType(str, Enum):
    """Types of actions an agent can take"""
    MOVE = "move"
    INTERACT = "interact"
    USE_SKILL = "use_skill"
    SPEAK = "speak"
    OBSERVE = "observe"
    REST = "rest"
    WAIT = "wait"


@dataclass
class PerceptionData:
    """Data from perceiving the environment"""
    location: Optional[str] = None
    location_x: Optional[int] = None
    location_y: Optional[int] = None
    nearby_agents: List[Dict[str, Any]] = field(default_factory=list)
    nearby_objects: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActionResult:
    """Result of an action"""
    action_type: ActionType
    success: bool
    description: str
    effects: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentGraphState:
    """
    State that flows through the agent execution graph.

    This implements the Stanford paper's agent architecture:
    Perceive → Retrieve → (Reflect) → Plan → Act

    The state is passed between nodes and updated as the agent
    processes information and decides on actions.
    """

    # Agent identity
    agent_id: UUID = None
    agent_name: str = ""
    agent_summary: str = ""

    # Current phase
    phase: AgentPhase = AgentPhase.IDLE

    # Simulation time
    current_time: datetime = field(default_factory=datetime.utcnow)
    time_step: int = 0

    # Perception data
    perception: Optional[PerceptionData] = None
    new_observations: List[Observation] = field(default_factory=list)

    # Retrieved memories
    retrieved_memories: List[Memory] = field(default_factory=list)
    relevance_query: Optional[str] = None

    # Reflection state
    should_reflect: bool = False
    new_reflections: List[Reflection] = field(default_factory=list)
    importance_accumulator: float = 0.0

    # Planning state
    current_plan: Optional[Plan] = None
    current_action: Optional[str] = None
    plan_context: Dict[str, Any] = field(default_factory=dict)

    # Action state
    pending_action: Optional[Dict[str, Any]] = None
    last_action_result: Optional[ActionResult] = None

    # Reaction state (for interrupts)
    interrupt_event: Optional[str] = None
    reaction_decision: Optional[Dict[str, Any]] = None

    # Conversation state
    active_conversation: Optional[UUID] = None
    conversation_partner: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0

    # Execution metadata
    nodes_visited: List[str] = field(default_factory=list)
    execution_start: Optional[datetime] = None
    execution_duration_ms: float = 0.0

    def mark_phase(self, phase: AgentPhase) -> None:
        """Update current phase and track node visit"""
        self.phase = phase
        self.nodes_visited.append(phase.value)

    def add_observation(self, observation: Observation) -> None:
        """Add a new observation"""
        self.new_observations.append(observation)
        # Track importance for reflection trigger
        self.importance_accumulator += observation.importance * 10

    def clear_for_next_cycle(self) -> None:
        """Clear transient state for next execution cycle"""
        self.new_observations = []
        self.new_reflections = []
        self.retrieved_memories = []
        self.pending_action = None
        self.interrupt_event = None
        self.reaction_decision = None
        self.error = None
        self.nodes_visited = []
        self.execution_start = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "agent_name": self.agent_name,
            "phase": self.phase.value,
            "current_time": self.current_time.isoformat(),
            "time_step": self.time_step,
            "should_reflect": self.should_reflect,
            "importance_accumulator": self.importance_accumulator,
            "current_action": self.current_action,
            "error": self.error,
            "nodes_visited": self.nodes_visited,
        }

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        agent_name: str,
        agent_summary: str = "",
        current_time: Optional[datetime] = None,
    ) -> "AgentGraphState":
        """Create a new agent state"""
        return cls(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_summary=agent_summary,
            current_time=current_time or datetime.utcnow(),
            execution_start=datetime.utcnow(),
        )


def merge_states(
    current: AgentGraphState,
    update: Dict[str, Any]
) -> AgentGraphState:
    """
    Merge an update dictionary into the current state.

    Used by LangGraph-style reducers.
    """
    for key, value in update.items():
        if hasattr(current, key):
            setattr(current, key, value)
    return current
