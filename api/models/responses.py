"""
API Response Models
Pydantic models for API response serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class AgentTier(str, Enum):
    ACTIVE = "active"       # Full reasoning every tick
    NEARBY = "nearby"       # Reduced reasoning frequency
    BACKGROUND = "background"  # Minimal updates
    DORMANT = "dormant"     # No updates until triggered


class CognitivePhase(str, Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    RETRIEVING = "retrieving"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    ACTING = "acting"


# =============================================================================
# World Responses
# =============================================================================

class TileData(BaseModel):
    """Single tile data."""
    x: int
    y: int
    biome: str
    elevation: float
    temperature: float
    moisture: float
    is_water: bool = False
    settlement_id: Optional[str] = None
    road_level: int = 0
    river: bool = False
    magic_level: float = 0.0


class WorldResponse(BaseModel):
    """Response for world creation/retrieval."""
    world_id: str
    name: str
    seed: Optional[int] = None
    size: Optional[str] = None
    created_at: Optional[datetime] = None
    status: str = "ready"
    message: Optional[str] = None

    # World stats
    num_settlements: int = 0
    num_agents: int = 0
    num_tiles: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "world_id": "world-123",
                "name": "My Fantasy World",
                "seed": 12345,
                "size": "MEDIUM",
                "created_at": "2024-01-01T00:00:00Z",
                "status": "ready",
                "num_settlements": 15,
                "num_agents": 150,
                "num_tiles": 1048576
            }
        }


class WorldListResponse(BaseModel):
    """List of worlds."""
    worlds: List[WorldResponse]
    total: int


class ChunkResponse(BaseModel):
    """Response for world chunk data."""
    world_id: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    tiles: List[TileData]
    settlements: List[Dict[str, Any]] = []
    agents: List[Dict[str, Any]] = []
    roads: List[Dict[str, Any]] = []
    rivers: List[Dict[str, Any]] = []


class WorldChunksResponse(BaseModel):
    """Response for world chunks query."""
    world_id: str
    chunks: List[Dict[str, Any]] = []
    bounds: Dict[str, int] = {}


class GenerationProgressResponse(BaseModel):
    """World generation progress."""
    world_id: str
    status: str  # generating, complete, failed
    current_pass: int
    total_passes: int
    pass_name: str
    progress_percent: float
    estimated_remaining_seconds: Optional[float] = None


# =============================================================================
# Game Session Responses
# =============================================================================

class SessionResponse(BaseModel):
    """Game session details."""
    id: str
    world_id: str
    player_id: str
    party_id: Optional[str] = None
    created_at: datetime
    status: str  # active, paused, ended
    current_tick: int = 0
    game_time: str  # In-game time string
    settings: Dict[str, Any] = {}


class AgentUpdateEntry(BaseModel):
    """Single agent update during a tick."""
    agent_id: str
    changes: Dict[str, Any] = {}


class TickResponse(BaseModel):
    """Response after simulation tick."""
    session_id: str
    tick_number: int
    game_time: str
    events: List[Dict[str, Any]] = []
    agent_updates: List[AgentUpdateEntry] = []
    agents_updated: int = 0
    duration_ms: float = 0


class AutoTickResponse(BaseModel):
    """Response for auto-tick control."""
    session_id: str
    auto_tick_enabled: bool
    interval_ms: int


class SessionStateResponse(BaseModel):
    """Current state of a game session."""
    session_id: str
    status: str
    current_tick: int
    game_time: str
    player_position: Optional[Dict[str, int]] = None
    party_size: int = 0
    auto_tick_enabled: bool = False


# =============================================================================
# Agent Responses
# =============================================================================

class AgentResponse(BaseModel):
    """Agent details."""
    id: str
    name: str
    x: int
    y: int

    # Type info
    agent_type: str  # individual, settlement, kingdom
    role: Optional[str] = None  # warrior, mage, merchant, etc.

    # State
    health: float = 1.0
    energy: float = 1.0
    tier: AgentTier = AgentTier.BACKGROUND

    # Current activity
    current_action: Optional[str] = None
    current_thought: Optional[str] = None

    # Affiliation
    settlement_id: Optional[str] = None
    faction_id: Optional[str] = None

    # Stats
    stats: Dict[str, int] = {}
    traits: List[str] = []


class AgentListResponse(BaseModel):
    """List of agents."""
    agents: List[AgentResponse]
    total: int
    by_tier: Dict[str, int] = {}


class DialogueResponse(BaseModel):
    """Response from agent dialogue."""
    agent_id: str
    agent_name: str
    message: str
    emotion: Optional[str] = None

    # Conversation tracking
    conversation_id: str
    turn_number: int

    # Debug info (if enabled)
    memories_retrieved: Optional[List[str]] = None
    reasoning: Optional[str] = None


# =============================================================================
# Player Responses
# =============================================================================

class PlayerResponse(BaseModel):
    """Player character details."""
    id: str
    name: str
    x: int
    y: int

    # Vitals
    health: float = 100.0
    max_health: float = 100.0
    stamina: float = 100.0
    max_stamina: float = 100.0

    # Stats
    stats: Dict[str, int] = {}

    # Inventory summary
    gold: int = 0
    inventory_slots_used: int = 0
    inventory_slots_total: int = 20


class InteractionResponse(BaseModel):
    """Response from player interaction."""
    success: bool
    action: str
    target_type: str
    target_id: str
    result: Dict[str, Any] = {}
    message: Optional[str] = None


class ObservationResponse(BaseModel):
    """What the player can currently observe."""
    player_id: str
    x: int
    y: int

    # Current tile
    current_tile: TileData

    # Nearby entities
    nearby_agents: List[Dict[str, Any]] = []
    nearby_objects: List[Dict[str, Any]] = []
    nearby_settlements: List[Dict[str, Any]] = []

    # Events
    recent_events: List[Dict[str, Any]] = []


# =============================================================================
# Party Responses
# =============================================================================

class PartyMemberResponse(BaseModel):
    """Party member details."""
    id: str
    name: str
    role: str

    # Position relative to player
    x: int
    y: int
    distance_to_player: float = 0

    # State
    health: float = 1.0
    energy: float = 1.0
    morale: float = 1.0

    # Current activity
    current_action: Optional[str] = None
    following: bool = True

    # Relationship with player
    loyalty: float = 0.5
    trust: float = 0.5
    affection: float = 0.5


class PartyResponse(BaseModel):
    """Party details."""
    id: str
    leader_id: str
    members: List[PartyMemberResponse]

    # Party state
    formation: str = "follow"  # follow, spread, guard
    average_morale: float = 1.0

    # Position
    center_x: float = 0
    center_y: float = 0


# =============================================================================
# Debug Responses
# =============================================================================

class MemoryEntry(BaseModel):
    """Single memory entry."""
    id: str
    content: str
    memory_type: str  # observation, reflection, plan
    importance: float
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    embedding: Optional[List[float]] = None


class MemoryListResponse(BaseModel):
    """List of agent memories."""
    agent_id: str
    memories: List[MemoryEntry]
    total: int


class AgentThoughtsResponse(BaseModel):
    """Agent's current cognitive state."""
    agent_id: str
    phase: CognitivePhase
    current_thought: Optional[str] = None

    # What's being processed
    current_perception: Optional[str] = None
    retrieved_memories: List[str] = []
    pending_action: Optional[str] = None

    # Timing
    last_think_time: Optional[datetime] = None
    think_duration_ms: Optional[float] = None


class PlanEntry(BaseModel):
    """A single plan entry."""
    description: str
    start_time: str
    duration_minutes: int
    location: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, abandoned


class AgentPlansResponse(BaseModel):
    """Agent's planning hierarchy."""
    agent_id: str

    # Plan levels
    day_plan: List[PlanEntry] = []
    hour_plan: List[PlanEntry] = []
    current_action: Optional[PlanEntry] = None

    # Progress
    day_progress: float = 0.0  # 0-1
    hour_progress: float = 0.0


class RelationshipEntry(BaseModel):
    """Relationship with another agent."""
    other_id: str
    other_name: str

    # Relationship dimensions
    familiarity: float = 0.0  # How well they know each other
    trust: float = 0.5       # Trust level
    affection: float = 0.5   # Like/dislike
    respect: float = 0.5     # Professional respect

    # Context
    relationship_type: Optional[str] = None  # friend, rival, family, etc.
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0


class RelationshipListResponse(BaseModel):
    """List of agent relationships."""
    agent_id: str
    relationships: List[RelationshipEntry]
    total: int


class LLMCallEntry(BaseModel):
    """Record of an LLM call."""
    id: str
    timestamp: datetime

    # Call details
    purpose: str  # perceive, reflect, plan, dialogue, etc.
    prompt_summary: str
    response_summary: str

    # Performance
    tokens_in: int
    tokens_out: int
    duration_ms: float
    model: str


class LLMCallHistoryResponse(BaseModel):
    """History of LLM calls for an agent."""
    agent_id: str
    calls: List[LLMCallEntry]
    total: int

    # Aggregates
    total_tokens: int = 0
    total_cost_estimate: float = 0.0


class SimulationStatsResponse(BaseModel):
    """Overall simulation statistics."""
    session_id: str
    current_tick: int
    game_time: str

    # Agent counts
    total_agents: int = 0
    active_agents: int = 0
    agents_by_tier: Dict[str, int] = {}

    # Performance
    avg_tick_duration_ms: float = 0
    llm_calls_per_tick: float = 0
    memory_operations_per_tick: float = 0

    # World state
    total_settlements: int = 0
    total_factions: int = 0
    active_events: int = 0


# =============================================================================
# WebSocket Messages
# =============================================================================

class WSMessage(BaseModel):
    """Base WebSocket message."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str


class WSTickUpdate(WSMessage):
    """Tick update broadcast."""
    type: str = "tick"
    tick_number: int
    game_time: str
    events: List[Dict[str, Any]] = []


class WSAgentUpdate(WSMessage):
    """Agent state update."""
    type: str = "agent_update"
    agent_id: str
    changes: Dict[str, Any]


class WSChatMessage(WSMessage):
    """Chat message broadcast."""
    type: str = "chat"
    channel: str
    sender_id: str
    sender_name: str
    message: str


class WSEventNotification(WSMessage):
    """World event notification."""
    type: str = "event"
    event_type: str
    description: str
    location: Optional[Dict[str, int]] = None
    involved_agents: List[str] = []
