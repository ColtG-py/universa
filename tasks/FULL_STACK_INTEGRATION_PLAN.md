# Universa Full-Stack Integration Plan

## Overview

This document outlines the plan to fully integrate the world generation system (`/builder`), agent framework (`/agents`), and visualization layer (`/frontend/universa`) into a cohesive, playable experience.

**Goals:**
1. World generation and parameterization through the UI
2. Agent instantiation with configurable party sizes
3. Agents interact with Ollama for agentic behavior
4. Memory and thinking debug tools for agent execution
5. Human player can create a world, spawn with a party, and interact
6. World ticks with agent-to-agent interactions beyond the rendered canvas
7. Hierarchical agent structures (villages, kingdoms) to reduce concurrent execution
8. Debug mode for agent thinking, planning, and memory inspection

---

## Part 1: System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │ World Setup │  │ Game Canvas │  │ Chat/Dialog │  │ Debug Panel         ││
│  │ (Create)    │  │ (PixiJS)    │  │ (Party/NPC) │  │ (Memory/Thinking)   ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
└─────────┼────────────────┼────────────────┼────────────────────┼───────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │ /worlds     │  │ /game       │  │ /agents     │  │ /debug              ││
│  │ - create    │  │ - tick      │  │ - actions   │  │ - memories          ││
│  │ - load      │  │ - state     │  │ - dialogue  │  │ - thoughts          ││
│  │ - params    │  │ - events    │  │ - party     │  │ - plans             ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
└─────────┼────────────────┼────────────────┼────────────────────┼───────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIMULATION ENGINE                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │ World       │  │ Simulation  │  │ Agent       │  │ Hierarchical        ││
│  │ Builder     │  │ Orchestrator│  │ Graph       │  │ Scheduler           ││
│  │ (18 passes) │  │ (Ticks)     │  │ (LangGraph) │  │ (Villages/Kingdoms) ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
└─────────┼────────────────┼────────────────┼────────────────────┼───────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │ Supabase (PostgreSQL)   │  │ Ollama (Local LLM)                      │  │
│  │ - World chunks          │  │ - qwen2.5:7b (reasoning)                │  │
│  │ - Agent state           │  │ - nomic-embed-text (embeddings)         │  │
│  │ - Memory stream         │  │ - Tool calling support                  │  │
│  │ - Relationships         │  │                                         │  │
│  └─────────────────────────┘  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
User Action → Frontend → API → Simulation → Database/LLM → API → Frontend (Real-time)
                                    ↓
                              Supabase Realtime
                                    ↓
                              Frontend Updates
```

---

## Part 2: Implementation Phases

### Phase 1: API Layer Foundation (Week 1)

Create the FastAPI backend that bridges all systems.

#### 1.1 Project Structure

```
universa/
├── api/                          # NEW: FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Environment configuration
│   │
│   ├── routers/                  # API endpoints
│   │   ├── __init__.py
│   │   ├── worlds.py             # World creation/management
│   │   ├── game.py               # Game session management
│   │   ├── agents.py             # Agent interactions
│   │   ├── player.py             # Player actions
│   │   ├── party.py              # Party management
│   │   └── debug.py              # Debug/inspection endpoints
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── world_service.py      # World generation orchestration
│   │   ├── game_service.py       # Game session logic
│   │   ├── agent_service.py      # Agent execution
│   │   └── party_service.py      # Party mechanics
│   │
│   ├── models/                   # Pydantic models for API
│   │   ├── __init__.py
│   │   ├── requests.py           # Request schemas
│   │   └── responses.py          # Response schemas
│   │
│   └── websocket/                # Real-time communication
│       ├── __init__.py
│       ├── manager.py            # WebSocket connection manager
│       └── handlers.py           # Message handlers
```

#### 1.2 Core API Endpoints

```python
# World Management
POST   /api/worlds                    # Create new world with params
GET    /api/worlds                    # List user's worlds
GET    /api/worlds/{world_id}         # Get world details
GET    /api/worlds/{world_id}/chunks  # Get world chunk data
DELETE /api/worlds/{world_id}         # Delete world

# Game Sessions
POST   /api/game/sessions             # Start new game session
GET    /api/game/sessions/{id}        # Get session state
POST   /api/game/sessions/{id}/tick   # Advance simulation (manual)
POST   /api/game/sessions/{id}/start  # Start auto-tick
POST   /api/game/sessions/{id}/pause  # Pause simulation
POST   /api/game/sessions/{id}/stop   # Stop and save

# Player Actions
POST   /api/player/create             # Create player character
POST   /api/player/move               # Move player
POST   /api/player/interact           # Interact with world/agent
POST   /api/player/speak              # Send message

# Party Management
POST   /api/party/create              # Create party with agents
GET    /api/party/{id}                # Get party state
POST   /api/party/{id}/command        # Issue party command
POST   /api/party/{id}/agents/{agent_id}/dialogue  # Talk to party member

# Agent Interactions
GET    /api/agents/{id}               # Get agent details
POST   /api/agents/{id}/dialogue      # Initiate dialogue
GET    /api/agents/nearby             # Get nearby agents

# Debug Endpoints
GET    /api/debug/agents/{id}/memories     # Get agent memories
GET    /api/debug/agents/{id}/thoughts     # Get current thinking
GET    /api/debug/agents/{id}/plans        # Get agent plans
GET    /api/debug/agents/{id}/relationships # Get relationships
GET    /api/debug/simulation/stats         # Get simulation stats

# WebSocket
WS     /ws/game/{session_id}          # Real-time game updates
```

#### 1.3 Key Files to Create

**`api/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import worlds, game, agents, player, party, debug
from api.websocket import manager

app = FastAPI(title="Universa API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(worlds.router, prefix="/api/worlds", tags=["worlds"])
app.include_router(game.router, prefix="/api/game", tags=["game"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(player.router, prefix="/api/player", tags=["player"])
app.include_router(party.router, prefix="/api/party", tags=["party"])
app.include_router(debug.router, prefix="/api/debug", tags=["debug"])

# WebSocket endpoint
@app.websocket("/ws/game/{session_id}")
async def game_websocket(websocket, session_id: str):
    await manager.game_connection(websocket, session_id)
```

---

### Phase 2: World Generation Integration (Week 2)

Connect the builder pipeline to the API and database.

#### 2.1 World Generation Service

```python
# api/services/world_service.py

from builder.generation.pipeline import GenerationPipeline
from builder.models.world import WorldState, WorldGenerationParams
from agents.db.supabase_client import SupabaseClient

class WorldService:
    def __init__(self):
        self.db = SupabaseClient()
        self.pipeline = GenerationPipeline()

    async def create_world(self, params: WorldGenerationParams) -> str:
        """Generate a new world and store in database."""
        # 1. Generate world using builder pipeline
        world_state = await self.pipeline.generate(params)

        # 2. Store world metadata
        world_id = await self.db.create_world(
            seed=params.seed,
            size=params.size,
            params=params.dict()
        )

        # 3. Store chunks (batch insert)
        for chunk_key, chunk in world_state.chunks.items():
            await self.db.insert_chunk(world_id, chunk_key, chunk.to_dict())

        # 4. Store settlements
        for settlement in world_state.settlements:
            await self.db.insert_settlement(world_id, settlement)

        # 5. Store factions
        for faction in world_state.factions:
            await self.db.insert_faction(world_id, faction)

        return world_id

    async def get_world_for_display(self, world_id: str, viewport: dict) -> dict:
        """Get world data optimized for frontend display."""
        # Only return chunks within viewport + buffer
        chunks = await self.db.get_chunks_in_range(
            world_id,
            x_min=viewport['x'] - 32,
            x_max=viewport['x'] + viewport['width'] + 32,
            y_min=viewport['y'] - 32,
            y_max=viewport['y'] + viewport['height'] + 32
        )
        return {
            'chunks': chunks,
            'settlements': await self.db.get_settlements(world_id),
            'roads': await self.db.get_roads(world_id)
        }
```

#### 2.2 Database Schema for Worlds

```sql
-- Add to Supabase migrations

-- World chunks storage (compressed)
CREATE TABLE world_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id) ON DELETE CASCADE,
    chunk_key VARCHAR(20) NOT NULL,  -- "x,y" format
    chunk_x INT NOT NULL,
    chunk_y INT NOT NULL,
    data JSONB NOT NULL,  -- Compressed chunk data
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(world_id, chunk_key)
);

-- Spatial index for chunk queries
CREATE INDEX idx_chunk_position ON world_chunks(world_id, chunk_x, chunk_y);

-- Settlements
CREATE TABLE settlements (
    settlement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    settlement_type VARCHAR(50) NOT NULL,
    x INT NOT NULL,
    y INT NOT NULL,
    population INT DEFAULT 0,
    faction_id UUID,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Factions
CREATE TABLE factions (
    faction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    faction_type VARCHAR(50) NOT NULL,
    capital_settlement_id UUID,
    territory_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 2.3 Frontend World Creation UI

Create new page: `app/worlds/create/page.tsx`

```typescript
// World Creation Form
interface WorldParams {
  name: string;
  seed?: number;
  size: 'SMALL' | 'MEDIUM' | 'LARGE' | 'HUGE';

  // Advanced options
  planetRadius?: number;
  axialTilt?: number;
  oceanPercentage?: number;
  numPlates?: number;

  // Feature toggles
  enableMagic: boolean;
  enableCaves: boolean;
  settlementDensity: 'low' | 'medium' | 'high';
}

// Component shows:
// - Basic settings (name, seed, size)
// - Expandable advanced settings
// - Preview map during generation
// - Progress bar with pass names
// - "Start Game" button when complete
```

---

### Phase 3: Game Session & Player Setup (Week 3)

#### 3.1 Game Session Management

```python
# api/services/game_service.py

from agents.simulation.orchestrator import SimulationOrchestrator
from agents.simulation.time_manager import TimeManager
from agents.graph.agent_graph import AgentGraphRunner

class GameService:
    def __init__(self):
        self.sessions: dict[str, GameSession] = {}

    async def create_session(
        self,
        world_id: str,
        player_config: PlayerConfig,
        party_config: PartyConfig
    ) -> str:
        """Create a new game session with player and party."""
        session_id = str(uuid4())

        # 1. Load world state
        world = await self.load_world(world_id)

        # 2. Create player character
        player = await self.create_player(player_config, world)

        # 3. Create party agents
        party = await self.create_party(party_config, player, world)

        # 4. Initialize simulation orchestrator
        orchestrator = SimulationOrchestrator(
            world_interface=world.interface,
            time_manager=TimeManager(),
            agent_runner=AgentGraphRunner()
        )

        # 5. Register all agents (party + world NPCs)
        for agent in party.members:
            orchestrator.register_agent(agent)

        # 6. Load nearby world agents
        nearby_agents = await self.load_nearby_agents(world_id, player.x, player.y)
        for agent in nearby_agents:
            orchestrator.register_agent(agent)

        # 7. Create session
        session = GameSession(
            id=session_id,
            world_id=world_id,
            player=player,
            party=party,
            orchestrator=orchestrator,
            state='ready'
        )

        self.sessions[session_id] = session
        return session_id

    async def tick(self, session_id: str) -> TickResult:
        """Advance simulation by one tick."""
        session = self.sessions[session_id]

        # Run simulation tick
        result = await session.orchestrator.tick()

        # Collect updates for frontend
        updates = {
            'time': session.orchestrator.time_manager.current_time,
            'agent_updates': result.agent_updates,
            'events': result.events,
            'player_observations': self.get_player_observations(session)
        }

        # Broadcast via WebSocket
        await self.broadcast_updates(session_id, updates)

        return updates
```

#### 3.2 Party System

```python
# api/services/party_service.py

from agents.models.agent_state import AgentState
from agents.reasoning.dialogue import DialogueSystem

class PartyService:
    """Manages the player's party of agents."""

    async def create_party(
        self,
        player: PlayerCharacter,
        config: PartyConfig
    ) -> Party:
        """Create a party with configurable size and composition."""
        party = Party(
            leader=player,
            members=[],
            formation='follow'  # follow, spread, guard
        )

        # Generate party members based on config
        for i in range(config.size):
            agent = await self.generate_party_member(
                index=i,
                role=config.roles[i] if i < len(config.roles) else 'companion',
                player=player
            )
            party.members.append(agent)

            # Initialize relationship with player
            agent.relationships.add(
                player.id,
                familiarity=0.8,  # They know the player well
                trust=0.7,
                affection=0.5
            )

        return party

    async def generate_party_member(
        self,
        index: int,
        role: str,
        player: PlayerCharacter
    ) -> AgentState:
        """Generate a party member agent."""
        # Use templates based on role
        templates = {
            'warrior': {'str': 14, 'con': 13, 'occupation': 'Warrior'},
            'mage': {'int': 14, 'wis': 13, 'occupation': 'Mage'},
            'healer': {'wis': 14, 'cha': 13, 'occupation': 'Healer'},
            'rogue': {'dex': 14, 'int': 13, 'occupation': 'Rogue'},
            'companion': {'cha': 12, 'wis': 12, 'occupation': 'Companion'}
        }

        template = templates.get(role, templates['companion'])

        # Generate agent with role-appropriate stats
        agent = AgentState(
            name=await self.generate_name(),
            x=player.x,
            y=player.y,
            stats=self.generate_stats(template),
            traits={
                'occupation': template['occupation'],
                'personality': await self.generate_personality(),
                'goals': [f'Assist {player.name}', 'Survive', 'Grow stronger']
            }
        )

        return agent

    async def party_dialogue(
        self,
        party: Party,
        speaker_id: str,
        message: str
    ) -> list[DialogueResponse]:
        """Handle dialogue within the party."""
        responses = []

        for member in party.members:
            if member.id == speaker_id:
                continue

            # Determine if member should respond
            should_respond = await self.should_agent_respond(
                member, speaker_id, message
            )

            if should_respond:
                # Generate response using dialogue system
                response = await self.dialogue_system.generate_response(
                    agent=member,
                    speaker_name=party.get_member_name(speaker_id),
                    message=message,
                    context=party.recent_dialogue
                )
                responses.append(response)

        return responses
```

#### 3.3 Frontend Game Setup Flow

```
app/
├── worlds/
│   ├── page.tsx              # World list/selection
│   └── create/
│       └── page.tsx          # World creation
├── game/
│   ├── setup/
│   │   └── page.tsx          # Player & party configuration
│   ├── [sessionId]/
│   │   └── page.tsx          # Main game view (existing GameView enhanced)
│   └── demo/
│       └── page.tsx          # Demo mode (existing)
```

**Game Setup Page Components:**
1. **Player Creator**: Name, appearance, starting stats
2. **Party Configurator**: Size (1-6), roles, names (or random)
3. **Spawn Location**: Choose settlement to start in
4. **Game Settings**: Tick speed, difficulty, debug mode toggle

---

### Phase 4: Hierarchical Agent Execution (Week 4)

This is critical for performance - not all agents need to think every tick.

#### 4.1 Hierarchical Scheduler Design

```python
# agents/simulation/hierarchical_scheduler.py

from enum import Enum
from dataclasses import dataclass

class AgentTier(Enum):
    """Agent importance tiers for scheduling."""
    PLAYER_PARTY = 1      # Every tick
    NEARBY = 2            # Every 2-3 ticks
    SAME_SETTLEMENT = 3   # Every 5-10 ticks
    SAME_REGION = 4       # Every 20-50 ticks
    WORLD = 5             # Every 100+ ticks (or event-driven)

class HierarchicalScheduler:
    """
    Schedules agent execution based on proximity to player
    and organizational hierarchy.
    """

    def __init__(self):
        self.tick_intervals = {
            AgentTier.PLAYER_PARTY: 1,
            AgentTier.NEARBY: 3,
            AgentTier.SAME_SETTLEMENT: 10,
            AgentTier.SAME_REGION: 50,
            AgentTier.WORLD: 200
        }

        # Collective agents (settlements, guilds)
        self.collective_agents: dict[str, CollectiveAgent] = {}

    def classify_agent(
        self,
        agent: AgentState,
        player: PlayerCharacter,
        party_ids: set[str]
    ) -> AgentTier:
        """Determine agent's execution tier."""
        if agent.id in party_ids:
            return AgentTier.PLAYER_PARTY

        distance = self.calculate_distance(agent, player)

        if distance <= 5:  # Within 5 tiles
            return AgentTier.NEARBY
        elif agent.settlement_id == player.settlement_id:
            return AgentTier.SAME_SETTLEMENT
        elif self.same_region(agent, player):
            return AgentTier.SAME_REGION
        else:
            return AgentTier.WORLD

    def get_agents_to_execute(self, tick: int) -> list[AgentState]:
        """Get agents that should execute this tick."""
        agents_to_run = []

        for agent_id, agent in self.all_agents.items():
            tier = self.agent_tiers[agent_id]
            interval = self.tick_intervals[tier]

            if tick % interval == 0:
                agents_to_run.append(agent)

        return agents_to_run

    async def execute_collective(
        self,
        collective: CollectiveAgent,
        tick: int
    ) -> list[AgentAction]:
        """
        Execute a collective agent (settlement/kingdom).
        Returns high-level actions that affect member agents.
        """
        # Collective agents make decisions for their members
        # e.g., "Village decides to increase patrols"
        # This generates events/modifiers for individual agents

        decision = await collective.make_collective_decision(
            context=self.get_collective_context(collective)
        )

        return decision.member_effects
```

#### 4.2 Collective Agents (Settlements/Kingdoms)

```python
# agents/collective/settlement_agent.py

class SettlementAgent:
    """
    A settlement acts as a collective agent that:
    - Makes decisions on behalf of residents
    - Generates events and opportunities
    - Manages settlement-wide state
    """

    def __init__(self, settlement: Settlement):
        self.settlement = settlement
        self.residents: list[str] = []  # Agent IDs
        self.mood = 0.5  # Aggregate happiness
        self.resources = {}
        self.events_queue = []

    async def collective_tick(self, game_time: SimulationTime):
        """
        Settlement-level thinking (runs less frequently).
        Affects all residents without individual LLM calls.
        """
        # 1. Update aggregate mood from resident states
        self.mood = self.calculate_aggregate_mood()

        # 2. Generate settlement events
        if random.random() < 0.1:  # 10% chance per collective tick
            event = await self.generate_settlement_event()
            self.events_queue.append(event)

        # 3. Update resource availability
        self.update_resources()

        # 4. Return modifiers for resident agents
        return SettlementModifiers(
            mood_modifier=self.mood - 0.5,
            safety_modifier=self.calculate_safety(),
            opportunity_events=self.events_queue
        )

    def get_resident_context(self, agent_id: str) -> dict:
        """
        Get settlement context for a resident's decision-making.
        This is injected into the agent's perception.
        """
        return {
            'settlement_name': self.settlement.name,
            'settlement_mood': self.mood,
            'recent_events': self.recent_events,
            'available_resources': self.resources,
            'population': len(self.residents)
        }


class KingdomAgent:
    """
    Kingdom-level collective agent.
    Makes political/military decisions affecting settlements.
    """

    def __init__(self, faction: Faction):
        self.faction = faction
        self.settlements: list[SettlementAgent] = []
        self.policies = {}
        self.relations = {}  # With other kingdoms

    async def kingdom_tick(self, game_time: SimulationTime):
        """
        Kingdom-level decisions (runs very infrequently).
        """
        # Only think about major decisions
        if self.should_make_decision(game_time):
            decision = await self.llm_kingdom_decision()
            await self.apply_kingdom_decision(decision)
```

#### 4.3 Agent Tier Visualization

The frontend should show which agents are "active" vs "dormant":

```typescript
// In AgentInfoPanel or new component
interface AgentActivityIndicator {
  tier: 'party' | 'nearby' | 'settlement' | 'region' | 'world';
  lastThought: Date;
  nextThought: Date;
  isActive: boolean;
}
```

---

### Phase 5: Debug Tools & Memory Inspector (Week 5)

#### 5.1 Debug Panel Component

Create: `src/components/debug/DebugPanel.tsx`

```typescript
interface DebugPanelProps {
  isOpen: boolean;
  selectedAgentId: string | null;
}

// Sections:
// 1. Memory Stream - Recent memories with importance scores
// 2. Current Thoughts - What the agent is currently thinking
// 3. Plans - Day plan, hour plan, current action
// 4. Relationships - Graph of relationships with trust/affection
// 5. Needs - Visual bars for hunger, thirst, rest, etc.
// 6. Simulation Stats - Tick rate, active agents, LLM calls

interface DebugState {
  // Memory tab
  memories: Memory[];
  memoryFilter: 'all' | 'observation' | 'reflection' | 'plan';

  // Thinking tab
  currentThought: string;
  reasoningChain: string[];
  lastLLMCall: {
    prompt: string;
    response: string;
    duration: number;
  };

  // Plans tab
  dayPlan: DayPlan;
  hourPlan: HourPlan;
  currentAction: ActionPlan;

  // Relationships tab
  relationships: Relationship[];

  // Simulation tab
  simulationStats: {
    tickCount: number;
    activeAgents: number;
    totalAgents: number;
    avgTickDuration: number;
    llmCallsPerMinute: number;
  };
}
```

#### 5.2 Debug API Endpoints

```python
# api/routers/debug.py

@router.get("/agents/{agent_id}/memories")
async def get_agent_memories(
    agent_id: str,
    limit: int = 50,
    memory_type: str = None,
    min_importance: float = 0
):
    """Get agent's memory stream with filters."""
    memories = await memory_service.get_memories(
        agent_id=agent_id,
        limit=limit,
        memory_type=memory_type,
        min_importance=min_importance
    )
    return {
        'memories': [m.to_dict() for m in memories],
        'total_count': len(memories)
    }

@router.get("/agents/{agent_id}/thoughts")
async def get_agent_thoughts(agent_id: str):
    """Get agent's current cognitive state."""
    agent = await agent_service.get_agent(agent_id)
    graph_state = agent_service.get_graph_state(agent_id)

    return {
        'current_phase': graph_state.phase.value,
        'current_thought': graph_state.current_thought,
        'reasoning_chain': graph_state.reasoning_history,
        'last_perception': graph_state.last_perception,
        'retrieved_memories': [m.to_dict() for m in graph_state.retrieved_memories],
        'pending_action': graph_state.pending_action
    }

@router.get("/agents/{agent_id}/plans")
async def get_agent_plans(agent_id: str):
    """Get agent's planning hierarchy."""
    planner = await agent_service.get_planner(agent_id)

    return {
        'day_plan': planner.current_day_plan,
        'hour_plan': planner.current_hour_plan,
        'action_plan': planner.current_action,
        'plan_progress': planner.get_progress()
    }

@router.get("/agents/{agent_id}/llm-history")
async def get_llm_history(agent_id: str, limit: int = 10):
    """Get recent LLM calls for this agent (for debugging)."""
    history = await llm_service.get_call_history(agent_id, limit)
    return {
        'calls': [
            {
                'timestamp': call.timestamp,
                'prompt_preview': call.prompt[:200],
                'response_preview': call.response[:200],
                'duration_ms': call.duration_ms,
                'tokens_used': call.tokens_used
            }
            for call in history
        ]
    }

@router.get("/simulation/stats")
async def get_simulation_stats(session_id: str):
    """Get overall simulation statistics."""
    session = game_service.sessions[session_id]

    return {
        'tick_count': session.orchestrator.tick_count,
        'simulation_time': session.orchestrator.time_manager.current_time,
        'agents': {
            'total': len(session.orchestrator.agents),
            'active_this_tick': session.orchestrator.last_tick_active,
            'by_tier': session.orchestrator.agents_by_tier
        },
        'performance': {
            'avg_tick_ms': session.orchestrator.avg_tick_duration,
            'llm_calls_per_tick': session.orchestrator.llm_calls_per_tick,
            'memory_retrievals_per_tick': session.orchestrator.memory_retrievals_per_tick
        }
    }
```

#### 5.3 Memory Visualization Component

```typescript
// src/components/debug/MemoryStream.tsx

interface MemoryVisualization {
  id: string;
  type: 'observation' | 'reflection' | 'plan';
  description: string;
  importance: number;  // 0-1
  recency: number;     // 0-1 (decayed)
  relevance?: number;  // 0-1 (to current context)
  timestamp: Date;
  linkedMemories?: string[];  // For reflections
}

// Visual representation:
// - Timeline view (vertical)
// - Importance shown as bar width or color intensity
// - Type shown as icon (eye for observation, lightbulb for reflection, clipboard for plan)
// - Click to expand full details
// - Linked memories shown with connecting lines
```

#### 5.4 Thinking Visualization

```typescript
// src/components/debug/ThinkingPanel.tsx

// Shows real-time agent cognitive process:
//
// ┌─────────────────────────────────────────┐
// │ 🧠 Elena the Blacksmith is thinking...  │
// ├─────────────────────────────────────────┤
// │ Phase: RETRIEVING                       │
// │                                         │
// │ Current thought:                        │
// │ "I need to decide whether to help       │
// │ Thomas with his broken cart or          │
// │ continue working on the sword order."   │
// │                                         │
// │ Retrieved memories:                     │
// │ • Thomas helped me last winter [0.8]    │
// │ • The sword is for the captain [0.7]    │
// │ • Cart repairs are quick [0.5]          │
// │                                         │
// │ Reasoning:                              │
// │ → Thomas is a friend (trust: 0.7)       │
// │ → Sword order is important but not      │
// │   urgent                                │
// │ → Helping Thomas would take ~30 min     │
// │                                         │
// │ Decision: Help Thomas first             │
// └─────────────────────────────────────────┘
```

---

### Phase 6: Player-Agent Interaction System (Week 6)

#### 6.1 Dialogue System Integration

```python
# api/services/dialogue_service.py

class DialogueService:
    """Handles conversations between player and agents."""

    async def start_conversation(
        self,
        player: PlayerCharacter,
        agent: AgentState,
        opening: str = None
    ) -> Conversation:
        """Initialize a conversation."""
        conversation = Conversation(
            participants=[player.id, agent.id],
            location=(player.x, player.y),
            started_at=datetime.now()
        )

        if opening:
            # Player initiated with message
            conversation.add_turn(player.id, player.name, opening, 'player')

            # Generate agent response
            response = await self.generate_agent_response(
                agent=agent,
                conversation=conversation,
                player=player
            )
            conversation.add_turn(agent.id, agent.name, response.text, 'agent')
        else:
            # Agent initiates (player approached)
            greeting = await self.generate_greeting(agent, player)
            conversation.add_turn(agent.id, agent.name, greeting, 'agent')

        return conversation

    async def generate_agent_response(
        self,
        agent: AgentState,
        conversation: Conversation,
        player: PlayerCharacter
    ) -> DialogueResponse:
        """Generate contextual response from agent."""
        # 1. Retrieve relevant memories about player
        memories = await self.memory_service.retrieve(
            agent_id=agent.id,
            query=f"interactions with {player.name}",
            limit=5
        )

        # 2. Get relationship state
        relationship = agent.relationships.get(player.id)

        # 3. Build context
        context = {
            'agent_summary': agent.get_summary(),
            'relationship': relationship.to_dict() if relationship else None,
            'memories': [m.description for m in memories],
            'conversation_history': conversation.get_recent_turns(5),
            'current_time': self.time_manager.get_time_description(),
            'location': self.get_location_description(agent.x, agent.y)
        }

        # 4. Generate response via LLM
        response = await self.ollama.generate(
            prompt=self.build_dialogue_prompt(context),
            system=DIALOGUE_SYSTEM_PROMPT
        )

        # 5. Update agent memory
        await self.memory_service.add_observation(
            agent_id=agent.id,
            description=f"Talked to {player.name}: they said '{conversation.last_player_message}'",
            importance=0.4
        )

        # 6. Update relationship
        await self.update_relationship_from_conversation(
            agent, player, conversation
        )

        return DialogueResponse(
            text=response,
            emotion=self.detect_emotion(response),
            should_end=self.should_end_conversation(response)
        )
```

#### 6.2 Frontend Dialogue Modal

```typescript
// src/components/dialogue/DialogueModal.tsx

interface DialogueModalProps {
  isOpen: boolean;
  agent: GameAgent;
  onClose: () => void;
  onSendMessage: (message: string) => void;
}

// Features:
// - Agent portrait/avatar with emotion indicator
// - Chat-like message history
// - Text input for player
// - Quick response buttons (contextual)
// - Agent mood/disposition indicator
// - "End Conversation" button
// - Optional: Debug toggle to show agent's internal reasoning
```

#### 6.3 World Interaction System

```python
# api/services/interaction_service.py

class InteractionService:
    """Handles player interactions with world objects and locations."""

    async def interact(
        self,
        player: PlayerCharacter,
        target_type: str,
        target_id: str,
        action: str
    ) -> InteractionResult:
        """Process a player interaction."""

        if target_type == 'agent':
            return await self.interact_with_agent(player, target_id, action)
        elif target_type == 'object':
            return await self.interact_with_object(player, target_id, action)
        elif target_type == 'tile':
            return await self.interact_with_tile(player, target_id, action)
        elif target_type == 'settlement':
            return await self.interact_with_settlement(player, target_id, action)

    async def interact_with_agent(
        self,
        player: PlayerCharacter,
        agent_id: str,
        action: str
    ) -> InteractionResult:
        """Player interacts with an agent."""
        agent = await self.agent_service.get_agent(agent_id)

        if action == 'talk':
            conversation = await self.dialogue_service.start_conversation(
                player, agent
            )
            return InteractionResult(
                type='conversation_started',
                data={'conversation': conversation}
            )

        elif action == 'trade':
            # Open trade interface
            trade = await self.trade_service.initiate(player, agent)
            return InteractionResult(type='trade_started', data={'trade': trade})

        elif action == 'follow':
            # Request agent join party (if possible)
            result = await self.party_service.request_join(player, agent)
            return result

        elif action == 'give':
            # Give item to agent
            pass

        elif action == 'attack':
            # Initiate combat (enters combat mode)
            pass
```

---

### Phase 7: Real-Time Updates & WebSocket (Week 7)

#### 7.1 WebSocket Manager

```python
# api/websocket/manager.py

from fastapi import WebSocket
from typing import Dict, Set
import asyncio
import json

class GameWebSocketManager:
    """Manages WebSocket connections for real-time game updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # session_id -> connections
        self.player_connections: Dict[str, WebSocket] = {}  # player_id -> connection

    async def connect(self, websocket: WebSocket, session_id: str, player_id: str):
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()

        self.active_connections[session_id].add(websocket)
        self.player_connections[player_id] = websocket

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Send update to all players in session."""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    # Connection closed, remove it
                    self.active_connections[session_id].discard(connection)

    async def send_to_player(self, player_id: str, message: dict):
        """Send update to specific player."""
        if player_id in self.player_connections:
            await self.player_connections[player_id].send_json(message)


# Message types
class WSMessageType:
    TICK_UPDATE = 'tick_update'
    AGENT_UPDATE = 'agent_update'
    PLAYER_UPDATE = 'player_update'
    EVENT = 'event'
    DIALOGUE = 'dialogue'
    NOTIFICATION = 'notification'
    DEBUG = 'debug'
```

#### 7.2 Frontend WebSocket Hook

```typescript
// src/hooks/useGameWebSocket.ts

export function useGameWebSocket(sessionId: string) {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const updateAgent = useGameStore(state => state.updateAgent);
  const addMessage = useGameStore(state => state.addMessage);
  const addEvent = useGameStore(state => state.addEvent);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/game/${sessionId}`);

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'tick_update':
          // Update game time, batch agent updates
          handleTickUpdate(data.payload);
          break;

        case 'agent_update':
          // Single agent changed (moved, action, etc.)
          updateAgent(data.payload.agent);
          break;

        case 'dialogue':
          // Agent said something
          addMessage(data.payload.message);
          break;

        case 'event':
          // World event occurred
          addEvent(data.payload.event);
          break;

        case 'debug':
          // Debug info (if debug mode enabled)
          handleDebugUpdate(data.payload);
          break;
      }
    };

    wsRef.current = ws;

    return () => ws.close();
  }, [sessionId]);

  const sendPlayerAction = useCallback((action: PlayerAction) => {
    wsRef.current?.send(JSON.stringify({
      type: 'player_action',
      payload: action
    }));
  }, []);

  return { isConnected, sendPlayerAction };
}
```

---

### Phase 8: Full Integration & Polish (Week 8)

#### 8.1 Complete User Flow

```
1. LANDING PAGE (/)
   └── "Create World" / "Load World" / "Demo"

2. WORLD CREATION (/worlds/create)
   ├── Configure world parameters
   ├── Watch generation progress
   └── "Continue to Game Setup"

3. GAME SETUP (/game/setup?worldId=xxx)
   ├── Create player character
   ├── Configure party (size, roles)
   ├── Choose starting location
   └── "Start Adventure"

4. MAIN GAME (/game/[sessionId])
   ├── Game Canvas (PixiJS)
   │   ├── World tiles
   │   ├── Player & party
   │   ├── Nearby agents
   │   └── Settlements/features
   ├── Chat Panel
   │   ├── Party chat
   │   ├── Global events
   │   └── DM narration
   ├── Agent Info Panel (on selection)
   ├── Tile Info Panel (on hover)
   ├── Character Sheet (modal)
   ├── Debug Panel (toggle)
   └── Dialogue Modal (when talking)

5. DEBUG MODE (toggle in game)
   ├── Memory Stream view
   ├── Thinking visualization
   ├── Plan hierarchy
   ├── Relationship graph
   └── Simulation stats
```

#### 8.2 Key Integration Points Checklist

```
[x] World Builder → Database
    [x] Store generated chunks
    [x] Store settlements
    [x] Store factions
    [x] Store roads/features

[x] Database → Frontend
    [x] Load chunks for viewport
    [x] Load nearby agents
    [~] Real-time updates via Supabase (WebSocket ready, Supabase Realtime pending)

[x] Agent Framework → API
    [x] Execute agent ticks
    [x] Store/retrieve memories
    [x] Generate dialogue
    [x] Update relationships

[x] API → Frontend
    [x] WebSocket for real-time
    [x] REST for queries (86 routes)
    [x] Debug endpoints

[x] Player → Agents
    [x] Dialogue system
    [x] Party commands
    [x] Interactions

[x] Hierarchical Execution
    [x] Tier classification (7 tiers)
    [x] Collective agents (SettlementAgent, KingdomAgent)
    [x] Performance optimization (HierarchicalScheduler)
```

**Implementation Status (as of 2025-12-02):**
- Phase 1-3: API Layer, World Generation, Game Sessions ✅
- Phase 4: Hierarchical Agent Execution ✅
- Phase 5: Debug Tools & Memory Inspector ✅
- Phase 6: Player-Agent Interaction System ✅
- Phase 7: Real-Time Updates & WebSocket ✅
- Phase 8: Full Integration & Polish ✅

**Integration Test Results: 27/27 passing**

---

## Part 3: Database Migrations

### 3.1 New Tables Required

```sql
-- Game Sessions
CREATE TABLE game_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id) ON DELETE CASCADE,
    player_id UUID NOT NULL,
    party_id UUID,
    state VARCHAR(20) DEFAULT 'active',  -- active, paused, ended
    current_tick BIGINT DEFAULT 0,
    game_time JSONB,  -- SimulationTime as JSON
    settings JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW()
);

-- Player Characters
CREATE TABLE player_characters (
    player_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,  -- For auth later
    session_id UUID REFERENCES game_sessions(session_id),
    name VARCHAR(255) NOT NULL,
    x INT NOT NULL,
    y INT NOT NULL,
    stats JSONB NOT NULL,
    inventory JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Parties
CREATE TABLE parties (
    party_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES game_sessions(session_id),
    leader_id UUID NOT NULL,
    formation VARCHAR(50) DEFAULT 'follow',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Party Members (junction)
CREATE TABLE party_members (
    party_id UUID REFERENCES parties(party_id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(agent_id) ON DELETE CASCADE,
    role VARCHAR(50),
    join_order INT,
    PRIMARY KEY (party_id, agent_id)
);

-- Conversations Log
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES game_sessions(session_id),
    participants UUID[] NOT NULL,
    location_x INT,
    location_y INT,
    turns JSONB DEFAULT '[]',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Debug Logs (for LLM call history)
CREATE TABLE debug_llm_calls (
    call_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    agent_id UUID,
    prompt TEXT,
    response TEXT,
    duration_ms INT,
    tokens_used INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for debug queries
CREATE INDEX idx_debug_llm_session ON debug_llm_calls(session_id, created_at DESC);
CREATE INDEX idx_debug_llm_agent ON debug_llm_calls(agent_id, created_at DESC);
```

---

## Part 4: Configuration & Environment

### 4.1 Environment Variables

```bash
# .env

# Supabase
SUPABASE_URL=http://localhost:54321
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_QUEUE=100

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Debug
DEBUG_MODE=true
LOG_LLM_CALLS=true
```

### 4.2 Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=http://supabase:54321
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - supabase
      - ollama
    volumes:
      - ./api:/app
      - ./agents:/app/agents
      - ./builder:/app/builder

  frontend:
    build: ./frontend/universa
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    volumes:
      - ./frontend/universa:/app

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  supabase:
    # Use Supabase CLI or local setup
    # ...

volumes:
  ollama_data:
```

---

## Part 5: Testing Strategy

### 5.1 Test Categories

```
tests/
├── unit/
│   ├── test_world_service.py
│   ├── test_agent_service.py
│   ├── test_party_service.py
│   └── test_dialogue_service.py
├── integration/
│   ├── test_world_generation_flow.py
│   ├── test_game_session_flow.py
│   ├── test_agent_execution.py
│   └── test_websocket_updates.py
├── e2e/
│   ├── test_full_game_flow.py
│   └── test_dialogue_interactions.py
└── performance/
    ├── test_tick_performance.py
    ├── test_many_agents.py
    └── test_llm_throughput.py
```

### 5.2 Key Test Scenarios

1. **World Creation Flow**
   - Create world with parameters
   - Verify all 18 passes complete
   - Verify data stored in database
   - Verify chunks loadable by frontend

2. **Game Session Flow**
   - Create session with player and party
   - Verify agents initialized correctly
   - Run 100 ticks
   - Verify agent memories accumulated
   - Verify relationships formed

3. **Dialogue Flow**
   - Start conversation with agent
   - Send multiple messages
   - Verify agent responses coherent
   - Verify memories created for both parties
   - Verify relationship updated

4. **Hierarchical Execution**
   - Create 100 agents across tiers
   - Run 1000 ticks
   - Verify tier-appropriate execution frequency
   - Verify collective agents functioning

---

## Part 6: Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tick duration (10 party agents) | < 500ms | P95 latency |
| Tick duration (100 nearby agents) | < 2s | P95 latency |
| LLM calls per tick (party) | 1-3 | Average |
| Memory retrieval | < 50ms | P95 latency |
| WebSocket latency | < 100ms | P95 latency |
| Frontend FPS | > 30 | Sustained |
| Chunk load time | < 200ms | P95 latency |

---

## Part 7: Future Enhancements

After core integration is complete:

1. **Multiplayer** - Multiple human players in same world
2. **Persistent Worlds** - Continue simulation when player offline
3. **Quest System** - AI DM generates quests based on world state
4. **Combat System** - Turn-based combat with agent tactics
5. **Economy** - Agent-driven markets and trade
6. **Magic System** - Integrate ley lines with agent abilities
7. **Generations** - Agent reproduction and inheritance
8. **Mod Support** - Custom agent types, skills, events

---

*Document Version: 1.0*
*Created: 2025-12-02*
*Last Updated: 2025-12-02*
