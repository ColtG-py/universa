# Universa Agentic AI Implementation Plan

## Executive Summary

This document outlines the comprehensive implementation plan for integrating an agentic AI system into the Universa world generation project. The system is inspired by the Stanford "Generative Agents" paper (Park et al., 2023), which demonstrated believable simulacra of human behavior through a memory stream architecture with observation, reflection, and planning components.

The goal is to create autonomous agents that can:
- Inhabit and interact with the procedurally generated world from `/builder`
- Develop personalities, skills, and relationships over time
- Form memories, make plans, and reflect on experiences
- Coordinate with other agents and respond to environmental changes

---

## Part 1: World Builder Integration Analysis

### 1.1 Current World Generation Pipeline

The `/builder` folder contains an 18-pass procedural generation system that produces:

| Pass | Output | Agent Relevance |
|------|--------|-----------------|
| 1-3 | Planetary, Tectonics, Topography | Navigation terrain |
| 4 | Geology | Resource extraction |
| 5-7 | Atmosphere, Oceans, Climate | Survival needs, warmth |
| 8-9 | Erosion, Groundwater | Water availability |
| 10 | Rivers | Water sources, travel routes |
| 11-12 | Soil, Biomes | Agriculture, food sources |
| 13 | Fauna | Hunting, animal husbandry |
| 14 | Resources | Mining, gathering, crafting materials |
| 15 | Magic | Magic system interactions |
| 16 | Settlements | Agent spawn points, social hubs |
| 17 | Roads | Travel networks |
| 18 | Politics | Faction affiliations, governance |

### 1.2 Key Data Structures for Agent Integration

From `builder/models/world.py`:

```python
# WorldState contains all generated data
world_state.chunks           # Dict[str, WorldChunk] - 256x256 tiles
world_state.settlements      # List[Settlement] - population centers
world_state.road_network     # Road connections between settlements
world_state.factions         # Political entities
world_state.ley_line_network # Magic system

# WorldChunk contains per-tile data agents will query
chunk.elevation              # Navigation difficulty
chunk.temperature_c          # Warmth need
chunk.biome_type            # Available resources
chunk.settlement_presence    # Social locations
chunk.mana_concentration     # Magic availability
chunk.faction_territory      # Political affiliation
```

### 1.3 World-Agent Interface Requirements

Create a world query API that agents can use:

```python
class WorldInterface:
    """Bridge between agent reasoning and world data"""

    async def query_location(self, x: int, y: int) -> LocationData:
        """Get full environmental data at coordinates"""

    async def query_radius(self, x: int, y: int, radius: int) -> List[LocationData]:
        """Get area around agent for observation"""

    async def query_path(self, start: Tuple, end: Tuple) -> PathResult:
        """Calculate travel route using road network"""

    async def query_nearby_agents(self, x: int, y: int, radius: int) -> List[AgentSummary]:
        """Find other agents in perception range"""

    async def query_resources(self, x: int, y: int, resource_type: str) -> ResourceAvailability:
        """Check resource availability for gathering/crafting"""
```

---

## Part 2: Generative Agent Architecture

Based on the Stanford paper, the core architecture has three components:

### 2.1 Memory Stream

The memory stream is the foundation - a comprehensive record of all agent experiences stored in natural language.

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY STREAM                           │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Observations│  │ Reflections │  │    Plans    │         │
│  │             │  │             │  │             │         │
│  │ - Saw X     │  │ - I think X │  │ - Will do X │         │
│  │ - Heard Y   │  │ - X implies │  │ - At time T │         │
│  │ - Did Z     │  │ - Important │  │ - Because R │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          │                                  │
│                          ▼                                  │
│                   ┌─────────────┐                          │
│                   │  Retrieval  │                          │
│                   │  Function   │                          │
│                   └─────────────┘                          │
│                          │                                  │
│            ┌─────────────┼─────────────┐                   │
│            ▼             ▼             ▼                   │
│      ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│      │ Recency │   │Importance│   │Relevance│              │
│      │  Score  │   │  Score  │   │  Score  │              │
│      └─────────┘   └─────────┘   └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Retrieval Formula:**
```
score = α_recency × recency + α_importance × importance + α_relevance × relevance
```

Where:
- **Recency**: Exponential decay (0.995^hours_since_access)
- **Importance**: LLM-rated 1-10 scale of poignancy
- **Relevance**: Cosine similarity of embedding vectors

### 2.2 Reflection System

Reflections are higher-level inferences generated periodically when importance threshold is exceeded (e.g., sum > 150).

```
[Observation] Klaus is reading about gentrification
[Observation] Klaus is taking notes on urban development
[Observation] Klaus discussed housing with Maria
                    │
                    ▼
         ┌──────────────────┐
         │   REFLECTION     │
         │                  │
         │ "Klaus is deeply │
         │ interested in    │
         │ gentrification   │
         │ research"        │
         └──────────────────┘
```

### 2.3 Planning System

Plans are generated top-down and decomposed recursively:

```
Day Plan (broad strokes)
    │
    ├── 7:00 AM - Wake up, morning routine
    │       │
    │       ├── 7:00 - Get out of bed
    │       ├── 7:05 - Brush teeth
    │       ├── 7:15 - Get dressed
    │       └── 7:30 - Eat breakfast
    │
    ├── 9:00 AM - Go to work at forge
    │       │
    │       ├── 9:00 - Walk to forge
    │       ├── 9:30 - Light furnace
    │       ├── 10:00 - Begin smithing
    │       └── ...
    │
    └── 6:00 PM - Return home, evening activities
```

---

## Part 3: Database Schema (Supabase)

### 3.1 Core Tables

```sql
-- Worlds table (links to builder output)
CREATE TABLE worlds (
    world_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seed BIGINT NOT NULL,
    size VARCHAR(20) NOT NULL, -- SMALL, MEDIUM, LARGE, HUGE
    generation_params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_simulated TIMESTAMPTZ
);

-- Agents table
CREATE TABLE agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id) ON DELETE CASCADE,
    agent_type VARCHAR(50) NOT NULL, -- human, animal, settlement, guild, etc.
    name VARCHAR(255) NOT NULL,

    -- Location
    position_x INT NOT NULL,
    position_y INT NOT NULL,
    chunk_id VARCHAR(50),

    -- Core Stats (1-20 scale)
    strength INT DEFAULT 10 CHECK (strength BETWEEN 1 AND 20),
    dexterity INT DEFAULT 10 CHECK (dexterity BETWEEN 1 AND 20),
    constitution INT DEFAULT 10 CHECK (constitution BETWEEN 1 AND 20),
    intelligence INT DEFAULT 10 CHECK (intelligence BETWEEN 1 AND 20),
    wisdom INT DEFAULT 10 CHECK (wisdom BETWEEN 1 AND 20),
    charisma INT DEFAULT 10 CHECK (charisma BETWEEN 1 AND 20),

    -- Needs (0.0 = satisfied, 1.0 = critical)
    hunger FLOAT DEFAULT 0.0 CHECK (hunger BETWEEN 0 AND 1),
    thirst FLOAT DEFAULT 0.0 CHECK (thirst BETWEEN 0 AND 1),
    rest FLOAT DEFAULT 0.0 CHECK (rest BETWEEN 0 AND 1),
    warmth FLOAT DEFAULT 0.0 CHECK (warmth BETWEEN 0 AND 1),
    safety FLOAT DEFAULT 0.0 CHECK (safety BETWEEN 0 AND 1),
    social FLOAT DEFAULT 0.0 CHECK (social BETWEEN 0 AND 1),

    -- Alignment (-1.0 to 1.0)
    good_evil_score FLOAT DEFAULT 0.0 CHECK (good_evil_score BETWEEN -1 AND 1),
    lawful_chaotic_score FLOAT DEFAULT 0.0 CHECK (lawful_chaotic_score BETWEEN -1 AND 1),

    -- Health
    health FLOAT DEFAULT 1.0 CHECK (health BETWEEN 0 AND 1),
    stamina FLOAT DEFAULT 1.0 CHECK (stamina BETWEEN 0 AND 1),
    is_alive BOOLEAN DEFAULT TRUE,
    age_days INT DEFAULT 0,

    -- Genetics (stored as JSONB for flexibility)
    genome JSONB,
    physical_attributes JSONB,

    -- Faction
    faction_id UUID,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    generation INT DEFAULT 0,

    CONSTRAINT fk_world FOREIGN KEY (world_id) REFERENCES worlds(world_id)
);

-- Memory Stream (core of generative agent architecture)
CREATE TABLE memory_stream (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,

    -- Memory content
    memory_type VARCHAR(20) NOT NULL, -- 'observation', 'reflection', 'plan'
    description TEXT NOT NULL,

    -- Retrieval scores
    importance FLOAT DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    game_time TIMESTAMPTZ, -- In-simulation time

    -- Location context
    location_x INT,
    location_y INT,

    -- For reflections: which memories they're based on
    source_memories UUID[], -- Array of memory_ids this was derived from

    -- Embedding for semantic search (requires pgvector extension)
    embedding VECTOR(1536) -- OpenAI embedding dimension, adjust for local models
);

-- Index for fast retrieval
CREATE INDEX idx_memory_agent ON memory_stream(agent_id);
CREATE INDEX idx_memory_type ON memory_stream(memory_type);
CREATE INDEX idx_memory_time ON memory_stream(created_at DESC);

-- Skills table
CREATE TABLE skills (
    skill_id VARCHAR(100) PRIMARY KEY, -- e.g., 'combat.melee.swords'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    parent_skill VARCHAR(100) REFERENCES skills(skill_id),

    -- Governing stats
    primary_stat VARCHAR(20) NOT NULL,
    secondary_stat VARCHAR(20) NOT NULL,

    -- Difficulty and progression
    base_difficulty FLOAT DEFAULT 0.5,
    xp_per_use FLOAT DEFAULT 1.0,
    xp_to_level FLOAT DEFAULT 100.0,
    max_level INT DEFAULT 100,

    -- Requirements
    required_skills JSONB DEFAULT '{}',
    required_tools TEXT[],

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID -- Agent who invented this skill (for dynamic creation)
);

-- Agent skills junction table
CREATE TABLE agent_skills (
    agent_id UUID REFERENCES agents(agent_id) ON DELETE CASCADE,
    skill_id VARCHAR(100) REFERENCES skills(skill_id),

    level INT DEFAULT 0,
    experience FLOAT DEFAULT 0.0,
    times_used INT DEFAULT 0,
    success_count INT DEFAULT 0,
    fail_count INT DEFAULT 0,
    last_used TIMESTAMPTZ,

    PRIMARY KEY (agent_id, skill_id)
);

-- Relationships between agents
CREATE TABLE agent_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_a UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    agent_b UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,

    -- Relationship metrics
    familiarity FLOAT DEFAULT 0.0 CHECK (familiarity BETWEEN 0 AND 1),
    trust FLOAT DEFAULT 0.5 CHECK (trust BETWEEN -1 AND 1),
    affection FLOAT DEFAULT 0.0 CHECK (affection BETWEEN -1 AND 1),

    -- Relationship type
    relationship_type VARCHAR(50), -- friend, rival, family, colleague, etc.

    -- Memory of interactions
    interaction_count INT DEFAULT 0,
    last_interaction TIMESTAMPTZ,

    UNIQUE(agent_a, agent_b)
);

-- Episodic memory (specific events with full context)
CREATE TABLE episodic_memories (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,

    summary TEXT NOT NULL,
    context JSONB, -- Full context data
    actions JSONB, -- Actions taken
    outcomes JSONB, -- What happened
    skills_used TEXT[],
    reflection TEXT,

    importance FLOAT DEFAULT 0.5,
    success BOOLEAN,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    game_time TIMESTAMPTZ
);

-- Semantic facts (knowledge without temporal context)
CREATE TABLE semantic_facts (
    fact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,

    fact_text TEXT NOT NULL,
    category VARCHAR(50),
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(50), -- 'experience', 'told', 'observed'

    access_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    embedding VECTOR(1536)
);

-- Procedural knowledge (how-to)
CREATE TABLE procedural_memory (
    procedure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,

    procedure_name VARCHAR(255) NOT NULL,
    procedure_prompt TEXT NOT NULL, -- Natural language description of how to do it

    success_rate FLOAT DEFAULT 0.5,
    usage_count INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(agent_id, procedure_name)
);

-- Conversations log
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id),

    participants UUID[], -- Array of agent_ids
    location_x INT,
    location_y INT,

    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    -- Full dialogue stored as JSONB array
    dialogue JSONB -- [{speaker: uuid, utterance: text, timestamp: time}, ...]
);

-- Events chronicle (world history)
CREATE TABLE world_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id),

    event_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,

    -- Involved entities
    involved_agents UUID[],
    involved_factions UUID[],

    -- Location
    location_x INT,
    location_y INT,

    -- Impact
    importance FLOAT DEFAULT 0.5,

    game_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation state
CREATE TABLE simulation_state (
    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    world_id UUID REFERENCES worlds(world_id),

    current_game_time TIMESTAMPTZ,
    tick_count BIGINT DEFAULT 0,

    -- Performance metrics
    agents_active INT DEFAULT 0,
    last_tick_duration_ms INT,

    -- State
    is_running BOOLEAN DEFAULT FALSE,
    paused BOOLEAN DEFAULT FALSE,

    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.2 Enable Required Extensions

```sql
-- Enable pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 3.3 Key Indexes for Performance

```sql
-- Vector similarity search
CREATE INDEX idx_memory_embedding ON memory_stream
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_semantic_embedding ON semantic_facts
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Spatial queries for nearby agents
CREATE INDEX idx_agent_position ON agents(world_id, position_x, position_y);

-- Time-based retrieval
CREATE INDEX idx_memory_recency ON memory_stream(agent_id, last_accessed DESC);
```

---

## Part 4: Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Database Setup ✅ COMPLETE
- [x] Create Supabase project and configure connection
- [x] Run schema migrations (14 migrations applied)
- [x] Enable pgvector extension
- [x] Set up Row Level Security (RLS) policies
- [x] Create database helper functions (9 functions)

#### 1.2 World Interface Layer ✅ COMPLETE
- [x] Create `WorldInterface` class to bridge builder output with agents
- [x] Implement location query methods
- [x] Implement pathfinding using road network (A* pathfinder)
- [ ] Create serialization for world state to database

#### 1.3 Basic Agent Model ✅ COMPLETE
- [x] Implement `AgentState` Pydantic model
- [x] Implement `CoreStats` with stat modifiers
- [x] Implement `CoreNeeds` with time-based updates
- [x] Create agent CRUD operations in Supabase (AgentRepository, MemoryRepository)

### Phase 2: Memory System (Weeks 3-4) ✅ COMPLETE

#### 2.1 Memory Stream ✅ COMPLETE
- [x] Implement `MemoryStream` class
- [x] Create observation recording
- [x] Implement importance scoring via LLM (ImportanceScorer)
- [x] Create embedding generation for memories (EmbeddingGenerator)

#### 2.2 Retrieval Function ✅ COMPLETE
- [x] Implement recency scoring (exponential decay)
- [x] Implement importance normalization
- [x] Implement relevance via cosine similarity
- [x] Create combined retrieval with weighted scoring (MemoryRetrieval)
- [x] Add retrieval caching for performance (RetrievalCache)

#### 2.3 Three-Layer Memory ✅ COMPLETE
- [x] Implement `EpisodicMemory` with pgvector
- [x] Implement `SemanticMemory` for facts
- [x] Implement `ProceduralMemory` for skills
- [x] Create memory consolidation (MemoryConsolidator)

### Phase 3: Agent Reasoning (Weeks 5-6) ✅ COMPLETE

#### 3.1 Reflection System ✅
- [x] Create reflection trigger (importance threshold) - `agents/reasoning/reflection.py:ReflectionTrigger`
- [x] Implement question generation from recent memories - `ReflectionSystem._generate_questions()`
- [x] Implement insight extraction with citations - `ReflectionSystem._generate_insights()`
- [x] Store reflections back to memory stream - via `MemoryStream.add_reflection()`

#### 3.2 Planning System ✅
- [x] Create daily plan generation (broad strokes) - `agents/reasoning/planning.py:PlanningSystem.create_day_plan()`
- [x] Implement recursive decomposition (hour → 5-15 min) - `create_hour_plan()`, `create_action_plan()`
- [x] Create plan storage and retrieval - `DayPlan`, `HourPlan`, `ActionPlan` dataclasses
- [x] Implement plan re-evaluation on events - `PlanningSystem.replan_from_event()`

#### 3.3 Reaction System ✅
- [x] Create perception of environment changes - `agents/reasoning/reaction.py:ReactionSystem.perceive()`
- [x] Implement reaction decision (continue plan vs. react) - `ReactionSystem.should_react()`
- [x] Create dialogue generation for agent interactions - `agents/reasoning/dialogue.py:DialogueSystem`
- [x] Implement action execution pipeline - `ReactionSystem.execute_reaction()`

### Phase 4: LLM Integration (Weeks 7-8) ✅ COMPLETE

#### 4.1 Ollama Setup ✅
- [x] Configure Ollama with Qwen3-8B (0.933 F1 on BFCL tool calling) - `agents/llm/ollama_client.py`
- [x] Ollama handles concurrency natively (OLLAMA_NUM_PARALLEL=2 for 12GB VRAM)
- [x] Implement prompt templates - `agents/llm/prompts.py`
- [x] Add tool calling support - `OllamaClient.generate_with_tools()`, `chat_with_tools()`

#### 4.2 LangGraph Agent Architecture ✅
- [x] Create agent state graph - `agents/graph/state.py:AgentGraphState`
- [x] Implement perceive → retrieve → plan → act cycle - `agents/graph/agent_graph.py:AgentGraph`
- [x] Add memory integration nodes - `agents/graph/nodes.py:retrieve_node`
- [x] Create reflection trigger nodes - `agents/graph/nodes.py:reflect_node`

#### 4.3 Tool System (MCP) ✅
- [x] Define core tools - `agents/tools/core_tools.py` (observe, move, speak, use_skill, recall_memory, interact, rest)
- [x] Implement tool registry - `agents/tools/base.py:ToolRegistry`
- [x] Create memory recall tools - `recall_memory_tool`
- [x] Add world interaction tools - `interact_tool`, `rest_tool`
- [x] Tool executor with ReAct pattern - `agents/tools/executor.py:ToolExecutor`

### Phase 5: Skill System (Weeks 9-10) ✅ COMPLETE

#### 5.1 Skill Taxonomy ✅
- [x] Populate skills table with 100+ base skills - `agents/skills/taxonomy.py:get_base_skills()`
- [x] Implement hierarchical navigation - `SkillTree.get_children()`, `get_ancestors()`, `get_path_to_skill()`
- [x] Create stat-skill integration - `Skill.get_stat_modifier()`, `primary_stat`, `secondary_stat`
- [x] Add skill prerequisite validation - `SkillTree.check_requirements()`

#### 5.2 Skill Progression ✅
- [x] Implement XP gain on skill use - `agents/skills/progression.py:SkillProgressionSystem.use_skill()`
- [x] Create level-up mechanics - `add_xp()` with automatic level calculation
- [x] Add stat improvement from skill use - `get_stat_bonuses_from_skills()`
- [x] Implement parent skill XP sharing - `PARENT_XP_RATE = 0.5` propagation

#### 5.3 Skill Architect Agent ✅
- [x] Create skill creation request queue - `agents/skills/architect.py:SkillArchitect._pending_requests`
- [x] Implement skill validation logic - `_validate_with_llm()`, `_validate_heuristic()`
- [x] Add skill approval/rejection flow - `approve_request()`, `reject_request()`
- [x] Create skill genealogy tracking - `_skill_creators` dict, `get_skill_creator()`

### Phase 6: Social Systems (Weeks 11-12) ✅ COMPLETE

#### 6.1 Agent Interactions ✅
- [x] Implement agent perception - via `agents/graph/nodes.py:perceive_node`
- [x] Create conversation initiation logic - `agents/social/interactions.py:InteractionManager.start_conversation()`
- [x] Implement dialogue generation with memory context - `DialogueSystem` with memory retrieval
- [x] Add conversation recording to memory - `InteractionManager._record_observation()`

#### 6.2 Relationships ✅
- [x] Implement relationship tracking - `agents/social/relationships.py:RelationshipManager`
- [x] Create familiarity/trust/affection updates - `Relationship.update_from_interaction()`
- [x] Add relationship-based behavior modifiers - `Relationship.get_disposition()`
- [x] Implement relationship memory - `Relationship.shared_memories`, `RelationshipMemory`

#### 6.3 Information Diffusion ✅
- [x] Track information spread between agents - `agents/social/information.py:InformationNetwork`
- [x] Implement gossip/news mechanics - `share_information()`, `simulate_gossip_tick()`
- [x] Create faction-level information flow - `create_news_event()`, spread tracking

### Phase 7: Simulation Orchestration (Weeks 13-14) ✅ COMPLETE

#### 7.1 Time Management ✅
- [x] Create simulation tick system - `agents/simulation/time_manager.py:TimeManager.tick()`
- [x] Implement time acceleration - `TimeManager.set_speed()`, `time_multiplier`
- [x] Add day/night cycles affecting behavior - `TimeOfDay`, `get_environment_modifiers()`
- [x] Create seasonal effects - `Season`, `get_appropriate_activities()`

#### 7.2 Agent Scheduling ✅
- [x] Implement priority queue for agent actions - `agents/simulation/scheduler.py:AgentScheduler` with `heapq`
- [x] Create parallel agent execution - `execute_batch()`, `asyncio.gather()`
- [x] Add resource management (GPU/CPU) - `max_concurrent`, `_semaphore` concurrency control
- [x] Implement agent sleeping/waking - `max_per_agent`, `is_agent_busy()`, `wait_for_agent()`

#### 7.3 Event System ✅
- [x] Create world event detection - `agents/simulation/events.py:EventSystem`
- [x] Implement event broadcasting to agents - `subscribe()`, `_broadcast()`
- [x] Add historical event recording - `_history`, `get_recent_events()`
- [x] Create emergent event generation - `add_generation_rule()`, `check_generation_rules()`

#### 7.4 Simulation Orchestrator ✅
- [x] Main coordinator - `agents/simulation/orchestrator.py:SimulationOrchestrator`
- [x] Agent lifecycle management - `register_agent()`, `unregister_agent()`, `AgentContext`
- [x] Simulation control - `start()`, `stop()`, `pause()`, `resume()`, `step()`
- [x] Time-based event triggers - `_setup_time_callbacks()` for dawn/dusk/season

### Phase 8: Testing & Optimization (Weeks 15-16)

#### 8.1 Evaluation
- [ ] Create agent interview system (per paper)
- [ ] Implement believability metrics
- [ ] Add memory accuracy testing
- [ ] Create behavior consistency checks

#### 8.2 Performance
- [ ] Profile and optimize database queries
- [ ] Implement memory pruning/summarization
- [ ] Add caching layers
- [ ] Optimize LLM calls (batching, caching)

#### 8.3 Debugging Tools
- [ ] Create agent memory inspector
- [ ] Add simulation playback
- [ ] Implement agent behavior logging
- [ ] Create visualization for agent activities

---

## Part 5: Technical Stack

### 5.1 Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.12+ | Primary development |
| API | FastAPI | REST endpoints |
| Agent Framework | LangGraph | Multi-agent orchestration |
| LLM | Ollama (local) | Agent reasoning |
| Database | Supabase (PostgreSQL) | Persistence |
| Vector Store | pgvector | Semantic search |
| Caching | Redis | Message queue, caching |
| Task Queue | Celery | Background processing |

### 5.2 Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LOOP                          │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Time   │───▶│ Select  │───▶│ Execute │───▶│ Update  │  │
│  │  Tick   │    │ Agents  │    │ Actions │    │ World   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │                                            │        │
│       └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    AGENT CYCLE                              │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │Perceive │───▶│Retrieve │───▶│ Reason  │───▶│   Act   │  │
│  │         │    │ Memory  │    │ (LLM)   │    │         │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MEMORY STREAM (Supabase)               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 LLM Prompt Architecture

```python
# Agent Summary Description (cached, regenerated periodically)
AGENT_SUMMARY_TEMPLATE = """
Name: {name} (age: {age})
Innate traits: {traits}
{current_summary}
{daily_occupation}
{recent_progress}
"""

# Observation to Memory
IMPORTANCE_SCORING_PROMPT = """
On the scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, major discovery),
rate the likely poignancy of the following memory.

Memory: {memory_description}
Rating:
"""

# Reflection Generation
REFLECTION_PROMPT = """
Statements about {agent_name}:
{numbered_statements}

What 5 high-level insights can you infer from
the above statements? (example format: insight
(because of 1, 5, 3))
"""

# Planning
DAILY_PLAN_PROMPT = """
{agent_summary}
On {yesterday_date}, {agent_name} {yesterday_summary}
Today is {today_date}. Here is {agent_name}'s plan today in broad strokes:
1)
"""

# Reaction Decision
REACTION_PROMPT = """
{agent_summary}
It is {current_time}.
{agent_name}'s status: {current_status}
Observation: {observation}
Summary of relevant context from {agent_name}'s memory:
{memory_context}
Should {agent_name} react to the observation, and if so,
what would be an appropriate reaction?
"""

# Dialogue Generation
DIALOGUE_PROMPT = """
{agent_summary}
It is {current_time}.
{agent_name}'s status: {current_status}
Observation: {other_agent} is initiating a conversation with {agent_name}.
Summary of relevant context from {agent_name}'s memory:
{memory_about_other}
Here is the dialogue history:
{dialogue_history}
How would {agent_name} respond?
"""
```

---

## Part 6: Risk Mitigation

### 6.1 Performance Risks

| Risk | Mitigation |
|------|------------|
| LLM latency | Parallel execution, response caching |
| Memory growth | Periodic summarization, importance-based pruning |
| Database load | Connection pooling, read replicas, caching |
| Token costs | Local Ollama models, prompt optimization |

### 6.2 Behavioral Risks

| Risk | Mitigation |
|------|------------|
| Hallucination | Memory verification, citation requirements |
| Repetitive behavior | Reflection-based novelty seeking |
| Out-of-character | Consistent agent summaries, personality anchoring |
| Overly cooperative | Alignment-based decision modifiers |

### 6.3 Technical Risks

| Risk | Mitigation |
|------|------------|
| Model context limits | Memory retrieval prioritization |
| Embedding drift | Periodic re-embedding, version tracking |
| State inconsistency | Transactional updates, event sourcing |

---

## Part 7: Success Metrics

### 7.1 Believability Metrics (from Stanford paper)

1. **Self-knowledge**: Can agents accurately describe themselves?
2. **Memory**: Can agents recall specific past events and people?
3. **Planning**: Do agents maintain coherent long-term plans?
4. **Reactions**: Do agents respond appropriately to events?
5. **Reflections**: Can agents synthesize higher-level insights?

### 7.2 Emergent Behavior Metrics

1. **Information Diffusion**: Does news spread through the population?
2. **Relationship Formation**: Do new relationships form over time?
3. **Coordination**: Can agents organize group activities?
4. **Specialization**: Do agents develop distinct skill profiles?

### 7.3 Technical Metrics

1. **Simulation Speed**: Real-time or faster execution
2. **Memory Efficiency**: <1GB per 100 agents
3. **Response Quality**: >80% coherent responses
4. **Uptime**: 99% simulation availability

---

## Part 8: File Structure

```
universa/
├── builder/                    # Existing world generation
│   ├── generation/
│   ├── models/
│   └── utils/
│
├── agents/                     # NEW: Agent simulation layer
│   ├── __init__.py
│   ├── config.py              # Agent configuration
│   │
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── agent_state.py     # AgentState, CoreStats, CoreNeeds
│   │   ├── alignment.py       # Alignment system
│   │   ├── genetics.py        # Genome, inheritance
│   │   ├── memory.py          # Memory objects
│   │   └── skills.py          # SkillLevel, SkillDefinition
│   │
│   ├── memory/                # Memory systems
│   │   ├── __init__.py
│   │   ├── memory_stream.py   # Core memory stream
│   │   ├── episodic.py        # Episodic memory
│   │   ├── semantic.py        # Semantic memory
│   │   ├── procedural.py      # Procedural memory
│   │   └── retrieval.py       # Retrieval functions
│   │
│   ├── reasoning/             # Agent reasoning
│   │   ├── __init__.py
│   │   ├── reflection.py      # Reflection generation
│   │   ├── planning.py        # Plan generation/decomposition
│   │   ├── reaction.py        # Reaction decisions
│   │   └── dialogue.py        # Conversation generation
│   │
│   ├── skills/                # Skill system
│   │   ├── __init__.py
│   │   ├── taxonomy.py        # Skill tree definition
│   │   ├── navigator.py       # Skill navigation
│   │   ├── progression.py     # XP and leveling
│   │   └── architect.py       # Skill Architect agent
│   │
│   ├── tools/                 # MCP-compliant tools
│   │   ├── __init__.py
│   │   ├── registry.py        # Tool registry
│   │   ├── world_tools.py     # World interaction
│   │   ├── skill_tools.py     # Skill usage
│   │   └── memory_tools.py    # Memory recall
│   │
│   ├── world/                 # World interface
│   │   ├── __init__.py
│   │   ├── interface.py       # WorldInterface class
│   │   ├── queries.py         # Location/resource queries
│   │   └── pathfinding.py     # A* pathfinding
│   │
│   ├── simulation/            # Simulation orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Main simulation loop
│   │   ├── scheduler.py       # Agent scheduling
│   │   ├── time_manager.py    # Game time management
│   │   └── events.py          # Event system
│   │
│   ├── llm/                   # LLM integration
│   │   ├── __init__.py
│   │   ├── ollama_pool.py     # Ollama model pool
│   │   ├── prompts.py         # Prompt templates
│   │   └── parser.py          # Response parsing
│   │
│   └── db/                    # Database layer
│       ├── __init__.py
│       ├── supabase_client.py # Supabase connection
│       ├── repositories/      # CRUD operations
│       │   ├── agents.py
│       │   ├── memories.py
│       │   ├── skills.py
│       │   └── relationships.py
│       └── migrations/        # Schema migrations
│
├── supabase/                  # Supabase configuration
│   ├── config.toml
│   └── migrations/
│
├── tasks/                     # Task documentation
│   └── AGENTIC_IMPLEMENTATION_PLAN.md
│
├── docs/                      # Documentation
│   ├── CORE_AGENTS.md
│   ├── MEMORY_TOOLS.md
│   ├── SKILL_SYSTEM.md
│   └── 2304.03442v2.pdf
│
└── tests/                     # Tests
    ├── agents/
    └── integration/
```

---

## Appendix A: Sample Agent Lifecycle

```
DAY 1 - Agent "Elena the Blacksmith" Initialization

06:00 - Wake up
  [Memory] Elena woke up in her small room above the forge
  [Plan] Today: open forge, fulfill orders, visit market

07:00 - Morning routine
  [Memory] Elena ate bread and cheese for breakfast
  [Need Update] hunger: 0.3 → 0.0

08:00 - Open forge
  [Memory] Elena lit the forge fire and prepared her tools
  [Skill Use] crafting.smithing.blacksmithing (level 12)

10:00 - Customer arrives
  [Perception] Farmer Thomas entered the forge
  [Memory Retrieval] "Thomas ordered a plow blade last week"
  [Dialogue] "Good morning Thomas! Your blade is ready."
  [Memory] Thomas paid 5 silver for the plow blade

12:00 - Lunch break
  [Plan Update] Visit Hobbs Cafe for lunch
  [Memory] Elena walked to Hobbs Cafe
  [Perception] Noticed Sam and Isabella talking about election
  [Memory] Overheard Sam is running for mayor

14:00 - Reflection triggered (importance sum > 150)
  [Reflection] "Thomas is a reliable customer who pays on time"
  [Reflection] "The town seems interested in the upcoming election"

15:00 - Return to work
  [Memory] Elena began forging horseshoes
  [Skill XP] +15 XP to blacksmithing
  [Stat Improvement] Strength: 14 → 15 (rare event)

18:00 - Close forge
  [Memory] Elena cleaned the forge and banked the fire
  [Plan] Evening: visit tavern, socialize

20:00 - Tavern
  [Perception] Klaus Mueller is reading alone
  [Decision] Approach Klaus based on social need (0.6)
  [Dialogue] "Mind if I join you? I'm Elena."
  [Memory] Met Klaus, he's researching gentrification
  [Relationship] New: Elena ↔ Klaus (familiarity: 0.1)

22:00 - Return home, sleep
  [Memory] Elena went to bed after a productive day
  [Need Update] rest: 0.7 → 0.0
```

---

## Appendix B: Key Differences from Stanford Paper

| Aspect | Stanford Paper | Universa Implementation |
|--------|----------------|------------------------|
| Environment | 2D Sims-like game | Procedurally generated fantasy world |
| Agents | 25 fixed characters | Unlimited, with genetics/reproduction |
| Skills | Implicit | Explicit 100+ skill taxonomy |
| Stats | None | D&D-style 6 stat system |
| Needs | Implicit | Explicit survival needs |
| Alignment | None | Good/Evil, Lawful/Chaotic axes |
| Magic | None | Full magic system integration |
| Politics | Limited | Full faction system |
| Time scale | 2 days simulation | Long-term multi-generation |
| LLM | GPT-3.5/4 (cloud) | Ollama (local) |
| Storage | Local files | Supabase (PostgreSQL) |

---

## Appendix C: Supabase Setup Checklist

1. **Create Project**
   - [ ] Create new Supabase project
   - [ ] Note project URL and anon key
   - [ ] Enable pgvector extension

2. **Configure MCP**
   - [ ] Add Supabase MCP to `.mcp.json`
   - [ ] Configure connection credentials
   - [ ] Test connection

3. **Run Migrations**
   - [ ] Create tables in order (worlds → agents → memories → etc.)
   - [ ] Create indexes
   - [ ] Set up RLS policies

4. **Seed Data**
   - [ ] Import skill taxonomy
   - [ ] Create test world
   - [ ] Generate initial agent population

---

*Document Version: 1.0*
*Created: 2025-12-01*
*Last Updated: 2025-12-01*
