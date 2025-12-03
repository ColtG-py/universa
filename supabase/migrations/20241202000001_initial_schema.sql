-- ============================================================================
-- Universa Initial Schema
-- Database tables for world generation, game sessions, and agent simulation
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For embeddings

-- ============================================================================
-- WORLDS
-- ============================================================================

CREATE TABLE IF NOT EXISTS worlds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    seed BIGINT NOT NULL,
    size VARCHAR(20) NOT NULL DEFAULT 'MEDIUM',
    dimensions INT NOT NULL DEFAULT 1024,
    status VARCHAR(20) NOT NULL DEFAULT 'generating',

    -- World configuration
    config JSONB NOT NULL DEFAULT '{}',

    -- Statistics
    num_settlements INT DEFAULT 0,
    num_agents INT DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_size CHECK (size IN ('SMALL', 'MEDIUM', 'LARGE', 'HUGE')),
    CONSTRAINT valid_status CHECK (status IN ('generating', 'ready', 'failed', 'archived'))
);

CREATE INDEX idx_worlds_status ON worlds(status);
CREATE INDEX idx_worlds_created_at ON worlds(created_at DESC);

-- ============================================================================
-- WORLD TILES (Chunked storage)
-- ============================================================================

CREATE TABLE IF NOT EXISTS world_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,
    chunk_x INT NOT NULL,
    chunk_y INT NOT NULL,

    -- Chunk data (16x16 tiles compressed)
    tile_data JSONB NOT NULL,

    -- Metadata
    biome_summary JSONB DEFAULT '{}',
    has_settlement BOOLEAN DEFAULT FALSE,
    has_road BOOLEAN DEFAULT FALSE,
    has_river BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_chunk UNIQUE (world_id, chunk_x, chunk_y)
);

CREATE INDEX idx_world_chunks_world ON world_chunks(world_id);
CREATE INDEX idx_world_chunks_coords ON world_chunks(world_id, chunk_x, chunk_y);

-- ============================================================================
-- SETTLEMENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS settlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    settlement_type VARCHAR(50) NOT NULL DEFAULT 'village',

    -- Position
    x INT NOT NULL,
    y INT NOT NULL,

    -- Stats
    population INT DEFAULT 0,
    prosperity FLOAT DEFAULT 0.5,
    defense_level FLOAT DEFAULT 0.5,

    -- Faction
    faction_id UUID,

    -- Data
    resources JSONB DEFAULT '{}',
    buildings JSONB DEFAULT '[]',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_settlements_world ON settlements(world_id);
CREATE INDEX idx_settlements_faction ON settlements(faction_id);
CREATE INDEX idx_settlements_coords ON settlements(world_id, x, y);

-- ============================================================================
-- FACTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS factions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    faction_type VARCHAR(50) NOT NULL DEFAULT 'kingdom',

    -- Leadership
    leader_id UUID,
    capital_id UUID,

    -- Relations
    relations JSONB DEFAULT '{}',

    -- Stats
    power_level FLOAT DEFAULT 0.5,
    territory_size INT DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_factions_world ON factions(world_id);

-- ============================================================================
-- GAME SESSIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS game_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,

    status VARCHAR(20) NOT NULL DEFAULT 'active',

    -- Time tracking
    current_tick BIGINT DEFAULT 0,
    game_time VARCHAR(50) DEFAULT 'Day 1, 08:00',

    -- Settings
    settings JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    CONSTRAINT valid_session_status CHECK (status IN ('active', 'paused', 'ended'))
);

CREATE INDEX idx_sessions_world ON game_sessions(world_id);
CREATE INDEX idx_sessions_status ON game_sessions(status);

-- ============================================================================
-- PLAYERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    name VARCHAR(100) NOT NULL,

    -- Position
    x INT NOT NULL,
    y INT NOT NULL,

    -- Vitals
    health FLOAT DEFAULT 100.0,
    max_health FLOAT DEFAULT 100.0,
    stamina FLOAT DEFAULT 100.0,
    max_stamina FLOAT DEFAULT 100.0,

    -- Stats and inventory
    stats JSONB DEFAULT '{}',
    inventory JSONB DEFAULT '[]',
    gold INT DEFAULT 0,

    -- Appearance
    appearance JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_players_session ON players(session_id);

-- ============================================================================
-- PARTIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS parties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    leader_id UUID NOT NULL REFERENCES players(id) ON DELETE CASCADE,

    formation VARCHAR(20) DEFAULT 'follow',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parties_session ON parties(session_id);
CREATE INDEX idx_parties_leader ON parties(leader_id);

-- ============================================================================
-- AGENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,

    name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(50) NOT NULL DEFAULT 'individual',
    role VARCHAR(50),

    -- Position
    x INT NOT NULL,
    y INT NOT NULL,

    -- State
    health FLOAT DEFAULT 1.0,
    energy FLOAT DEFAULT 1.0,
    tier VARCHAR(20) DEFAULT 'background',

    -- Current activity
    current_action TEXT,
    current_thought TEXT,

    -- Affiliations
    settlement_id UUID REFERENCES settlements(id) ON DELETE SET NULL,
    faction_id UUID REFERENCES factions(id) ON DELETE SET NULL,
    party_id UUID REFERENCES parties(id) ON DELETE SET NULL,

    -- Stats and traits
    stats JSONB DEFAULT '{}',
    traits JSONB DEFAULT '[]',
    personality JSONB DEFAULT '{}',

    -- For party members
    is_party_member BOOLEAN DEFAULT FALSE,
    loyalty FLOAT DEFAULT 0.5,
    morale FLOAT DEFAULT 1.0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_tier CHECK (tier IN ('active', 'nearby', 'background', 'dormant'))
);

CREATE INDEX idx_agents_session ON agents(session_id);
CREATE INDEX idx_agents_world ON agents(world_id);
CREATE INDEX idx_agents_settlement ON agents(settlement_id);
CREATE INDEX idx_agents_faction ON agents(faction_id);
CREATE INDEX idx_agents_party ON agents(party_id);
CREATE INDEX idx_agents_tier ON agents(tier);
CREATE INDEX idx_agents_coords ON agents(session_id, x, y);

-- ============================================================================
-- AGENT MEMORIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    content TEXT NOT NULL,
    memory_type VARCHAR(20) NOT NULL DEFAULT 'observation',

    importance FLOAT DEFAULT 0.5,

    -- Embedding for semantic search
    embedding vector(1536),

    -- Access tracking
    access_count INT DEFAULT 0,
    last_accessed TIMESTAMPTZ,

    -- Context
    game_tick BIGINT,
    location_x INT,
    location_y INT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_memory_type CHECK (memory_type IN ('observation', 'reflection', 'plan'))
);

CREATE INDEX idx_memories_agent ON agent_memories(agent_id);
CREATE INDEX idx_memories_session ON agent_memories(session_id);
CREATE INDEX idx_memories_type ON agent_memories(memory_type);
CREATE INDEX idx_memories_importance ON agent_memories(importance DESC);

-- Vector similarity index (requires pgvector)
CREATE INDEX idx_memories_embedding ON agent_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- AGENT RELATIONSHIPS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    other_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    -- Relationship dimensions
    familiarity FLOAT DEFAULT 0.0,
    trust FLOAT DEFAULT 0.5,
    affection FLOAT DEFAULT 0.5,
    respect FLOAT DEFAULT 0.5,

    relationship_type VARCHAR(50),

    -- Interaction history
    interaction_count INT DEFAULT 0,
    last_interaction TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_relationship UNIQUE (agent_id, other_agent_id, session_id)
);

CREATE INDEX idx_relationships_agent ON agent_relationships(agent_id);
CREATE INDEX idx_relationships_other ON agent_relationships(other_agent_id);
CREATE INDEX idx_relationships_session ON agent_relationships(session_id);

-- ============================================================================
-- AGENT PLANS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    plan_level VARCHAR(20) NOT NULL, -- day, hour, action

    description TEXT NOT NULL,
    start_time VARCHAR(20),
    duration_minutes INT DEFAULT 30,
    location VARCHAR(255),

    status VARCHAR(20) DEFAULT 'pending',
    priority INT DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    CONSTRAINT valid_plan_level CHECK (plan_level IN ('day', 'hour', 'action')),
    CONSTRAINT valid_plan_status CHECK (status IN ('pending', 'in_progress', 'completed', 'abandoned'))
);

CREATE INDEX idx_plans_agent ON agent_plans(agent_id);
CREATE INDEX idx_plans_session ON agent_plans(session_id);
CREATE INDEX idx_plans_status ON agent_plans(status);

-- ============================================================================
-- LLM CALL LOG
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    session_id UUID REFERENCES game_sessions(id) ON DELETE SET NULL,

    purpose VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,

    prompt_text TEXT,
    prompt_summary VARCHAR(500),
    response_text TEXT,
    response_summary VARCHAR(500),

    tokens_in INT DEFAULT 0,
    tokens_out INT DEFAULT 0,
    duration_ms FLOAT DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_llm_calls_agent ON llm_calls(agent_id);
CREATE INDEX idx_llm_calls_session ON llm_calls(session_id);
CREATE INDEX idx_llm_calls_created ON llm_calls(created_at DESC);

-- ============================================================================
-- SIMULATION EVENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS simulation_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    event_type VARCHAR(50) NOT NULL,
    description TEXT,

    -- Location
    x INT,
    y INT,

    -- Related entities
    involved_agents UUID[] DEFAULT '{}',
    involved_settlements UUID[] DEFAULT '{}',

    -- Event data
    data JSONB DEFAULT '{}',

    game_tick BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_session ON simulation_events(session_id);
CREATE INDEX idx_events_type ON simulation_events(event_type);
CREATE INDEX idx_events_tick ON simulation_events(game_tick);

-- ============================================================================
-- CONVERSATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    -- Participants
    player_id UUID REFERENCES players(id) ON DELETE SET NULL,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,

    -- State
    turn_count INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_agent ON conversations(agent_id);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,

    turn_number INT NOT NULL,
    speaker VARCHAR(20) NOT NULL, -- player, agent

    message TEXT NOT NULL,
    emotion VARCHAR(50),

    -- Debug info
    memories_retrieved TEXT[],
    reasoning TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_turns_conversation ON conversation_turns(conversation_id);

-- ============================================================================
-- CHAT MESSAGES (Party chat, local chat, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    channel VARCHAR(20) NOT NULL, -- party, local, global

    sender_id UUID,
    sender_name VARCHAR(100) NOT NULL,
    sender_type VARCHAR(20) NOT NULL, -- player, agent, system

    message TEXT NOT NULL,

    -- Location (for local messages)
    x INT,
    y INT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_session ON chat_messages(session_id);
CREATE INDEX idx_chat_channel ON chat_messages(channel);
CREATE INDEX idx_chat_created ON chat_messages(created_at DESC);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all tables with updated_at
CREATE TRIGGER update_worlds_updated_at
    BEFORE UPDATE ON worlds
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_settlements_updated_at
    BEFORE UPDATE ON settlements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_factions_updated_at
    BEFORE UPDATE ON factions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON game_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_players_updated_at
    BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_parties_updated_at
    BEFORE UPDATE ON parties
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_relationships_updated_at
    BEFORE UPDATE ON agent_relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - for multi-tenant setup)
-- ============================================================================

-- Enable RLS on all tables (can be customized per deployment)
-- ALTER TABLE worlds ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE game_sessions ENABLE ROW LEVEL SECURITY;
-- etc.
