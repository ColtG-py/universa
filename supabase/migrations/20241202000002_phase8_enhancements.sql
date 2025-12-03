-- ============================================================================
-- Phase 8 Enhancements Migration
-- Additional tables and columns for full integration
-- ============================================================================

-- ============================================================================
-- UPDATE AGENTS TABLE for Hierarchical Scheduler
-- ============================================================================

-- Update tier constraint to include all new tiers
ALTER TABLE agents DROP CONSTRAINT IF EXISTS valid_tier;
ALTER TABLE agents ADD CONSTRAINT valid_tier CHECK (
    tier IN ('player_party', 'active', 'nearby', 'same_settlement', 'same_region', 'background', 'dormant')
);

-- Add columns for hierarchical scheduling
ALTER TABLE agents ADD COLUMN IF NOT EXISTS last_executed_tick BIGINT DEFAULT 0;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS execution_frequency INT DEFAULT 10;
ALTER TABLE agents ADD COLUMN IF NOT EXISTS in_active_interaction BOOLEAN DEFAULT FALSE;

-- ============================================================================
-- ENHANCED CONVERSATIONS (Multi-participant support)
-- ============================================================================

-- Add location to conversations
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS location_x INT;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS location_y INT;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS state VARCHAR(20) DEFAULT 'active';

-- Create conversation participants table for multi-participant support
CREATE TABLE IF NOT EXISTS conversation_participants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    participant_id UUID NOT NULL,  -- Can be player or agent
    participant_type VARCHAR(20) NOT NULL,  -- 'player' or 'agent'
    participant_name VARCHAR(100) NOT NULL,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    left_at TIMESTAMPTZ,

    CONSTRAINT valid_participant_type CHECK (participant_type IN ('player', 'agent'))
);

CREATE INDEX IF NOT EXISTS idx_conv_participants_conv ON conversation_participants(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_participants_id ON conversation_participants(participant_id);

-- Update conversation_turns for speaker identification
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS speaker_id UUID;
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS speaker_name VARCHAR(100);

-- ============================================================================
-- COLLECTIVE AGENTS (Settlements as agents)
-- ============================================================================

CREATE TABLE IF NOT EXISTS collective_agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    collective_type VARCHAR(50) NOT NULL,  -- settlement, kingdom, guild
    entity_id UUID NOT NULL,  -- Reference to settlement or faction

    -- Collective state
    aggregate_mood FLOAT DEFAULT 0.5,
    aggregate_wealth FLOAT DEFAULT 0.5,
    aggregate_safety FLOAT DEFAULT 0.5,

    -- Last decisions
    recent_decisions JSONB DEFAULT '[]',

    -- Timing
    last_tick BIGINT DEFAULT 0,
    tick_interval INT DEFAULT 100,  -- Collective thinks less often

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_collective_type CHECK (collective_type IN ('settlement', 'kingdom', 'guild'))
);

CREATE INDEX IF NOT EXISTS idx_collective_session ON collective_agents(session_id);
CREATE INDEX IF NOT EXISTS idx_collective_entity ON collective_agents(entity_id);

-- ============================================================================
-- WORLD GENERATION PROGRESS
-- ============================================================================

CREATE TABLE IF NOT EXISTS world_generation_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    world_id UUID NOT NULL REFERENCES worlds(id) ON DELETE CASCADE,

    current_pass INT DEFAULT 0,
    total_passes INT DEFAULT 18,
    current_pass_name VARCHAR(100),
    progress_percent FLOAT DEFAULT 0,

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Pass history
    pass_history JSONB DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_gen_progress_world ON world_generation_progress(world_id);

-- ============================================================================
-- DEBUG/INSPECTION SNAPSHOTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,

    -- Snapshot data
    snapshot_data JSONB NOT NULL,

    -- Cognitive state
    cognitive_phase VARCHAR(50),
    current_thought TEXT,
    current_action TEXT,

    -- Timing
    game_tick BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_agent ON agent_snapshots(agent_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON agent_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_tick ON agent_snapshots(game_tick DESC);

-- Keep only last 100 snapshots per agent
CREATE OR REPLACE FUNCTION cleanup_old_snapshots()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM agent_snapshots
    WHERE agent_id = NEW.agent_id
    AND id NOT IN (
        SELECT id FROM agent_snapshots
        WHERE agent_id = NEW.agent_id
        ORDER BY created_at DESC
        LIMIT 100
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_cleanup_snapshots
    AFTER INSERT ON agent_snapshots
    FOR EACH ROW EXECUTE FUNCTION cleanup_old_snapshots();

-- ============================================================================
-- PLAYER INTERACTIONS LOG
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    player_id UUID NOT NULL REFERENCES players(id) ON DELETE CASCADE,

    interaction_type VARCHAR(50) NOT NULL,  -- talk, trade, attack, etc.
    target_type VARCHAR(50) NOT NULL,  -- agent, object, tile, settlement
    target_id VARCHAR(255),

    action VARCHAR(100) NOT NULL,
    result JSONB DEFAULT '{}',

    location_x INT,
    location_y INT,
    game_tick BIGINT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON player_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_player ON player_interactions(player_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON player_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_interactions_tick ON player_interactions(game_tick);

-- ============================================================================
-- PARTY MEMBER STATUS
-- ============================================================================

CREATE TABLE IF NOT EXISTS party_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    party_id UUID NOT NULL REFERENCES parties(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,

    role VARCHAR(50),  -- warrior, mage, healer, rogue, companion
    join_order INT DEFAULT 0,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    formation_position INT DEFAULT 0,  -- Position in formation

    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    left_at TIMESTAMPTZ,

    CONSTRAINT unique_party_member UNIQUE (party_id, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_party_members_party ON party_members(party_id);
CREATE INDEX IF NOT EXISTS idx_party_members_agent ON party_members(agent_id);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE TRIGGER update_collective_agents_updated_at
    BEFORE UPDATE ON collective_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for active session summary
CREATE OR REPLACE VIEW session_summary AS
SELECT
    gs.id as session_id,
    gs.world_id,
    gs.status,
    gs.current_tick,
    gs.game_time,
    w.name as world_name,
    COUNT(DISTINCT a.id) as agent_count,
    COUNT(DISTINCT CASE WHEN a.tier = 'player_party' THEN a.id END) as party_size,
    COUNT(DISTINCT CASE WHEN a.tier IN ('active', 'nearby') THEN a.id END) as active_agents,
    gs.created_at,
    gs.updated_at
FROM game_sessions gs
JOIN worlds w ON gs.world_id = w.id
LEFT JOIN agents a ON a.session_id = gs.id
GROUP BY gs.id, w.name;

-- View for agent tier distribution
CREATE OR REPLACE VIEW agent_tier_distribution AS
SELECT
    session_id,
    tier,
    COUNT(*) as count
FROM agents
GROUP BY session_id, tier;
