"""
Debug API Router
Endpoints for inspecting agent internals, memories, and simulation state.
"""

from fastapi import APIRouter, HTTPException, Query
from uuid import UUID
from typing import Optional
import logging

from api.models.responses import (
    MemoryListResponse,
    AgentThoughtsResponse,
    AgentPlansResponse,
    RelationshipListResponse,
    SimulationStatsResponse,
    LLMCallHistoryResponse
)
from api.services.debug_service import DebugService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
debug_service = DebugService()


# =============================================================================
# Agent Memory Inspection
# =============================================================================

@router.get("/agents/{agent_id}/memories", response_model=MemoryListResponse)
async def get_agent_memories(
    agent_id: UUID,
    session_id: UUID,
    memory_type: Optional[str] = Query(None, description="Filter: observation, reflection, plan"),
    min_importance: float = Query(0.0, description="Minimum importance score"),
    limit: int = Query(50, description="Maximum memories to return"),
    include_embeddings: bool = Query(False, description="Include embedding vectors")
):
    """
    Get agent's memory stream with optional filters.

    Memory types:
    - observation: What the agent has perceived
    - reflection: Higher-level insights
    - plan: Future intentions
    """
    try:
        memories = await debug_service.get_memories(
            session_id=str(session_id),
            agent_id=str(agent_id),
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit,
            include_embeddings=include_embeddings
        )
        return MemoryListResponse(
            agent_id=str(agent_id),
            memories=memories,
            total=len(memories)
        )
    except Exception as e:
        logger.error(f"Failed to get memories for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/memories/search")
async def search_agent_memories(
    agent_id: UUID,
    session_id: UUID,
    query: str,
    limit: int = 10
):
    """
    Search agent's memories using semantic similarity.

    Returns memories most relevant to the query text.
    """
    try:
        memories = await debug_service.search_memories(
            session_id=str(session_id),
            agent_id=str(agent_id),
            query=query,
            limit=limit
        )
        return {"agent_id": str(agent_id), "query": query, "memories": memories}
    except Exception as e:
        logger.error(f"Memory search failed for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Agent Cognitive State
# =============================================================================

@router.get("/agents/{agent_id}/thoughts", response_model=AgentThoughtsResponse)
async def get_agent_thoughts(agent_id: UUID, session_id: UUID):
    """
    Get agent's current cognitive state.

    Returns:
    - Current phase (perceiving, retrieving, planning, acting, etc.)
    - Current thought/reasoning
    - Retrieved memories being considered
    - Pending action
    """
    try:
        thoughts = await debug_service.get_thoughts(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        return thoughts
    except Exception as e:
        logger.error(f"Failed to get thoughts for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/plans", response_model=AgentPlansResponse)
async def get_agent_plans(agent_id: UUID, session_id: UUID):
    """
    Get agent's planning hierarchy.

    Returns:
    - Day plan: Broad strokes for the day
    - Hour plan: Current hour's activities
    - Action plan: Current 5-15 minute action
    - Progress through plans
    """
    try:
        plans = await debug_service.get_plans(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        return plans
    except Exception as e:
        logger.error(f"Failed to get plans for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/needs")
async def get_agent_needs(agent_id: UUID, session_id: UUID):
    """
    Get agent's current needs state.

    Returns hunger, thirst, rest, warmth, safety, social levels (0-1).
    """
    try:
        needs = await debug_service.get_needs(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        return {"agent_id": str(agent_id), "needs": needs}
    except Exception as e:
        logger.error(f"Failed to get needs for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Agent Relationships
# =============================================================================

@router.get("/agents/{agent_id}/relationships", response_model=RelationshipListResponse)
async def get_agent_relationships(
    agent_id: UUID,
    session_id: UUID,
    min_familiarity: float = 0.0
):
    """
    Get agent's relationships with other agents.

    Returns familiarity, trust, affection, respect for each known agent.
    """
    try:
        relationships = await debug_service.get_relationships(
            session_id=str(session_id),
            agent_id=str(agent_id),
            min_familiarity=min_familiarity
        )
        return RelationshipListResponse(
            agent_id=str(agent_id),
            relationships=relationships,
            total=len(relationships)
        )
    except Exception as e:
        logger.error(f"Failed to get relationships for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/relationships/{other_id}")
async def get_relationship_detail(
    agent_id: UUID,
    other_id: UUID,
    session_id: UUID
):
    """Get detailed relationship between two agents."""
    try:
        relationship = await debug_service.get_relationship_detail(
            session_id=str(session_id),
            agent_id=str(agent_id),
            other_id=str(other_id)
        )
        return relationship
    except Exception as e:
        logger.error(f"Failed to get relationship {agent_id} -> {other_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LLM Call History
# =============================================================================

@router.get("/agents/{agent_id}/llm-history", response_model=LLMCallHistoryResponse)
async def get_llm_history(
    agent_id: UUID,
    session_id: UUID,
    limit: int = 10
):
    """
    Get recent LLM calls made by this agent.

    Useful for debugging agent reasoning.
    """
    try:
        history = await debug_service.get_llm_history(
            session_id=str(session_id),
            agent_id=str(agent_id),
            limit=limit
        )
        return LLMCallHistoryResponse(
            agent_id=str(agent_id),
            calls=history,
            total=len(history)
        )
    except Exception as e:
        logger.error(f"Failed to get LLM history for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Simulation Statistics
# =============================================================================

@router.get("/simulation/stats", response_model=SimulationStatsResponse)
async def get_simulation_stats(session_id: UUID):
    """
    Get overall simulation statistics.

    Returns tick count, agent counts, performance metrics, etc.
    """
    try:
        stats = await debug_service.get_simulation_stats(str(session_id))
        return stats
    except Exception as e:
        logger.error(f"Failed to get simulation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/agents-by-tier")
async def get_agents_by_tier(session_id: UUID):
    """
    Get agents grouped by execution tier.

    Shows which agents are actively thinking vs dormant.
    """
    try:
        tiers = await debug_service.get_agents_by_tier(str(session_id))
        return {"session_id": str(session_id), "tiers": tiers}
    except Exception as e:
        logger.error(f"Failed to get agents by tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/active-agents")
async def get_active_agents(session_id: UUID):
    """Get list of agents that executed in the last tick."""
    try:
        agents = await debug_service.get_active_agents(str(session_id))
        return {"session_id": str(session_id), "active_agents": agents}
    except Exception as e:
        logger.error(f"Failed to get active agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation/events")
async def get_simulation_events(
    session_id: UUID,
    limit: int = 50,
    event_type: Optional[str] = None
):
    """Get recent simulation events."""
    try:
        events = await debug_service.get_events(
            session_id=str(session_id),
            limit=limit,
            event_type=event_type
        )
        return {"session_id": str(session_id), "events": events}
    except Exception as e:
        logger.error(f"Failed to get simulation events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Collective Agents (Settlements/Kingdoms)
# =============================================================================

@router.get("/collectives/{collective_id}")
async def get_collective_state(collective_id: UUID, session_id: UUID):
    """
    Get state of a collective agent (settlement or kingdom).

    Shows aggregate mood, resources, recent decisions.
    """
    try:
        state = await debug_service.get_collective_state(
            session_id=str(session_id),
            collective_id=str(collective_id)
        )
        return state
    except Exception as e:
        logger.error(f"Failed to get collective state {collective_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collectives/{collective_id}/members")
async def get_collective_members(collective_id: UUID, session_id: UUID):
    """Get all agents belonging to a collective."""
    try:
        members = await debug_service.get_collective_members(
            session_id=str(session_id),
            collective_id=str(collective_id)
        )
        return {"collective_id": str(collective_id), "members": members}
    except Exception as e:
        logger.error(f"Failed to get collective members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LLM Usage Statistics
# =============================================================================

@router.get("/llm/stats")
async def get_llm_stats(session_id: UUID):
    """
    Get LLM usage statistics.

    Returns token counts, call counts, duration stats, and breakdown by purpose.
    """
    try:
        stats = await debug_service.get_llm_stats(str(session_id))
        return {"session_id": str(session_id), "llm_stats": stats}
    except Exception as e:
        logger.error(f"Failed to get LLM stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/recent-calls")
async def get_recent_llm_calls(
    session_id: UUID,
    limit: int = Query(20, description="Max calls to return"),
    purpose: Optional[str] = Query(None, description="Filter by purpose")
):
    """
    Get recent LLM calls across all agents.

    Useful for monitoring overall LLM usage.
    """
    try:
        from agents.debug.llm_tracker import get_tracker
        tracker = get_tracker()
        calls = tracker.get_recent_calls(limit=limit, purpose=purpose)
        return {
            "session_id": str(session_id),
            "calls": [c.to_dict() for c in calls]
        }
    except Exception as e:
        logger.error(f"Failed to get recent LLM calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Agent Snapshot / State Inspection
# =============================================================================

@router.get("/agents/{agent_id}/snapshot")
async def get_agent_snapshot(agent_id: UUID, session_id: UUID):
    """
    Get a complete snapshot of an agent's state.

    Includes cognitive state, plans, needs, memories, and execution stats.
    """
    try:
        from agents.debug.inspector import get_inspector
        inspector = get_inspector()
        snapshot = inspector.get_latest_snapshot(agent_id)
        if snapshot:
            return snapshot.to_dict()
        return {"error": "No snapshot available for this agent"}
    except Exception as e:
        logger.error(f"Failed to get agent snapshot {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/snapshot-history")
async def get_agent_snapshot_history(
    agent_id: UUID,
    session_id: UUID,
    limit: int = 20
):
    """
    Get recent snapshot history for an agent.

    Useful for tracking agent state changes over time.
    """
    try:
        from agents.debug.inspector import get_inspector
        inspector = get_inspector()
        snapshots = inspector.get_snapshot_history(agent_id, limit=limit)
        return {
            "agent_id": str(agent_id),
            "snapshots": [s.to_dict() for s in snapshots]
        }
    except Exception as e:
        logger.error(f"Failed to get snapshot history {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationship-graph")
async def get_relationship_graph(session_id: UUID):
    """
    Get relationship graph data for visualization.

    Returns nodes (agents) and edges (relationships) suitable for D3.js or similar.
    """
    try:
        from agents.debug.inspector import get_inspector
        inspector = get_inspector()

        # Build graph structure
        nodes = []
        edges = []

        # Get all agents from the inspector
        for agent_id in inspector._relationships.keys():
            relationships = inspector.get_relationships(agent_id)

            # Add agent as node (if not already added)
            agent_node = {
                "id": str(agent_id),
                "label": "Agent",  # Would be replaced with actual name
            }
            if agent_node not in nodes:
                nodes.append(agent_node)

            # Add edges for relationships
            for rel in relationships:
                edges.append({
                    "source": str(agent_id),
                    "target": rel.other_id,
                    "familiarity": rel.familiarity,
                    "trust": rel.trust,
                    "affection": rel.affection,
                    "type": rel.relationship_type,
                })

                # Add target node if not exists
                target_node = {"id": rel.other_id, "label": rel.other_name}
                if target_node not in nodes:
                    nodes.append(target_node)

        return {
            "session_id": str(session_id),
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Failed to get relationship graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
