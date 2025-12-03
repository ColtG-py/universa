"""
Game Session API Router
Endpoints for managing game sessions and simulation.
"""

from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
import logging

from api.models.requests import (
    CreateSessionRequest,
    TickRequest
)
from api.models.responses import (
    SessionResponse,
    TickResponse,
    SessionStateResponse
)
from api.services.game_service import GameService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
game_service = GameService()


@router.get("/sessions")
async def list_sessions(
    world_id: str = None,
    status: str = None,
    limit: int = 50
):
    """
    List available game sessions.

    Filter by world_id or status ('active', 'ended').
    Use this to show resumable games.
    """
    try:
        sessions = await game_service.list_sessions(
            world_id=world_id,
            status=status,
            limit=limit
        )
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new game session.

    This initializes the simulation with:
    - The specified world
    - Player character
    - Party members
    - Nearby world agents
    """
    try:
        # Convert Pydantic models to dicts for the service
        player_dict = request.player.model_dump() if request.player else {}
        party_dict = request.party.model_dump() if request.party else None
        settings_dict = request.settings.model_dump() if request.settings else None

        session = await game_service.create_session(
            world_id=request.world_id,
            player_config=player_dict,
            party_config=party_dict,
            settings=settings_dict
        )
        logger.info(f"Created game session: {session.id}")
        # The service already returns a properly formed SessionResponse
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: UUID):
    """Get the current state of a game session with player data."""
    try:
        session_state = await game_service.get_session_state(str(session_id))
        if not session_state:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_state
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/tick", response_model=TickResponse)
async def tick_session(session_id: UUID, request: TickRequest = None):
    """
    Advance the simulation by one or more ticks.

    Returns all updates that occurred during the tick(s).
    """
    try:
        num_ticks = request.num_ticks if request else 1
        result = await game_service.tick(str(session_id), num_ticks=num_ticks)
        return result
    except Exception as e:
        logger.error(f"Failed to tick session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/start")
async def start_auto_tick(session_id: UUID):
    """
    Start automatic tick advancement.

    The simulation will tick at the configured interval
    until paused or stopped.
    """
    try:
        await game_service.start_auto_tick(str(session_id))
        return {"status": "running", "session_id": str(session_id)}
    except Exception as e:
        logger.error(f"Failed to start auto-tick for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/pause")
async def pause_session(session_id: UUID):
    """Pause the simulation (stops auto-tick)."""
    try:
        result = await game_service.stop_auto_tick(str(session_id))
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "paused", "session_id": str(session_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/resume")
async def resume_session(session_id: UUID):
    """
    Resume a previously saved game session.

    Loads session from database and initializes it for play.
    """
    try:
        session = await game_service.resume_session(str(session_id))
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or cannot be resumed")
        return {"status": "running", "session_id": str(session_id), "session": session}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/stop")
async def stop_session(session_id: UUID):
    """Stop and save the game session."""
    try:
        result = await game_service.end_session(str(session_id))
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "stopped", "session_id": str(session_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/time")
async def get_game_time(session_id: UUID):
    """Get the current in-game time."""
    try:
        time_info = await game_service.get_game_time(str(session_id))
        return time_info
    except Exception as e:
        logger.error(f"Failed to get time for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/events")
async def get_recent_events(session_id: UUID, limit: int = 20):
    """Get recent world events for the session."""
    try:
        events = await game_service.get_recent_events(str(session_id), limit=limit)
        return {"events": events}
    except Exception as e:
        logger.error(f"Failed to get events for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Hierarchical Agent Tier Endpoints
# =============================================================================

@router.get("/sessions/{session_id}/tiers")
async def get_tier_distribution(session_id: UUID):
    """
    Get count of agents in each execution tier.

    Tiers from most active to least:
    - PLAYER_PARTY: Always runs (party members)
    - ACTIVE: Always runs (in dialogue/interaction)
    - NEARBY: Every 3 ticks (within perception range)
    - SAME_SETTLEMENT: Every 10 ticks (same settlement)
    - SAME_REGION: Every 50 ticks (same region)
    - BACKGROUND: Every 200 ticks (distant)
    - DORMANT: Event-driven only
    """
    try:
        distribution = await game_service.get_tier_distribution(str(session_id))
        return {"session_id": str(session_id), "tier_distribution": distribution}
    except Exception as e:
        logger.error(f"Failed to get tier distribution for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/hierarchical-stats")
async def get_hierarchical_stats(session_id: UUID):
    """
    Get detailed hierarchical execution statistics.

    Includes:
    - Tier execution counts
    - Full vs simplified cycle counts
    - Collective agent stats
    """
    try:
        stats = await game_service.get_hierarchical_stats(str(session_id))
        return {"session_id": str(session_id), "hierarchical_stats": stats}
    except Exception as e:
        logger.error(f"Failed to get hierarchical stats for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/player-position")
async def update_player_position(
    session_id: UUID,
    x: int,
    y: int,
    settlement_id: str = None
):
    """
    Update player position for tier recalculation.

    This triggers agent tier reassignment based on the new player location.
    """
    try:
        success = await game_service.update_player_position(
            str(session_id), x, y, settlement_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "updated", "position": {"x": x, "y": y}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update player position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/agents/{agent_id}/interaction/start")
async def start_agent_interaction(session_id: UUID, agent_id: UUID):
    """
    Mark an agent as in active interaction with the player.

    This promotes the agent to ACTIVE tier for immediate response.
    """
    try:
        success = await game_service.start_interaction(str(session_id), str(agent_id))
        return {"status": "interaction_started" if success else "orchestrator_unavailable"}
    except Exception as e:
        logger.error(f"Failed to start interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/agents/{agent_id}/interaction/end")
async def end_agent_interaction(session_id: UUID, agent_id: UUID):
    """
    End active interaction with an agent.

    The agent will return to normal tier classification.
    """
    try:
        success = await game_service.end_interaction(str(session_id), str(agent_id))
        return {"status": "interaction_ended" if success else "orchestrator_unavailable"}
    except Exception as e:
        logger.error(f"Failed to end interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/agents/{agent_id}/tier")
async def get_agent_tier(session_id: UUID, agent_id: UUID):
    """Get the current execution tier of a specific agent."""
    try:
        tier = await game_service.get_agent_tier(str(session_id), str(agent_id))
        return {
            "agent_id": str(agent_id),
            "tier": tier or "UNKNOWN",
            "is_active": tier in ["PLAYER_PARTY", "ACTIVE", "NEARBY"] if tier else False
        }
    except Exception as e:
        logger.error(f"Failed to get agent tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/collective-stats")
async def get_collective_stats(session_id: UUID):
    """
    Get statistics about collective agents (settlements, kingdoms).

    Includes population, tracked residents, territory size, etc.
    """
    try:
        stats = await game_service.get_collective_stats(str(session_id))
        return {"session_id": str(session_id), "collective_stats": stats}
    except Exception as e:
        logger.error(f"Failed to get collective stats for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
