"""
Agent API Router
Endpoints for interacting with NPCs and agents.
"""

from fastapi import APIRouter, HTTPException, Query
from uuid import UUID
from typing import Optional
import logging

from api.models.requests import DialogueRequest, AgentCommandRequest
from api.models.responses import AgentResponse, AgentListResponse, DialogueResponse
from api.services.agent_service import AgentService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
agent_service = AgentService()


@router.get("", response_model=AgentListResponse)
async def list_agents(
    session_id: UUID = Query(..., description="Game session ID"),
    x: Optional[int] = Query(None, description="Center X coordinate"),
    y: Optional[int] = Query(None, description="Center Y coordinate"),
    radius: int = Query(10, description="Search radius in tiles"),
    limit: int = Query(50, description="Maximum agents to return")
):
    """
    List agents in a session, optionally filtered by location.

    If x and y are provided, returns agents within the specified radius.
    """
    try:
        if x is not None and y is not None:
            agents = await agent_service.get_agents_near(
                session_id=str(session_id),
                x=x,
                y=y,
                radius=radius,
                limit=limit
            )
        else:
            agents = await agent_service.get_all_agents(
                session_id=str(session_id),
                limit=limit
            )
        return AgentListResponse(agents=agents, total=len(agents))
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nearby")
async def get_nearby_agents(
    session_id: UUID,
    x: int,
    y: int,
    radius: int = 15
):
    """Get agents near a specific location."""
    try:
        agents = await agent_service.get_agents_near(
            session_id=str(session_id),
            x=x,
            y=y,
            radius=radius
        )
        # Convert AgentResponse objects to dicts for JSON serialization
        agent_dicts = [a.model_dump() for a in agents] if agents else []
        return {"agents": agent_dicts, "center": {"x": x, "y": y}, "radius": radius}
    except Exception as e:
        logger.error(f"Failed to get nearby agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: UUID, session_id: UUID):
    """Get detailed information about a specific agent."""
    try:
        agent = await agent_service.get_agent(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/dialogue", response_model=DialogueResponse)
async def dialogue_with_agent(
    agent_id: UUID,
    request: DialogueRequest
):
    """
    Initiate or continue dialogue with an agent.

    If no conversation_id is provided, starts a new conversation.
    """
    try:
        response = await agent_service.dialogue(
            session_id=request.session_id,
            agent_id=str(agent_id),
            player_message=request.message,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        logger.error(f"Dialogue failed with agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/command")
async def command_agent(
    agent_id: UUID,
    request: AgentCommandRequest
):
    """
    Issue a command to an agent (for party members or controlled agents).

    Commands: follow, wait, move_to, attack, defend, etc.
    """
    try:
        result = await agent_service.execute_command(
            session_id=request.session_id,
            agent_id=str(agent_id),
            command=request.command,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Command failed for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: UUID, session_id: UUID):
    """Get the current status and activity of an agent."""
    try:
        status = await agent_service.get_agent_status(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        if not status:
            raise HTTPException(status_code=404, detail="Agent not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/inventory")
async def get_agent_inventory(agent_id: UUID, session_id: UUID):
    """Get an agent's inventory (for trading)."""
    try:
        inventory = await agent_service.get_inventory(
            session_id=str(session_id),
            agent_id=str(agent_id)
        )
        return {"agent_id": str(agent_id), "inventory": inventory}
    except Exception as e:
        logger.error(f"Failed to get inventory for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
