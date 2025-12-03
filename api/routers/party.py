"""
Party Management API Router
Endpoints for managing the player's party of agents.
"""

from fastapi import APIRouter, HTTPException
from uuid import UUID
import logging

from api.models.requests import (
    CreatePartyRequest,
    PartyCommandRequest,
    PartyDialogueRequest
)
from api.models.responses import PartyResponse, PartyMemberResponse
from api.services.party_service import PartyService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
party_service = PartyService()


@router.post("/create", response_model=PartyResponse)
async def create_party(request: CreatePartyRequest):
    """
    Create a party with AI-controlled companions.

    Party members are generated based on:
    - size: Number of companions (1-6)
    - roles: Optional role assignments (warrior, mage, healer, etc.)
    - If roles not specified, balanced party is generated
    """
    try:
        party = await party_service.create_party(
            session_id=request.session_id,
            player_id=request.player_id,
            size=request.size,
            roles=request.roles,
            names=request.names
        )
        logger.info(f"Created party with {len(party.members)} members")
        return party
    except Exception as e:
        logger.error(f"Failed to create party: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{party_id}", response_model=PartyResponse)
async def get_party(party_id: UUID, session_id: UUID):
    """Get party details and member list."""
    try:
        party = await party_service.get_party(
            session_id=str(session_id),
            party_id=str(party_id)
        )
        if not party:
            raise HTTPException(status_code=404, detail="Party not found")
        return party
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get party {party_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{party_id}/members")
async def get_party_members(party_id: UUID, session_id: UUID):
    """Get detailed info on all party members."""
    try:
        members = await party_service.get_members(
            session_id=str(session_id),
            party_id=str(party_id)
        )
        return {"party_id": str(party_id), "members": members}
    except Exception as e:
        logger.error(f"Failed to get members for party {party_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{party_id}/members/{agent_id}", response_model=PartyMemberResponse)
async def get_party_member(party_id: UUID, agent_id: UUID, session_id: UUID):
    """Get detailed info on a specific party member."""
    try:
        member = await party_service.get_member(
            session_id=str(session_id),
            party_id=str(party_id),
            agent_id=str(agent_id)
        )
        if not member:
            raise HTTPException(status_code=404, detail="Party member not found")
        return member
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get party member {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{party_id}/command")
async def party_command(party_id: UUID, request: PartyCommandRequest):
    """
    Issue a command to the entire party.

    Commands:
    - follow: Party follows the player
    - wait: Party waits at current location
    - spread: Party spreads out around player
    - guard: Party takes defensive positions
    - rest: Party rests to recover
    """
    try:
        result = await party_service.execute_command(
            session_id=request.session_id,
            party_id=str(party_id),
            command=request.command,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Party command failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{party_id}/agents/{agent_id}/dialogue")
async def dialogue_with_party_member(
    party_id: UUID,
    agent_id: UUID,
    request: PartyDialogueRequest
):
    """
    Talk to a specific party member.

    Party members have more context about the player
    and shared adventures.
    """
    try:
        response = await party_service.dialogue_with_member(
            session_id=request.session_id,
            party_id=str(party_id),
            agent_id=str(agent_id),
            message=request.message,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        logger.error(f"Party dialogue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{party_id}/add/{agent_id}")
async def add_to_party(party_id: UUID, agent_id: UUID, session_id: UUID):
    """
    Add an agent to the party.

    The agent must agree to join (based on relationship and disposition).
    """
    try:
        result = await party_service.request_join(
            session_id=str(session_id),
            party_id=str(party_id),
            agent_id=str(agent_id)
        )
        return result
    except Exception as e:
        logger.error(f"Failed to add agent {agent_id} to party: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{party_id}/remove/{agent_id}")
async def remove_from_party(party_id: UUID, agent_id: UUID, session_id: UUID):
    """Remove an agent from the party."""
    try:
        result = await party_service.remove_member(
            session_id=str(session_id),
            party_id=str(party_id),
            agent_id=str(agent_id)
        )
        return result
    except Exception as e:
        logger.error(f"Failed to remove agent {agent_id} from party: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{party_id}/chat")
async def get_party_chat(party_id: UUID, session_id: UUID, limit: int = 50):
    """Get recent party chat history."""
    try:
        chat = await party_service.get_chat_history(
            session_id=str(session_id),
            party_id=str(party_id),
            limit=limit
        )
        return {"party_id": str(party_id), "messages": chat}
    except Exception as e:
        logger.error(f"Failed to get party chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
