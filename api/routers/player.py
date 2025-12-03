"""
Player API Router
Endpoints for player character actions and state.
"""

from fastapi import APIRouter, HTTPException
from uuid import UUID
import logging

from api.models.requests import (
    CreatePlayerRequest,
    PlayerMoveRequest,
    PlayerInteractRequest,
    PlayerSpeakRequest
)
from api.models.responses import PlayerResponse, InteractionResponse
from api.services.player_service import PlayerService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
player_service = PlayerService()


@router.post("/create", response_model=PlayerResponse)
async def create_player(request: CreatePlayerRequest):
    """
    Create a new player character for a game session.
    """
    try:
        player = await player_service.create_player(
            session_id=request.session_id,
            name=request.name,
            x=request.spawn_x,
            y=request.spawn_y,
            stats=request.stats,
            appearance=request.appearance
        )
        logger.info(f"Created player {player.id} in session {request.session_id}")
        return player
    except Exception as e:
        logger.error(f"Failed to create player: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: UUID, session_id: UUID):
    """Get player character details."""
    try:
        player = await player_service.get_player(
            session_id=str(session_id),
            player_id=str(player_id)
        )
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")
        return player
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get player {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move")
async def move_player(request: PlayerMoveRequest):
    """
    Move the player to a new location.

    Can specify:
    - target_x, target_y: Move to absolute position
    - direction: Move one tile in direction (north, south, east, west)
    """
    try:
        result = await player_service.move(
            session_id=request.session_id,
            player_id=request.player_id,
            target_x=request.target_x,
            target_y=request.target_y,
            direction=request.direction
        )
        return result
    except Exception as e:
        logger.error(f"Failed to move player: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interact", response_model=InteractionResponse)
async def interact(request: PlayerInteractRequest):
    """
    Player interacts with something in the world.

    Target types: agent, object, tile, settlement
    Actions vary by target type:
    - agent: talk, trade, attack, follow
    - object: use, examine, pickup
    - tile: examine, gather
    - settlement: enter, examine
    """
    try:
        result = await player_service.interact(
            session_id=request.session_id,
            player_id=request.player_id,
            target_type=request.target_type,
            target_id=request.target_id,
            action=request.action,
            parameters=request.parameters
        )
        return result
    except Exception as e:
        logger.error(f"Interaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speak")
async def speak(request: PlayerSpeakRequest):
    """
    Player speaks (broadcasts message).

    Channels:
    - party: Only party members hear
    - local: Nearby agents hear
    - global: All connected players hear (for multiplayer)
    """
    try:
        result = await player_service.speak(
            session_id=request.session_id,
            player_id=request.player_id,
            message=request.message,
            channel=request.channel
        )
        return result
    except Exception as e:
        logger.error(f"Speak failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{player_id}/inventory")
async def get_inventory(player_id: UUID, session_id: UUID):
    """Get player's inventory."""
    try:
        inventory = await player_service.get_inventory(
            session_id=str(session_id),
            player_id=str(player_id)
        )
        return {"player_id": str(player_id), "inventory": inventory}
    except Exception as e:
        logger.error(f"Failed to get inventory for {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{player_id}/stats")
async def get_stats(player_id: UUID, session_id: UUID):
    """Get player's stats and vitals."""
    try:
        stats = await player_service.get_stats(
            session_id=str(session_id),
            player_id=str(player_id)
        )
        return {"player_id": str(player_id), "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get stats for {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{player_id}/observations")
async def get_observations(player_id: UUID, session_id: UUID):
    """
    Get what the player can currently observe.

    Returns nearby agents, objects, tile info, and any notable events.
    """
    try:
        observations = await player_service.get_observations(
            session_id=str(session_id),
            player_id=str(player_id)
        )
        return observations
    except Exception as e:
        logger.error(f"Failed to get observations for {player_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
