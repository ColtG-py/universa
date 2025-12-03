"""
Dialogue API Router
Endpoints for managing conversations between players and agents.
"""

from fastapi import APIRouter, HTTPException, Query
from uuid import UUID
from typing import Optional
import logging
from pydantic import BaseModel, Field

from api.services.dialogue_service import DialogueService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance
dialogue_service = DialogueService()


# =============================================================================
# Request/Response Models
# =============================================================================

class StartConversationRequest(BaseModel):
    """Request to start a conversation."""
    session_id: str = Field(..., description="Game session ID")
    player_id: str = Field(..., description="Player's ID")
    player_name: str = Field(..., description="Player's display name")
    agent_id: str = Field(..., description="Agent to talk to")
    agent_name: str = Field(..., description="Agent's display name")
    player_x: int = Field(..., description="Player's X position")
    player_y: int = Field(..., description="Player's Y position")
    opening_message: Optional[str] = Field(None, description="Optional opening message from player")


class ContinueConversationRequest(BaseModel):
    """Request to continue a conversation."""
    conversation_id: str = Field(..., description="Conversation ID")
    message: str = Field(..., description="Message content")
    speaker_id: str = Field(..., description="ID of the speaker")
    speaker_name: str = Field(..., description="Name of the speaker")


class EndConversationRequest(BaseModel):
    """Request to end a conversation."""
    conversation_id: str = Field(..., description="Conversation ID")
    reason: str = Field("player_ended", description="Reason for ending")


class GetAgentResponseRequest(BaseModel):
    """Request agent's response in conversation."""
    conversation_id: str = Field(..., description="Conversation ID")
    agent_id: str = Field(..., description="Agent's ID")
    agent_name: str = Field(..., description="Agent's name")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start")
async def start_conversation(request: StartConversationRequest):
    """
    Start a new conversation with an agent.

    If opening_message is provided, the player initiates the conversation.
    Otherwise, the agent greets the player first.

    Returns:
        Conversation object with initial turns
    """
    try:
        conversation = await dialogue_service.start_conversation(
            session_id=request.session_id,
            player_id=request.player_id,
            player_name=request.player_name,
            agent_id=request.agent_id,
            agent_name=request.agent_name,
            player_x=request.player_x,
            player_y=request.player_y,
            opening_message=request.opening_message
        )
        return conversation.to_dict()
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue")
async def continue_conversation(request: ContinueConversationRequest):
    """
    Add a message to an ongoing conversation.

    Use this to send player messages. Then call /response to get agent reply.

    Returns:
        The new dialogue turn
    """
    try:
        turn = await dialogue_service.continue_conversation(
            conversation_id=request.conversation_id,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
            message=request.message,
            speaker_type='player'
        )
        if not turn:
            raise HTTPException(status_code=404, detail="Conversation not found or not active")
        return turn.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to continue conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/response")
async def get_agent_response(request: GetAgentResponseRequest):
    """
    Get the agent's response in a conversation.

    Call this after adding a player message with /continue.

    Returns:
        DialogueResponse with agent's reply, emotion, and whether conversation should end
    """
    try:
        response = await dialogue_service.get_agent_response(
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
            agent_name=request.agent_name
        )
        if not response:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return response.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end")
async def end_conversation(request: EndConversationRequest):
    """
    End an active conversation.

    This will:
    - Mark the conversation as ended
    - Create memories for participants
    - Notify game service to update agent tiers

    Returns:
        The ended conversation
    """
    try:
        conversation = await dialogue_service.end_conversation(
            conversation_id=request.conversation_id,
            reason=request.reason
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation by ID.

    Returns the full conversation including all turns.
    """
    conversation = dialogue_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation.to_dict()


@router.get("/agent/{agent_id}/active")
async def get_agent_active_conversation(agent_id: str):
    """
    Get an agent's active conversation, if any.

    Useful for checking if an agent is currently in a conversation.
    """
    conversation = dialogue_service.get_agent_conversation(agent_id)
    if not conversation:
        return {"agent_id": agent_id, "has_active_conversation": False}
    return {
        "agent_id": agent_id,
        "has_active_conversation": True,
        "conversation": conversation.to_dict()
    }


@router.get("/session/{session_id}")
async def get_session_conversations(
    session_id: str,
    active_only: bool = Query(True, description="Only return active conversations")
):
    """
    Get all conversations in a session.

    By default, only returns active conversations.
    """
    conversations = dialogue_service.get_session_conversations(
        session_id=session_id,
        active_only=active_only
    )
    return {
        "session_id": session_id,
        "conversations": [c.to_dict() for c in conversations],
        "total": len(conversations)
    }


# =============================================================================
# Quick Talk Endpoint
# =============================================================================

@router.post("/quick-talk")
async def quick_talk(request: StartConversationRequest):
    """
    Start a conversation and immediately get agent's response.

    Convenience endpoint that combines /start with initial player message.
    Useful for single-exchange interactions.

    Returns:
        Conversation with agent's response included
    """
    try:
        # Start conversation with player's message
        conversation = await dialogue_service.start_conversation(
            session_id=request.session_id,
            player_id=request.player_id,
            player_name=request.player_name,
            agent_id=request.agent_id,
            agent_name=request.agent_name,
            player_x=request.player_x,
            player_y=request.player_y,
            opening_message=request.opening_message
        )

        return {
            "conversation": conversation.to_dict(),
            "agent_response": conversation.turns[-1].to_dict() if conversation.turns else None
        }
    except Exception as e:
        logger.error(f"Quick talk failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility function for wiring services
# =============================================================================

def set_game_service(game_service):
    """Wire up game service dependency."""
    dialogue_service.set_game_service(game_service)


def set_llm_service(llm_service):
    """Wire up LLM service dependency."""
    dialogue_service.set_llm_service(llm_service)
