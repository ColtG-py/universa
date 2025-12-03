"""
Agent Service
Handles agent retrieval, dialogue, and commands.
"""

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from uuid import uuid4
from datetime import datetime

from api.models.responses import (
    AgentResponse,
    AgentListResponse,
    DialogueResponse,
    AgentTier
)

if TYPE_CHECKING:
    from api.services.game_service import GameService

logger = logging.getLogger(__name__)


class AgentService:
    """
    Service for agent interactions.

    Integrates with:
    - GameService for session-based agent storage
    - agents/ framework for cognition
    - Ollama for LLM reasoning
    - Memory system for retrieval
    """

    def __init__(self):
        # Reference to game service (set via set_game_service)
        self._game_service: Optional["GameService"] = None
        # Active conversations
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def set_game_service(self, game_service: "GameService"):
        """Set the game service reference for agent data access."""
        self._game_service = game_service
        logger.info("AgentService connected to GameService")

    def _get_session_agents(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Get agents dict from game service session."""
        if not self._game_service:
            logger.warning("GameService not connected to AgentService")
            return {}
        session = self._game_service._sessions.get(session_id)
        if not session:
            return {}
        return session.get("agents", {})

    async def get_agents_near(
        self,
        session_id: str,
        x: int,
        y: int,
        radius: int = 10,
        limit: int = 50
    ) -> List[AgentResponse]:
        """Get agents near a location."""
        session_agents = self._get_session_agents(session_id)

        nearby = []
        for agent_data in session_agents.values():
            ax, ay = agent_data.get("x", 0), agent_data.get("y", 0)
            dist = ((ax - x) ** 2 + (ay - y) ** 2) ** 0.5
            if dist <= radius:
                nearby.append(self._to_response(agent_data))

        return nearby[:limit]

    async def get_all_agents(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[AgentResponse]:
        """Get all agents in a session."""
        session_agents = self._get_session_agents(session_id)
        agents = [self._to_response(a) for a in list(session_agents.values())[:limit]]
        return agents

    async def get_agent(self, session_id: str, agent_id: str) -> Optional[AgentResponse]:
        """Get agent by ID."""
        session_agents = self._get_session_agents(session_id)
        agent_data = session_agents.get(agent_id)
        if not agent_data:
            return None
        return self._to_response(agent_data)

    def _to_response(self, agent_data: Dict[str, Any]) -> AgentResponse:
        """Convert internal agent data to response model."""
        return AgentResponse(
            id=agent_data["id"],
            name=agent_data.get("name", "Unknown"),
            x=agent_data.get("x", 0),
            y=agent_data.get("y", 0),
            agent_type=agent_data.get("agent_type", "individual"),
            role=agent_data.get("role"),
            health=agent_data.get("health", 1.0),
            energy=agent_data.get("energy", 1.0),
            tier=AgentTier(agent_data.get("tier", "background")),
            current_action=agent_data.get("current_action"),
            current_thought=agent_data.get("current_thought"),
            settlement_id=agent_data.get("settlement_id"),
            faction_id=agent_data.get("faction_id"),
            stats=agent_data.get("stats", {}),
            traits=agent_data.get("traits", [])
        )

    async def dialogue(
        self,
        session_id: str,
        agent_id: str,
        player_message: str,
        conversation_id: Optional[str] = None
    ) -> Optional[DialogueResponse]:
        """
        Have a dialogue with an agent.

        This triggers the full agent cognition cycle:
        1. Perceive the player's message
        2. Retrieve relevant memories
        3. Generate response using LLM
        4. Store new memory
        """
        session_agents = self._get_session_agents(session_id)
        agent_data = session_agents.get(agent_id)
        if not agent_data:
            return None

        # Get or create conversation
        if not conversation_id:
            conversation_id = str(uuid4())
            self._conversations[conversation_id] = {
                "id": conversation_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "turns": [],
                "created_at": datetime.utcnow()
            }

        conversation = self._conversations.get(conversation_id, {})
        turn_number = len(conversation.get("turns", [])) + 1

        # TODO: Integrate with actual agent cognition system
        # For now, generate a stub response
        agent_response = f"*{agent_data.get('name', 'The agent')} considers your words carefully.*\n\n\"Interesting... I'll have to think about that.\""

        # Store turn
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["turns"].append({
                "turn": turn_number,
                "player_message": player_message,
                "agent_response": agent_response,
                "timestamp": datetime.utcnow()
            })

        return DialogueResponse(
            agent_id=agent_id,
            agent_name=agent_data.get("name", "Unknown"),
            message=agent_response,
            emotion="thoughtful",
            conversation_id=conversation_id,
            turn_number=turn_number,
            memories_retrieved=None,
            reasoning=None
        )

    async def command(
        self,
        session_id: str,
        agent_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Issue a command to an agent.

        Commands:
        - follow: Follow the player
        - wait: Stay at current location
        - move_to: Move to coordinates
        - attack: Attack a target
        - defend: Take defensive stance
        """
        session_agents = self._get_session_agents(session_id)
        agent_data = session_agents.get(agent_id)
        if not agent_data:
            return {"success": False, "error": "Agent not found"}

        # TODO: Implement actual command processing
        # For now, just update the agent's current action
        agent_data["current_action"] = command
        agent_data["current_thought"] = f"Executing command: {command}"

        return {
            "success": True,
            "agent_id": agent_id,
            "command": command,
            "message": f"{agent_data.get('name', 'Agent')} acknowledges the command."
        }

    async def get_agents_by_settlement(
        self,
        session_id: str,
        settlement_id: str
    ) -> List[AgentResponse]:
        """Get all agents in a settlement."""
        session_agents = self._get_session_agents(session_id)

        return [
            self._to_response(a)
            for a in session_agents.values()
            if a.get("settlement_id") == settlement_id
        ]

    async def get_agents_by_faction(
        self,
        session_id: str,
        faction_id: str
    ) -> List[AgentResponse]:
        """Get all agents in a faction."""
        session_agents = self._get_session_agents(session_id)

        return [
            self._to_response(a)
            for a in session_agents.values()
            if a.get("faction_id") == faction_id
        ]

    async def spawn_agent(
        self,
        session_id: str,
        name: str,
        x: int,
        y: int,
        agent_type: str = "individual",
        role: Optional[str] = None,
        traits: Optional[List[str]] = None,
        settlement_id: Optional[str] = None,
        faction_id: Optional[str] = None
    ) -> Optional[AgentResponse]:
        """Spawn a new agent in the session via GameService."""
        if not self._game_service:
            logger.warning("Cannot spawn agent: GameService not connected")
            return None

        agent = await self._game_service.spawn_agent(
            session_id=session_id,
            name=name,
            x=x,
            y=y,
            agent_type=agent_type,
            role=role,
        )
        if agent:
            return self._to_response(agent)
        return None

    async def get_agent_status(
        self,
        session_id: str,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get agent status."""
        session_agents = self._get_session_agents(session_id)
        agent_data = session_agents.get(agent_id)
        if not agent_data:
            return None
        return {
            "id": agent_data.get("id"),
            "name": agent_data.get("name"),
            "status": agent_data.get("status", "active"),
            "current_action": agent_data.get("current_action"),
            "current_thought": agent_data.get("current_thought"),
            "health": agent_data.get("health", 1.0),
            "energy": agent_data.get("energy", 1.0),
        }

    async def get_inventory(
        self,
        session_id: str,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get agent inventory."""
        session_agents = self._get_session_agents(session_id)
        agent_data = session_agents.get(agent_id)
        if not agent_data:
            return []
        return agent_data.get("inventory", [])

    async def execute_command(
        self,
        session_id: str,
        agent_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a command on an agent (alias for command method)."""
        return await self.command(session_id, agent_id, command, parameters)
