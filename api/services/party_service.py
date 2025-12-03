"""
Party Service
Handles party management and member interactions.
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

from api.models.responses import PartyResponse, PartyMemberResponse
from api.models.requests import PartyRole

logger = logging.getLogger(__name__)


class PartyService:
    """
    Service for party management.

    Handles:
    - Party creation with AI companions
    - Party commands and formations
    - Member dialogue and relationships
    """

    def __init__(self):
        # Parties by session
        self._parties: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Chat history by party
        self._chat_history: Dict[str, List[Dict[str, Any]]] = {}

    async def create_party(
        self,
        session_id: str,
        player_id: str,
        size: int = 3,
        roles: Optional[List[PartyRole]] = None,
        names: Optional[List[str]] = None
    ) -> PartyResponse:
        """
        Create a party with AI-controlled companions.

        Members are generated based on:
        - size: Number of companions
        - roles: Role assignments (warrior, mage, etc.)
        - names: Custom names (or auto-generated)
        """
        party_id = str(uuid4())

        # Default roles if not specified
        default_roles = [
            PartyRole.WARRIOR,
            PartyRole.MAGE,
            PartyRole.HEALER,
            PartyRole.ROGUE,
            PartyRole.RANGER,
            PartyRole.COMPANION
        ]
        actual_roles = roles[:size] if roles else default_roles[:size]

        # Default names if not specified
        default_names = [
            "Aldric", "Elena", "Theron", "Lyra", "Kael", "Mira"
        ]
        actual_names = names[:size] if names else default_names[:size]

        members = []
        for i in range(size):
            member = {
                "id": str(uuid4()),
                "name": actual_names[i] if i < len(actual_names) else f"Companion {i+1}",
                "role": actual_roles[i].value if i < len(actual_roles) else "companion",
                "x": 0,  # Will be positioned relative to player
                "y": 0,
                "distance_to_player": 1.0 + i * 0.5,
                "health": 1.0,
                "energy": 1.0,
                "morale": 1.0,
                "current_action": None,
                "following": True,
                "loyalty": 0.6,
                "trust": 0.5,
                "affection": 0.5,
                "joined_at": datetime.utcnow()
            }
            members.append(member)

        party_data = {
            "id": party_id,
            "session_id": session_id,
            "leader_id": player_id,
            "members": members,
            "formation": "follow",
            "created_at": datetime.utcnow()
        }

        if session_id not in self._parties:
            self._parties[session_id] = {}
        self._parties[session_id][party_id] = party_data

        # Initialize chat history
        self._chat_history[party_id] = []

        logger.info(f"Created party {party_id} with {size} members")

        return self._to_response(party_data)

    async def get_party(self, session_id: str, party_id: str) -> Optional[PartyResponse]:
        """Get party by ID."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return None
        return self._to_response(party_data)

    async def get_members(self, session_id: str, party_id: str) -> List[PartyMemberResponse]:
        """Get detailed info on all party members."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return []

        return [self._member_to_response(m) for m in party_data["members"]]

    async def get_member(
        self,
        session_id: str,
        party_id: str,
        agent_id: str
    ) -> Optional[PartyMemberResponse]:
        """Get a specific party member."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return None

        for member in party_data["members"]:
            if member["id"] == agent_id:
                return self._member_to_response(member)
        return None

    def _to_response(self, party_data: Dict[str, Any]) -> PartyResponse:
        """Convert internal party data to response model."""
        members = [self._member_to_response(m) for m in party_data["members"]]

        avg_morale = sum(m.morale for m in members) / len(members) if members else 1.0
        center_x = sum(m.x for m in members) / len(members) if members else 0
        center_y = sum(m.y for m in members) / len(members) if members else 0

        return PartyResponse(
            id=party_data["id"],
            leader_id=party_data["leader_id"],
            members=members,
            formation=party_data.get("formation", "follow"),
            average_morale=avg_morale,
            center_x=center_x,
            center_y=center_y
        )

    def _member_to_response(self, member: Dict[str, Any]) -> PartyMemberResponse:
        """Convert internal member data to response model."""
        return PartyMemberResponse(
            id=member["id"],
            name=member["name"],
            role=member["role"],
            x=member.get("x", 0),
            y=member.get("y", 0),
            distance_to_player=member.get("distance_to_player", 0),
            health=member.get("health", 1.0),
            energy=member.get("energy", 1.0),
            morale=member.get("morale", 1.0),
            current_action=member.get("current_action"),
            following=member.get("following", True),
            loyalty=member.get("loyalty", 0.5),
            trust=member.get("trust", 0.5),
            affection=member.get("affection", 0.5)
        )

    async def execute_command(
        self,
        session_id: str,
        party_id: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a party command.

        Commands:
        - follow: Party follows the player
        - wait: Party waits at current location
        - spread: Party spreads out around player
        - guard: Party takes defensive positions
        - rest: Party rests to recover
        """
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return {"success": False, "error": "Party not found"}

        party_data["formation"] = command

        # Update each member's state based on command
        for member in party_data["members"]:
            if command == "follow":
                member["following"] = True
                member["current_action"] = "Following leader"
            elif command == "wait":
                member["following"] = False
                member["current_action"] = "Waiting"
            elif command == "spread":
                member["following"] = False
                member["current_action"] = "Spreading out"
            elif command == "guard":
                member["following"] = False
                member["current_action"] = "Guarding"
            elif command == "rest":
                member["following"] = False
                member["current_action"] = "Resting"
                member["energy"] = min(1.0, member["energy"] + 0.1)

        return {
            "success": True,
            "party_id": party_id,
            "command": command,
            "message": f"Party is now in {command} formation."
        }

    async def dialogue_with_member(
        self,
        session_id: str,
        party_id: str,
        agent_id: str,
        message: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Talk to a specific party member."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return {"success": False, "error": "Party not found"}

        member = None
        for m in party_data["members"]:
            if m["id"] == agent_id:
                member = m
                break

        if not member:
            return {"success": False, "error": "Party member not found"}

        # TODO: Integrate with agent cognition system
        # For now, generate a contextual response based on role
        role_responses = {
            "warrior": "I'll protect you with my life. What do you need?",
            "mage": "The arcane energies here are fascinating. What troubles you?",
            "healer": "Are you injured? I can help with that.",
            "rogue": "Keep your voice down. What's the plan?",
            "ranger": "I've been watching our surroundings. All clear for now.",
            "companion": "I'm here for you. What would you like to discuss?"
        }

        response = role_responses.get(
            member["role"],
            "Yes? I'm listening."
        )

        # Store in chat history
        chat_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "speaker_id": agent_id,
            "speaker_name": member["name"],
            "player_message": message,
            "response": response
        }
        if party_id not in self._chat_history:
            self._chat_history[party_id] = []
        self._chat_history[party_id].append(chat_entry)

        return {
            "success": True,
            "agent_id": agent_id,
            "agent_name": member["name"],
            "response": response,
            "emotion": "attentive"
        }

    async def request_join(
        self,
        session_id: str,
        party_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """Request an agent to join the party."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return {"success": False, "error": "Party not found"}

        # Check party size limit
        if len(party_data["members"]) >= 6:
            return {
                "success": False,
                "error": "Party is full",
                "message": "Cannot add more than 6 members to the party."
            }

        # TODO: Check agent's disposition and relationship
        # For now, always accept
        new_member = {
            "id": agent_id,
            "name": f"Recruit {len(party_data['members']) + 1}",
            "role": "companion",
            "x": 0,
            "y": 0,
            "distance_to_player": 2.0,
            "health": 1.0,
            "energy": 1.0,
            "morale": 0.8,
            "following": True,
            "loyalty": 0.3,
            "trust": 0.3,
            "affection": 0.3,
            "joined_at": datetime.utcnow()
        }

        party_data["members"].append(new_member)

        return {
            "success": True,
            "agent_id": agent_id,
            "message": f"{new_member['name']} has joined the party!"
        }

    async def remove_member(
        self,
        session_id: str,
        party_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """Remove an agent from the party."""
        session_parties = self._parties.get(session_id, {})
        party_data = session_parties.get(party_id)
        if not party_data:
            return {"success": False, "error": "Party not found"}

        for i, member in enumerate(party_data["members"]):
            if member["id"] == agent_id:
                removed = party_data["members"].pop(i)
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "message": f"{removed['name']} has left the party."
                }

        return {"success": False, "error": "Member not found in party"}

    async def get_chat_history(
        self,
        session_id: str,
        party_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent party chat history."""
        history = self._chat_history.get(party_id, [])
        return history[-limit:]
