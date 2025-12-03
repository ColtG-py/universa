"""
Player Service
Handles player character actions and state.
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

from api.models.responses import (
    PlayerResponse,
    InteractionResponse,
    ObservationResponse,
    TileData
)
from api.models.requests import Direction, InteractionAction

logger = logging.getLogger(__name__)


class PlayerService:
    """
    Service for player character management.

    Handles:
    - Player creation and state
    - Movement and pathfinding
    - Interactions with world entities
    - Inventory management
    """

    def __init__(self):
        # Players by session
        self._players: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def create_player(
        self,
        session_id: str,
        name: str,
        x: int,
        y: int,
        stats: Optional[Dict[str, int]] = None,
        appearance: Optional[Dict[str, Any]] = None
    ) -> PlayerResponse:
        """Create a new player character."""
        player_id = str(uuid4())

        # Default stats
        default_stats = {
            "strength": 10,
            "dexterity": 10,
            "constitution": 10,
            "intelligence": 10,
            "wisdom": 10,
            "charisma": 10
        }
        if stats:
            default_stats.update(stats)

        player_data = {
            "id": player_id,
            "name": name,
            "x": x,
            "y": y,
            "health": 100.0,
            "max_health": 100.0,
            "stamina": 100.0,
            "max_stamina": 100.0,
            "stats": default_stats,
            "appearance": appearance or {},
            "gold": 50,
            "inventory": [],
            "inventory_slots_total": 20,
            "created_at": datetime.utcnow()
        }

        if session_id not in self._players:
            self._players[session_id] = {}
        self._players[session_id][player_id] = player_data

        logger.info(f"Created player {player_id} ({name}) at ({x}, {y})")

        return self._to_response(player_data)

    async def get_player(self, session_id: str, player_id: str) -> Optional[PlayerResponse]:
        """Get player by ID."""
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return None
        return self._to_response(player_data)

    def _to_response(self, player_data: Dict[str, Any]) -> PlayerResponse:
        """Convert internal player data to response model."""
        return PlayerResponse(
            id=player_data["id"],
            name=player_data["name"],
            x=player_data["x"],
            y=player_data["y"],
            health=player_data.get("health", 100.0),
            max_health=player_data.get("max_health", 100.0),
            stamina=player_data.get("stamina", 100.0),
            max_stamina=player_data.get("max_stamina", 100.0),
            stats=player_data.get("stats", {}),
            gold=player_data.get("gold", 0),
            inventory_slots_used=len(player_data.get("inventory", [])),
            inventory_slots_total=player_data.get("inventory_slots_total", 20)
        )

    async def move(
        self,
        session_id: str,
        player_id: str,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
        direction: Optional[Direction] = None
    ) -> Dict[str, Any]:
        """
        Move the player.

        Can specify:
        - target_x, target_y: Absolute position (pathfinding)
        - direction: Single tile movement
        """
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return {"success": False, "error": "Player not found"}

        old_x, old_y = player_data["x"], player_data["y"]

        if direction:
            # Direction-based movement
            dx, dy = self._direction_to_delta(direction)
            player_data["x"] += dx
            player_data["y"] += dy
        elif target_x is not None and target_y is not None:
            # TODO: Implement pathfinding for longer distances
            # For now, just teleport
            player_data["x"] = target_x
            player_data["y"] = target_y

        # Deduct stamina
        distance = abs(player_data["x"] - old_x) + abs(player_data["y"] - old_y)
        player_data["stamina"] = max(0, player_data["stamina"] - distance * 0.5)

        return {
            "success": True,
            "player_id": player_id,
            "old_position": {"x": old_x, "y": old_y},
            "new_position": {"x": player_data["x"], "y": player_data["y"]},
            "stamina": player_data["stamina"]
        }

    def _direction_to_delta(self, direction: Direction) -> tuple[int, int]:
        """Convert direction to coordinate delta."""
        deltas = {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
            Direction.NORTHEAST: (1, -1),
            Direction.NORTHWEST: (-1, -1),
            Direction.SOUTHEAST: (1, 1),
            Direction.SOUTHWEST: (-1, 1),
        }
        return deltas.get(direction, (0, 0))

    async def interact(
        self,
        session_id: str,
        player_id: str,
        target_type: str,
        target_id: str,
        action: InteractionAction,
        parameters: Optional[Dict[str, Any]] = None
    ) -> InteractionResponse:
        """
        Player interacts with something in the world.

        Target types: agent, object, tile, settlement
        """
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return InteractionResponse(
                success=False,
                action=action.value,
                target_type=target_type,
                target_id=target_id,
                message="Player not found"
            )

        # TODO: Implement actual interaction logic based on type and action
        result = {}

        if target_type == "agent":
            if action == InteractionAction.TALK:
                result["initiated_dialogue"] = True
            elif action == InteractionAction.TRADE:
                result["opened_trade"] = True
            elif action == InteractionAction.ATTACK:
                result["initiated_combat"] = True
        elif target_type == "object":
            if action == InteractionAction.PICKUP:
                result["item_picked_up"] = True
            elif action == InteractionAction.USE:
                result["item_used"] = True
        elif target_type == "tile":
            if action == InteractionAction.GATHER:
                result["resources_gathered"] = []

        return InteractionResponse(
            success=True,
            action=action.value,
            target_type=target_type,
            target_id=target_id,
            result=result,
            message=f"Interacted with {target_type} using {action.value}"
        )

    async def speak(
        self,
        session_id: str,
        player_id: str,
        message: str,
        channel: str
    ) -> Dict[str, Any]:
        """Player speaks (broadcasts message)."""
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return {"success": False, "error": "Player not found"}

        # TODO: Broadcast to appropriate channel via WebSocket
        return {
            "success": True,
            "player_id": player_id,
            "player_name": player_data["name"],
            "message": message,
            "channel": channel
        }

    async def get_inventory(
        self,
        session_id: str,
        player_id: str
    ) -> List[Dict[str, Any]]:
        """Get player's inventory."""
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return []
        return player_data.get("inventory", [])

    async def get_stats(
        self,
        session_id: str,
        player_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get player's stats and vitals."""
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return None

        return {
            "base_stats": player_data.get("stats", {}),
            "health": player_data.get("health", 100),
            "max_health": player_data.get("max_health", 100),
            "stamina": player_data.get("stamina", 100),
            "max_stamina": player_data.get("max_stamina", 100),
            "gold": player_data.get("gold", 0)
        }

    async def get_observations(
        self,
        session_id: str,
        player_id: str
    ) -> Optional[ObservationResponse]:
        """Get what the player can currently observe."""
        session_players = self._players.get(session_id, {})
        player_data = session_players.get(player_id)
        if not player_data:
            return None

        x, y = player_data["x"], player_data["y"]

        # TODO: Get actual tile and nearby entity data
        current_tile = TileData(
            x=x,
            y=y,
            biome="grassland",
            elevation=0.5,
            temperature=0.5,
            moisture=0.5
        )

        return ObservationResponse(
            player_id=player_id,
            x=x,
            y=y,
            current_tile=current_tile,
            nearby_agents=[],
            nearby_objects=[],
            nearby_settlements=[],
            recent_events=[]
        )
