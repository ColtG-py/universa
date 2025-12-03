"""
Interaction Service
Handles player interactions with world objects, agents, and locations.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InteractionType(str, Enum):
    """Types of interactions."""
    TALK = "talk"
    TRADE = "trade"
    EXAMINE = "examine"
    USE = "use"
    PICKUP = "pickup"
    DROP = "drop"
    ATTACK = "attack"
    FOLLOW = "follow"
    DISMISS = "dismiss"
    ENTER = "enter"
    EXIT = "exit"


class TargetType(str, Enum):
    """Types of interaction targets."""
    AGENT = "agent"
    OBJECT = "object"
    TILE = "tile"
    SETTLEMENT = "settlement"
    BUILDING = "building"
    ITEM = "item"


@dataclass
class InteractionResult:
    """Result of an interaction."""
    success: bool
    interaction_type: str
    target_type: str
    target_id: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    state_changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "interaction_type": self.interaction_type,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "message": self.message,
            "data": self.data,
            "events": self.events,
            "state_changes": self.state_changes
        }


class InteractionService:
    """
    Service for handling player interactions with the world.

    Features:
    - Agent interactions (talk, trade, follow, etc.)
    - Object interactions (examine, use, pickup)
    - Location interactions (enter buildings, explore areas)
    - Settlement interactions (enter, view services)
    """

    def __init__(self):
        self._game_service = None
        self._dialogue_service = None
        self._party_service = None

    def set_game_service(self, game_service):
        """Set the game service reference."""
        self._game_service = game_service

    def set_dialogue_service(self, dialogue_service):
        """Set the dialogue service reference."""
        self._dialogue_service = dialogue_service

    def set_party_service(self, party_service):
        """Set the party service reference."""
        self._party_service = party_service

    async def interact(
        self,
        session_id: str,
        player_id: str,
        player_name: str,
        player_x: int,
        player_y: int,
        target_type: str,
        target_id: str,
        action: str,
        action_data: Optional[Dict[str, Any]] = None
    ) -> InteractionResult:
        """
        Process a player interaction.

        Args:
            session_id: Game session ID
            player_id: Player's ID
            player_name: Player's name
            player_x: Player's X position
            player_y: Player's Y position
            target_type: Type of target (agent, object, tile, etc.)
            target_id: ID of the target
            action: Action to perform
            action_data: Additional action data

        Returns:
            InteractionResult with outcome
        """
        action_data = action_data or {}

        try:
            target = TargetType(target_type)
        except ValueError:
            return InteractionResult(
                success=False,
                interaction_type=action,
                target_type=target_type,
                target_id=target_id,
                message=f"Unknown target type: {target_type}"
            )

        # Route to appropriate handler
        if target == TargetType.AGENT:
            return await self._interact_with_agent(
                session_id, player_id, player_name, player_x, player_y,
                target_id, action, action_data
            )
        elif target == TargetType.OBJECT:
            return await self._interact_with_object(
                session_id, player_id, target_id, action, action_data
            )
        elif target == TargetType.TILE:
            return await self._interact_with_tile(
                session_id, player_id, target_id, action, action_data
            )
        elif target == TargetType.SETTLEMENT:
            return await self._interact_with_settlement(
                session_id, player_id, target_id, action, action_data
            )
        elif target == TargetType.BUILDING:
            return await self._interact_with_building(
                session_id, player_id, target_id, action, action_data
            )
        elif target == TargetType.ITEM:
            return await self._interact_with_item(
                session_id, player_id, target_id, action, action_data
            )
        else:
            return InteractionResult(
                success=False,
                interaction_type=action,
                target_type=target_type,
                target_id=target_id,
                message=f"Unhandled target type: {target_type}"
            )

    # ==========================================================================
    # Agent Interactions
    # ==========================================================================

    async def _interact_with_agent(
        self,
        session_id: str,
        player_id: str,
        player_name: str,
        player_x: int,
        player_y: int,
        agent_id: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with an agent."""

        # Get agent info
        agent_info = await self._get_agent_info(session_id, agent_id)
        if not agent_info:
            return InteractionResult(
                success=False,
                interaction_type=action,
                target_type="agent",
                target_id=agent_id,
                message="Agent not found"
            )

        agent_name = agent_info.get('name', 'Unknown')

        if action == InteractionType.TALK.value:
            return await self._start_dialogue(
                session_id, player_id, player_name, player_x, player_y,
                agent_id, agent_name, action_data.get('message')
            )

        elif action == InteractionType.TRADE.value:
            return await self._initiate_trade(
                session_id, player_id, agent_id, agent_name
            )

        elif action == InteractionType.EXAMINE.value:
            return await self._examine_agent(agent_info)

        elif action == InteractionType.FOLLOW.value:
            return await self._request_follow(
                session_id, player_id, player_name, agent_id, agent_name
            )

        elif action == InteractionType.DISMISS.value:
            return await self._dismiss_from_party(
                session_id, player_id, agent_id, agent_name
            )

        elif action == InteractionType.ATTACK.value:
            return await self._initiate_combat(
                session_id, player_id, agent_id, agent_name
            )

        else:
            return InteractionResult(
                success=False,
                interaction_type=action,
                target_type="agent",
                target_id=agent_id,
                message=f"Unknown action: {action}"
            )

    async def _start_dialogue(
        self,
        session_id: str,
        player_id: str,
        player_name: str,
        player_x: int,
        player_y: int,
        agent_id: str,
        agent_name: str,
        opening_message: Optional[str]
    ) -> InteractionResult:
        """Start a dialogue with an agent."""
        if not self._dialogue_service:
            return InteractionResult(
                success=False,
                interaction_type="talk",
                target_type="agent",
                target_id=agent_id,
                message="Dialogue service not available"
            )

        try:
            conversation = await self._dialogue_service.start_conversation(
                session_id=session_id,
                player_id=player_id,
                player_name=player_name,
                agent_id=agent_id,
                agent_name=agent_name,
                player_x=player_x,
                player_y=player_y,
                opening_message=opening_message
            )

            return InteractionResult(
                success=True,
                interaction_type="talk",
                target_type="agent",
                target_id=agent_id,
                message=f"Started conversation with {agent_name}",
                data={
                    "conversation": conversation.to_dict(),
                    "agent_name": agent_name
                }
            )
        except Exception as e:
            logger.error(f"Failed to start dialogue: {e}")
            return InteractionResult(
                success=False,
                interaction_type="talk",
                target_type="agent",
                target_id=agent_id,
                message=f"Failed to start conversation: {str(e)}"
            )

    async def _initiate_trade(
        self,
        session_id: str,
        player_id: str,
        agent_id: str,
        agent_name: str
    ) -> InteractionResult:
        """Initiate trade with an agent."""
        # TODO: Implement trade system
        return InteractionResult(
            success=True,
            interaction_type="trade",
            target_type="agent",
            target_id=agent_id,
            message=f"{agent_name} is willing to trade",
            data={
                "trade_id": str(uuid4()),
                "agent_name": agent_name,
                "agent_inventory": [],  # Would be populated from agent state
                "trade_available": True
            }
        )

    async def _examine_agent(
        self,
        agent_info: Dict[str, Any]
    ) -> InteractionResult:
        """Examine an agent."""
        description = self._generate_agent_description(agent_info)

        return InteractionResult(
            success=True,
            interaction_type="examine",
            target_type="agent",
            target_id=agent_info.get('id', ''),
            message=description,
            data={
                "agent_info": agent_info
            }
        )

    async def _request_follow(
        self,
        session_id: str,
        player_id: str,
        player_name: str,
        agent_id: str,
        agent_name: str
    ) -> InteractionResult:
        """Request an agent to follow/join party."""
        if self._party_service:
            try:
                result = await self._party_service.request_join(
                    session_id, player_id, player_name, agent_id, agent_name
                )
                return result
            except Exception as e:
                logger.error(f"Failed to request follow: {e}")

        # Fallback without party service
        # Check if game service can add to party
        if self._game_service:
            try:
                await self._game_service.add_to_party(session_id, agent_id)
                return InteractionResult(
                    success=True,
                    interaction_type="follow",
                    target_type="agent",
                    target_id=agent_id,
                    message=f"{agent_name} has joined your party!",
                    data={"joined": True},
                    events=[{
                        "type": "party_join",
                        "agent_id": agent_id,
                        "agent_name": agent_name
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to add to party: {e}")

        return InteractionResult(
            success=False,
            interaction_type="follow",
            target_type="agent",
            target_id=agent_id,
            message=f"{agent_name} cannot join your party right now"
        )

    async def _dismiss_from_party(
        self,
        session_id: str,
        player_id: str,
        agent_id: str,
        agent_name: str
    ) -> InteractionResult:
        """Dismiss an agent from the party."""
        if self._game_service:
            try:
                await self._game_service.remove_from_party(session_id, agent_id)
                return InteractionResult(
                    success=True,
                    interaction_type="dismiss",
                    target_type="agent",
                    target_id=agent_id,
                    message=f"{agent_name} has left the party",
                    data={"dismissed": True},
                    events=[{
                        "type": "party_leave",
                        "agent_id": agent_id,
                        "agent_name": agent_name
                    }]
                )
            except Exception as e:
                logger.error(f"Failed to dismiss from party: {e}")

        return InteractionResult(
            success=False,
            interaction_type="dismiss",
            target_type="agent",
            target_id=agent_id,
            message=f"Could not dismiss {agent_name}"
        )

    async def _initiate_combat(
        self,
        session_id: str,
        player_id: str,
        agent_id: str,
        agent_name: str
    ) -> InteractionResult:
        """Initiate combat with an agent."""
        # TODO: Implement combat system
        return InteractionResult(
            success=True,
            interaction_type="attack",
            target_type="agent",
            target_id=agent_id,
            message=f"Entering combat with {agent_name}",
            data={
                "combat_id": str(uuid4()),
                "combatants": [player_id, agent_id],
                "combat_mode": "turn_based"
            },
            state_changes={
                "game_mode": "combat"
            }
        )

    # ==========================================================================
    # Object Interactions
    # ==========================================================================

    async def _interact_with_object(
        self,
        session_id: str,
        player_id: str,
        object_id: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with an object."""

        if action == InteractionType.EXAMINE.value:
            return InteractionResult(
                success=True,
                interaction_type="examine",
                target_type="object",
                target_id=object_id,
                message="You examine the object closely.",
                data={"object_id": object_id}
            )

        elif action == InteractionType.USE.value:
            return InteractionResult(
                success=True,
                interaction_type="use",
                target_type="object",
                target_id=object_id,
                message="You use the object.",
                data={"object_id": object_id, "used": True}
            )

        elif action == InteractionType.PICKUP.value:
            return InteractionResult(
                success=True,
                interaction_type="pickup",
                target_type="object",
                target_id=object_id,
                message="You pick up the object.",
                data={"object_id": object_id, "added_to_inventory": True},
                state_changes={"inventory_changed": True}
            )

        return InteractionResult(
            success=False,
            interaction_type=action,
            target_type="object",
            target_id=object_id,
            message=f"Cannot {action} this object"
        )

    # ==========================================================================
    # Tile Interactions
    # ==========================================================================

    async def _interact_with_tile(
        self,
        session_id: str,
        player_id: str,
        tile_coords: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with a tile."""

        if action == InteractionType.EXAMINE.value:
            # Get tile info
            tile_info = await self._get_tile_info(session_id, tile_coords)
            return InteractionResult(
                success=True,
                interaction_type="examine",
                target_type="tile",
                target_id=tile_coords,
                message=self._describe_tile(tile_info),
                data={"tile_info": tile_info}
            )

        return InteractionResult(
            success=False,
            interaction_type=action,
            target_type="tile",
            target_id=tile_coords,
            message=f"Cannot {action} this tile"
        )

    # ==========================================================================
    # Settlement Interactions
    # ==========================================================================

    async def _interact_with_settlement(
        self,
        session_id: str,
        player_id: str,
        settlement_id: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with a settlement."""

        if action == InteractionType.ENTER.value:
            return InteractionResult(
                success=True,
                interaction_type="enter",
                target_type="settlement",
                target_id=settlement_id,
                message="You enter the settlement.",
                data={
                    "settlement_id": settlement_id,
                    "entered": True
                },
                state_changes={
                    "current_settlement": settlement_id
                }
            )

        elif action == InteractionType.EXIT.value:
            return InteractionResult(
                success=True,
                interaction_type="exit",
                target_type="settlement",
                target_id=settlement_id,
                message="You leave the settlement.",
                data={"exited": True},
                state_changes={
                    "current_settlement": None
                }
            )

        elif action == InteractionType.EXAMINE.value:
            # Get settlement info
            settlement_info = await self._get_settlement_info(session_id, settlement_id)
            return InteractionResult(
                success=True,
                interaction_type="examine",
                target_type="settlement",
                target_id=settlement_id,
                message=self._describe_settlement(settlement_info),
                data={"settlement_info": settlement_info}
            )

        return InteractionResult(
            success=False,
            interaction_type=action,
            target_type="settlement",
            target_id=settlement_id,
            message=f"Cannot {action} this settlement"
        )

    # ==========================================================================
    # Building Interactions
    # ==========================================================================

    async def _interact_with_building(
        self,
        session_id: str,
        player_id: str,
        building_id: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with a building."""

        if action == InteractionType.ENTER.value:
            return InteractionResult(
                success=True,
                interaction_type="enter",
                target_type="building",
                target_id=building_id,
                message="You enter the building.",
                data={
                    "building_id": building_id,
                    "entered": True
                },
                state_changes={
                    "current_building": building_id
                }
            )

        elif action == InteractionType.EXIT.value:
            return InteractionResult(
                success=True,
                interaction_type="exit",
                target_type="building",
                target_id=building_id,
                message="You exit the building.",
                data={"exited": True},
                state_changes={
                    "current_building": None
                }
            )

        return InteractionResult(
            success=False,
            interaction_type=action,
            target_type="building",
            target_id=building_id,
            message=f"Cannot {action} this building"
        )

    # ==========================================================================
    # Item Interactions
    # ==========================================================================

    async def _interact_with_item(
        self,
        session_id: str,
        player_id: str,
        item_id: str,
        action: str,
        action_data: Dict[str, Any]
    ) -> InteractionResult:
        """Handle player interaction with an inventory item."""

        if action == InteractionType.USE.value:
            return InteractionResult(
                success=True,
                interaction_type="use",
                target_type="item",
                target_id=item_id,
                message="You use the item.",
                data={"item_id": item_id, "used": True},
                state_changes={"inventory_changed": True}
            )

        elif action == InteractionType.DROP.value:
            return InteractionResult(
                success=True,
                interaction_type="drop",
                target_type="item",
                target_id=item_id,
                message="You drop the item.",
                data={"item_id": item_id, "dropped": True},
                state_changes={"inventory_changed": True}
            )

        elif action == InteractionType.EXAMINE.value:
            return InteractionResult(
                success=True,
                interaction_type="examine",
                target_type="item",
                target_id=item_id,
                message="You examine the item closely.",
                data={"item_id": item_id}
            )

        return InteractionResult(
            success=False,
            interaction_type=action,
            target_type="item",
            target_id=item_id,
            message=f"Cannot {action} this item"
        )

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    async def _get_agent_info(
        self,
        session_id: str,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get agent information from game service."""
        if not self._game_service:
            return None

        try:
            session = self._game_service._sessions.get(session_id)
            if not session:
                return None

            # Check regular agents
            agent = session.get('agents', {}).get(agent_id)
            if agent:
                return agent

            # Check party members
            for member in session.get('party_members', []):
                if member.get('id') == agent_id:
                    return member

            return None
        except Exception as e:
            logger.error(f"Failed to get agent info: {e}")
            return None

    async def _get_tile_info(
        self,
        session_id: str,
        tile_coords: str
    ) -> Dict[str, Any]:
        """Get tile information."""
        # Parse coords (format: "x,y")
        try:
            x, y = map(int, tile_coords.split(','))
        except:
            return {"type": "unknown"}

        # Would get from world service
        return {
            "x": x,
            "y": y,
            "type": "grass",
            "features": []
        }

    async def _get_settlement_info(
        self,
        session_id: str,
        settlement_id: str
    ) -> Dict[str, Any]:
        """Get settlement information."""
        # Would get from world service
        return {
            "id": settlement_id,
            "name": "Unknown Settlement",
            "type": "village",
            "population": 0
        }

    def _generate_agent_description(self, agent_info: Dict[str, Any]) -> str:
        """Generate a description of an agent."""
        name = agent_info.get('name', 'Unknown')
        occupation = agent_info.get('occupation', 'person')
        traits = agent_info.get('traits', {})

        description = f"You see {name}, a {occupation}."

        if traits:
            personality = traits.get('personality', '')
            if personality:
                description += f" They appear to be {personality}."

        return description

    def _describe_tile(self, tile_info: Dict[str, Any]) -> str:
        """Generate a description of a tile."""
        tile_type = tile_info.get('type', 'terrain')
        features = tile_info.get('features', [])

        descriptions = {
            'grass': 'A grassy area with soft ground',
            'forest': 'Dense woodland with towering trees',
            'water': 'Clear water reflects the sky',
            'mountain': 'Rocky terrain rises steeply',
            'sand': 'Fine sand stretches before you',
            'snow': 'Snow blankets the ground'
        }

        base = descriptions.get(tile_type, f'An area of {tile_type}')

        if features:
            base += f". You notice: {', '.join(features)}"

        return base

    def _describe_settlement(self, settlement_info: Dict[str, Any]) -> str:
        """Generate a description of a settlement."""
        name = settlement_info.get('name', 'settlement')
        stype = settlement_info.get('type', 'settlement')
        pop = settlement_info.get('population', 0)

        return f"{name} is a {stype} with a population of about {pop}."

    # ==========================================================================
    # Nearby Detection
    # ==========================================================================

    async def get_interactable_nearby(
        self,
        session_id: str,
        player_x: int,
        player_y: int,
        radius: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all interactable entities near the player.

        Returns:
            Dict with keys: agents, objects, buildings, tiles
        """
        result = {
            "agents": [],
            "objects": [],
            "buildings": [],
            "tiles": []
        }

        if self._game_service:
            try:
                session = self._game_service._sessions.get(session_id)
                if session:
                    # Get nearby agents
                    for agent_id, agent in session.get('agents', {}).items():
                        ax = agent.get('x', 0)
                        ay = agent.get('y', 0)
                        dist = abs(ax - player_x) + abs(ay - player_y)
                        if dist <= radius:
                            result["agents"].append({
                                "id": agent_id,
                                "name": agent.get('name', 'Unknown'),
                                "distance": dist,
                                "position": {"x": ax, "y": ay},
                                "actions": ["talk", "examine", "trade", "follow"]
                            })
            except Exception as e:
                logger.error(f"Failed to get nearby entities: {e}")

        return result

    async def get_available_actions(
        self,
        target_type: str,
        target_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get available actions for a target.

        Returns:
            List of action names
        """
        context = context or {}

        if target_type == "agent":
            in_party = context.get('in_party', False)
            if in_party:
                return ["talk", "examine", "dismiss", "trade"]
            else:
                return ["talk", "examine", "follow", "trade", "attack"]

        elif target_type == "object":
            usable = context.get('usable', False)
            pickable = context.get('pickable', True)
            actions = ["examine"]
            if usable:
                actions.append("use")
            if pickable:
                actions.append("pickup")
            return actions

        elif target_type == "tile":
            return ["examine"]

        elif target_type == "settlement":
            inside = context.get('inside', False)
            return ["exit", "examine"] if inside else ["enter", "examine"]

        elif target_type == "building":
            inside = context.get('inside', False)
            return ["exit", "examine"] if inside else ["enter", "examine"]

        elif target_type == "item":
            return ["use", "examine", "drop"]

        return ["examine"]
