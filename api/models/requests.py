"""
API Request Models
Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class WorldSize(str, Enum):
    SMALL = "SMALL"      # 512x512
    MEDIUM = "MEDIUM"    # 1024x1024
    LARGE = "LARGE"      # 2048x2048
    HUGE = "HUGE"        # 4096x4096


class PartyRole(str, Enum):
    WARRIOR = "warrior"
    MAGE = "mage"
    HEALER = "healer"
    ROGUE = "rogue"
    RANGER = "ranger"
    COMPANION = "companion"


class ChatChannel(str, Enum):
    PARTY = "party"
    LOCAL = "local"
    GLOBAL = "global"


class InteractionAction(str, Enum):
    TALK = "talk"
    TRADE = "trade"
    ATTACK = "attack"
    FOLLOW = "follow"
    EXAMINE = "examine"
    USE = "use"
    PICKUP = "pickup"
    GATHER = "gather"


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


# =============================================================================
# World Requests
# =============================================================================

class CreateWorldRequest(BaseModel):
    """Request to create a new world."""
    name: str = Field(..., min_length=1, max_length=255)
    seed: Optional[int] = Field(None, description="Random seed for generation")
    size: WorldSize = Field(WorldSize.MEDIUM, description="World size")

    # Advanced options
    planet_radius_km: float = Field(6371.0, description="Planet radius")
    axial_tilt: float = Field(23.5, description="Axial tilt in degrees")
    ocean_percentage: float = Field(0.7, ge=0, le=1, description="Ocean coverage")
    num_plates: int = Field(12, ge=4, le=30, description="Tectonic plates")

    # Features
    enable_magic: bool = Field(True, description="Enable magic system")
    enable_caves: bool = Field(True, description="Generate caves")
    settlement_density: str = Field("medium", description="low/medium/high")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Fantasy World",
                "seed": 12345,
                "size": "MEDIUM",
                "enable_magic": True
            }
        }


class WorldQueryParams(BaseModel):
    """Query parameters for world data."""
    x_min: int = -50
    x_max: int = 50
    y_min: int = -50
    y_max: int = 50


# =============================================================================
# Game Session Requests
# =============================================================================

class PlayerConfig(BaseModel):
    """Configuration for player character creation."""
    name: str = Field(..., min_length=1, max_length=100)
    spawn_x: Optional[int] = None
    spawn_y: Optional[int] = None
    spawn_settlement_id: Optional[str] = None

    # Stats (optional - defaults used if not provided)
    strength: int = Field(10, ge=1, le=20)
    dexterity: int = Field(10, ge=1, le=20)
    constitution: int = Field(10, ge=1, le=20)
    intelligence: int = Field(10, ge=1, le=20)
    wisdom: int = Field(10, ge=1, le=20)
    charisma: int = Field(10, ge=1, le=20)


class PartyConfig(BaseModel):
    """Configuration for party creation."""
    size: int = Field(3, ge=0, le=6, description="Number of party members")
    roles: Optional[List[PartyRole]] = Field(None, description="Roles for each member")
    names: Optional[List[str]] = Field(None, description="Names for each member")
    auto_generate: bool = Field(True, description="Auto-generate missing details")


class GameSettings(BaseModel):
    """Game session settings."""
    tick_interval_ms: int = Field(1000, ge=100, le=10000)
    auto_tick: bool = Field(False, description="Start with auto-tick enabled")
    debug_mode: bool = Field(False, description="Enable debug features")
    difficulty: str = Field("normal", description="easy/normal/hard")


class CreateSessionRequest(BaseModel):
    """Request to create a new game session."""
    world_id: str
    player: PlayerConfig
    party: Optional[PartyConfig] = None
    settings: Optional[GameSettings] = None


class TickRequest(BaseModel):
    """Request to advance simulation."""
    num_ticks: int = Field(1, ge=1, le=100)


# =============================================================================
# Player Requests
# =============================================================================

class CreatePlayerRequest(BaseModel):
    """Request to create a player character."""
    session_id: str
    name: str = Field(..., min_length=1, max_length=100)
    spawn_x: int
    spawn_y: int
    stats: Optional[Dict[str, int]] = None
    appearance: Optional[Dict[str, Any]] = None


class PlayerMoveRequest(BaseModel):
    """Request to move the player."""
    session_id: str
    player_id: str
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    direction: Optional[Direction] = None


class PlayerInteractRequest(BaseModel):
    """Request for player interaction."""
    session_id: str
    player_id: str
    target_type: str  # agent, object, tile, settlement
    target_id: str
    action: InteractionAction
    parameters: Optional[Dict[str, Any]] = None


class PlayerSpeakRequest(BaseModel):
    """Request for player to speak."""
    session_id: str
    player_id: str
    message: str = Field(..., min_length=1, max_length=1000)
    channel: ChatChannel = ChatChannel.PARTY


# =============================================================================
# Agent Requests
# =============================================================================

class DialogueRequest(BaseModel):
    """Request to dialogue with an agent."""
    session_id: str
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None


class AgentCommandRequest(BaseModel):
    """Request to command an agent."""
    session_id: str
    command: str  # follow, wait, move_to, attack, defend
    parameters: Optional[Dict[str, Any]] = None


# =============================================================================
# Party Requests
# =============================================================================

class CreatePartyRequest(BaseModel):
    """Request to create a party."""
    session_id: str
    player_id: str
    size: int = Field(3, ge=1, le=6)
    roles: Optional[List[PartyRole]] = None
    names: Optional[List[str]] = None


class PartyCommandRequest(BaseModel):
    """Request to command the party."""
    session_id: str
    command: str  # follow, wait, spread, guard, rest
    parameters: Optional[Dict[str, Any]] = None


class PartyDialogueRequest(BaseModel):
    """Request to dialogue with a party member."""
    session_id: str
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
