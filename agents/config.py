"""
Agent Configuration and Constants
Contains all settings for the agent simulation system
"""

from enum import IntEnum, Enum
from typing import Optional
from pydantic import BaseModel, Field
import os

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# =============================================================================
# AGENT CONSTANTS
# =============================================================================

# Stats range (D&D style)
STAT_MIN = 1
STAT_MAX = 20
STAT_DEFAULT = 10

# Needs range (0 = satisfied, 1 = critical)
NEED_MIN = 0.0
NEED_MAX = 1.0

# Alignment range (-1.0 to 1.0)
ALIGNMENT_MIN = -1.0
ALIGNMENT_MAX = 1.0

# Need decay rates (per hour of game time)
HUNGER_DECAY_RATE = 0.04  # ~25 hours to go from satisfied to critical
THIRST_DECAY_RATE = 0.06  # ~17 hours
REST_DECAY_RATE = 0.04    # ~25 hours
WARMTH_DECAY_RATE = 0.02  # Environment dependent
SAFETY_DECAY_RATE = 0.01  # Very slow, context dependent
SOCIAL_DECAY_RATE = 0.02  # ~50 hours

# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================

# Memory retrieval weights (Stanford paper: all 1.0 initially)
ALPHA_RECENCY = 1.0
ALPHA_IMPORTANCE = 1.0
ALPHA_RELEVANCE = 1.0

# Recency decay factor (0.995^hours)
RECENCY_DECAY_FACTOR = 0.995

# Reflection trigger threshold (sum of importance over 24h)
REFLECTION_THRESHOLD = 150

# Maximum memories to retrieve at once
MAX_MEMORY_RETRIEVAL = 10

# Embedding dimension (1536 for OpenAI, adjust for local models)
EMBEDDING_DIMENSION = 1536

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# Time settings
SIMULATION_TICK_SECONDS = 300  # 5 minutes per tick
GAME_HOURS_PER_REAL_SECOND = 60  # 1 real second = 1 game minute

# Agent perception radius (in world tiles)
PERCEPTION_RADIUS = 10

# Maximum agents to simulate in parallel
MAX_PARALLEL_AGENTS = 100

# =============================================================================
# SKILL CONFIGURATION
# =============================================================================

# XP multipliers
BASE_XP_PER_USE = 1.0
XP_TO_LEVEL_BASE = 100.0
PARENT_XP_SHARE = 0.3  # 30% XP goes to parent skill

# Stat improvement from skill use
STAT_IMPROVEMENT_CHANCE = 0.001  # 0.1% chance per skill use

# Primary/Secondary stat weight for skill checks
PRIMARY_STAT_WEIGHT = 0.7
SECONDARY_STAT_WEIGHT = 0.3

# =============================================================================
# ENUMERATIONS
# =============================================================================

class AgentType(str, Enum):
    """Types of agents in the simulation"""
    HUMAN = "human"
    ANIMAL = "animal"
    SETTLEMENT = "settlement"  # Collective agent for settlements
    GUILD = "guild"  # Organization agent
    FACTION = "faction"  # Political entity


class LifeStage(str, Enum):
    """Life stages for human agents"""
    INFANT = "infant"      # 0-2 years
    CHILD = "child"        # 2-12 years
    ADOLESCENT = "adolescent"  # 12-18 years
    ADULT = "adult"        # 18-60 years
    ELDER = "elder"        # 60+ years


class MemoryType(str, Enum):
    """Types of memories in the memory stream"""
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PLAN = "plan"


class ActionStatus(str, Enum):
    """Status of an agent action"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
