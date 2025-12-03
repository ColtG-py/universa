"""
Hierarchical Agent Scheduler
Schedules agent execution based on proximity to player and organizational hierarchy.
Not all agents need to think every tick - this reduces LLM calls dramatically.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any
from uuid import UUID
import math
import logging

logger = logging.getLogger(__name__)


class AgentTier(IntEnum):
    """
    Agent importance tiers for scheduling.
    Lower values = higher priority = more frequent execution.
    """
    PLAYER_PARTY = 1      # Every tick - player's party members
    ACTIVE = 2            # Every tick - agents in active interaction
    NEARBY = 3            # Every 2-3 ticks - within perception radius
    SAME_SETTLEMENT = 4   # Every 5-10 ticks - same location as player
    SAME_REGION = 5       # Every 20-50 ticks - nearby settlements
    BACKGROUND = 6        # Every 100+ ticks - distant agents
    DORMANT = 7           # Event-driven only - far away, no activity


@dataclass
class TierConfig:
    """Configuration for a tier's execution behavior."""
    tick_interval: int           # Execute every N ticks
    max_agents_per_tick: int     # Max agents to execute per tick (0 = unlimited)
    use_simplified_cycle: bool   # Use simplified execution (no full LLM reasoning)
    batch_size: int              # How many to run in parallel


# Default tier configurations
DEFAULT_TIER_CONFIGS: Dict[AgentTier, TierConfig] = {
    AgentTier.PLAYER_PARTY: TierConfig(
        tick_interval=1,
        max_agents_per_tick=0,  # Always run all
        use_simplified_cycle=False,
        batch_size=6
    ),
    AgentTier.ACTIVE: TierConfig(
        tick_interval=1,
        max_agents_per_tick=10,
        use_simplified_cycle=False,
        batch_size=5
    ),
    AgentTier.NEARBY: TierConfig(
        tick_interval=3,
        max_agents_per_tick=20,
        use_simplified_cycle=False,
        batch_size=10
    ),
    AgentTier.SAME_SETTLEMENT: TierConfig(
        tick_interval=10,
        max_agents_per_tick=50,
        use_simplified_cycle=True,  # Simplified reasoning
        batch_size=20
    ),
    AgentTier.SAME_REGION: TierConfig(
        tick_interval=50,
        max_agents_per_tick=100,
        use_simplified_cycle=True,
        batch_size=25
    ),
    AgentTier.BACKGROUND: TierConfig(
        tick_interval=200,
        max_agents_per_tick=50,
        use_simplified_cycle=True,
        batch_size=25
    ),
    AgentTier.DORMANT: TierConfig(
        tick_interval=0,  # 0 = event-driven only
        max_agents_per_tick=0,
        use_simplified_cycle=True,
        batch_size=10
    ),
}


@dataclass
class AgentTierInfo:
    """Tracking info for an agent's tier status."""
    agent_id: UUID
    current_tier: AgentTier
    last_executed_tick: int = 0
    last_tier_change_tick: int = 0
    execution_count: int = 0
    is_in_party: bool = False
    is_in_active_interaction: bool = False
    settlement_id: Optional[UUID] = None
    region_id: Optional[str] = None


@dataclass
class PlayerContext:
    """Player position and state for tier classification."""
    player_id: UUID
    x: int
    y: int
    settlement_id: Optional[UUID] = None
    region_id: Optional[str] = None
    party_agent_ids: Set[UUID] = field(default_factory=set)
    active_interaction_agent_ids: Set[UUID] = field(default_factory=set)
    perception_radius: int = 10  # Tiles


class HierarchicalScheduler:
    """
    Schedules agent execution based on proximity to player and hierarchy.

    Key features:
    - Tier-based execution frequency
    - Automatic tier reclassification as player moves
    - Support for collective agents (settlements, kingdoms)
    - Simplified execution for distant agents
    """

    def __init__(
        self,
        tier_configs: Optional[Dict[AgentTier, TierConfig]] = None,
        distance_thresholds: Optional[Dict[str, int]] = None
    ):
        self.tier_configs = tier_configs or DEFAULT_TIER_CONFIGS

        # Distance thresholds for tier classification
        self.distance_thresholds = distance_thresholds or {
            'nearby': 15,           # Within 15 tiles = NEARBY
            'same_settlement': 0,   # Same settlement (uses settlement_id)
            'same_region': 100,     # Within 100 tiles or same region
            'background': 500,      # Within 500 tiles
        }

        # Agent tracking
        self._agents: Dict[UUID, AgentTierInfo] = {}

        # Current player context
        self._player_context: Optional[PlayerContext] = None

        # Settlement/region mappings
        self._settlement_agents: Dict[UUID, Set[UUID]] = {}  # settlement_id -> agent_ids
        self._region_agents: Dict[str, Set[UUID]] = {}       # region_id -> agent_ids

        # Statistics
        self._stats = {
            'total_classifications': 0,
            'tier_changes': 0,
            'executions_by_tier': {tier: 0 for tier in AgentTier},
        }

    def register_agent(
        self,
        agent_id: UUID,
        x: int,
        y: int,
        settlement_id: Optional[UUID] = None,
        region_id: Optional[str] = None
    ) -> AgentTierInfo:
        """Register a new agent for scheduling."""
        info = AgentTierInfo(
            agent_id=agent_id,
            current_tier=AgentTier.BACKGROUND,  # Default until classified
            settlement_id=settlement_id,
            region_id=region_id
        )
        self._agents[agent_id] = info

        # Track by settlement/region
        if settlement_id:
            if settlement_id not in self._settlement_agents:
                self._settlement_agents[settlement_id] = set()
            self._settlement_agents[settlement_id].add(agent_id)

        if region_id:
            if region_id not in self._region_agents:
                self._region_agents[region_id] = set()
            self._region_agents[region_id].add(agent_id)

        # Classify if player context exists
        if self._player_context:
            self._classify_agent(info, x, y)

        return info

    def unregister_agent(self, agent_id: UUID):
        """Remove an agent from scheduling."""
        info = self._agents.pop(agent_id, None)
        if info:
            # Remove from settlement/region tracking
            if info.settlement_id and info.settlement_id in self._settlement_agents:
                self._settlement_agents[info.settlement_id].discard(agent_id)
            if info.region_id and info.region_id in self._region_agents:
                self._region_agents[info.region_id].discard(agent_id)

    def update_player_context(self, context: PlayerContext):
        """Update player position and state, triggering reclassification."""
        old_context = self._player_context
        self._player_context = context

        # Check if significant movement occurred
        if old_context:
            dx = abs(context.x - old_context.x)
            dy = abs(context.y - old_context.y)
            settlement_changed = context.settlement_id != old_context.settlement_id

            # Only reclassify all if significant change
            if dx > 5 or dy > 5 or settlement_changed:
                logger.debug(f"Player moved significantly, reclassifying all agents")
                # Will be reclassified on next get_agents_to_execute call

    def set_agent_in_party(self, agent_id: UUID, in_party: bool):
        """Mark agent as in/out of player's party."""
        info = self._agents.get(agent_id)
        if info:
            info.is_in_party = in_party
            if self._player_context:
                if in_party:
                    self._player_context.party_agent_ids.add(agent_id)
                else:
                    self._player_context.party_agent_ids.discard(agent_id)
                # Immediately reclassify to update tier
                self._classify_agent(info, 0, 0)  # Position doesn't matter for party

    def set_agent_active_interaction(self, agent_id: UUID, active: bool):
        """Mark agent as in/out of active interaction with player."""
        info = self._agents.get(agent_id)
        if info:
            info.is_in_active_interaction = active
            if self._player_context:
                if active:
                    self._player_context.active_interaction_agent_ids.add(agent_id)
                else:
                    self._player_context.active_interaction_agent_ids.discard(agent_id)
                # Immediately reclassify to update tier
                self._classify_agent(info, 0, 0)  # Position doesn't matter for active

    def update_agent_position(
        self,
        agent_id: UUID,
        x: int,
        y: int,
        settlement_id: Optional[UUID] = None,
        region_id: Optional[str] = None
    ):
        """Update agent position and reclassify."""
        info = self._agents.get(agent_id)
        if not info:
            return

        # Update settlement/region tracking
        if info.settlement_id != settlement_id:
            if info.settlement_id and info.settlement_id in self._settlement_agents:
                self._settlement_agents[info.settlement_id].discard(agent_id)
            if settlement_id:
                if settlement_id not in self._settlement_agents:
                    self._settlement_agents[settlement_id] = set()
                self._settlement_agents[settlement_id].add(agent_id)
            info.settlement_id = settlement_id

        if info.region_id != region_id:
            if info.region_id and info.region_id in self._region_agents:
                self._region_agents[info.region_id].discard(agent_id)
            if region_id:
                if region_id not in self._region_agents:
                    self._region_agents[region_id] = set()
                self._region_agents[region_id].add(agent_id)
            info.region_id = region_id

        # Reclassify
        if self._player_context:
            self._classify_agent(info, x, y)

    def _classify_agent(self, info: AgentTierInfo, agent_x: int, agent_y: int) -> AgentTier:
        """Classify an agent into the appropriate tier."""
        if not self._player_context:
            return AgentTier.DORMANT

        self._stats['total_classifications'] += 1
        old_tier = info.current_tier
        new_tier: AgentTier

        # Check special states first
        if info.is_in_party or info.agent_id in self._player_context.party_agent_ids:
            new_tier = AgentTier.PLAYER_PARTY
        elif info.is_in_active_interaction or info.agent_id in self._player_context.active_interaction_agent_ids:
            new_tier = AgentTier.ACTIVE
        else:
            # Distance-based classification
            distance = self._calculate_distance(
                agent_x, agent_y,
                self._player_context.x, self._player_context.y
            )

            if distance <= self.distance_thresholds['nearby']:
                new_tier = AgentTier.NEARBY
            elif (info.settlement_id and
                  info.settlement_id == self._player_context.settlement_id):
                new_tier = AgentTier.SAME_SETTLEMENT
            elif distance <= self.distance_thresholds['same_region']:
                new_tier = AgentTier.SAME_REGION
            elif distance <= self.distance_thresholds['background']:
                new_tier = AgentTier.BACKGROUND
            else:
                new_tier = AgentTier.DORMANT

        if new_tier != old_tier:
            self._stats['tier_changes'] += 1
            info.last_tier_change_tick = info.last_executed_tick
            logger.debug(f"Agent {info.agent_id} tier changed: {old_tier.name} -> {new_tier.name}")

        info.current_tier = new_tier
        return new_tier

    def _calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_agents_to_execute(
        self,
        current_tick: int,
        agent_positions: Optional[Dict[UUID, tuple]] = None
    ) -> Dict[AgentTier, List[UUID]]:
        """
        Get agents that should execute this tick, grouped by tier.

        Args:
            current_tick: Current simulation tick number
            agent_positions: Optional dict of agent_id -> (x, y) for reclassification

        Returns:
            Dict mapping tier -> list of agent_ids to execute
        """
        # Reclassify agents if positions provided
        if agent_positions and self._player_context:
            for agent_id, (x, y) in agent_positions.items():
                if agent_id in self._agents:
                    self._classify_agent(self._agents[agent_id], x, y)

        result: Dict[AgentTier, List[UUID]] = {tier: [] for tier in AgentTier}

        for agent_id, info in self._agents.items():
            tier = info.current_tier
            config = self.tier_configs[tier]

            # Skip dormant agents (event-driven only)
            if config.tick_interval == 0:
                continue

            # Check if this agent should execute this tick
            ticks_since_last = current_tick - info.last_executed_tick
            if ticks_since_last >= config.tick_interval:
                result[tier].append(agent_id)

        # Apply per-tick limits
        for tier in AgentTier:
            config = self.tier_configs[tier]
            if config.max_agents_per_tick > 0 and len(result[tier]) > config.max_agents_per_tick:
                # Round-robin: prioritize agents that haven't run in longest
                result[tier].sort(key=lambda aid: self._agents[aid].last_executed_tick)
                result[tier] = result[tier][:config.max_agents_per_tick]

        return result

    def mark_executed(self, agent_id: UUID, tick: int):
        """Mark an agent as having executed at the given tick."""
        info = self._agents.get(agent_id)
        if info:
            info.last_executed_tick = tick
            info.execution_count += 1
            self._stats['executions_by_tier'][info.current_tier] += 1

    def wake_dormant_agent(self, agent_id: UUID, reason: str = "event"):
        """Wake a dormant agent for immediate execution."""
        info = self._agents.get(agent_id)
        if info and info.current_tier == AgentTier.DORMANT:
            # Temporarily promote to BACKGROUND for one execution
            info.current_tier = AgentTier.BACKGROUND
            info.last_executed_tick = 0  # Ensure it runs next tick
            logger.debug(f"Woke dormant agent {agent_id}: {reason}")

    def get_agent_tier(self, agent_id: UUID) -> Optional[AgentTier]:
        """Get an agent's current tier."""
        info = self._agents.get(agent_id)
        return info.current_tier if info else None

    def get_tier_stats(self) -> Dict[AgentTier, int]:
        """Get count of agents in each tier."""
        counts = {tier: 0 for tier in AgentTier}
        for info in self._agents.values():
            counts[info.current_tier] += 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        tier_counts = self.get_tier_stats()
        return {
            'total_agents': len(self._agents),
            'agents_by_tier': {tier.name: count for tier, count in tier_counts.items()},
            'total_classifications': self._stats['total_classifications'],
            'tier_changes': self._stats['tier_changes'],
            'executions_by_tier': {
                tier.name: count
                for tier, count in self._stats['executions_by_tier'].items()
            },
            'tier_configs': {
                tier.name: {
                    'tick_interval': config.tick_interval,
                    'max_per_tick': config.max_agents_per_tick,
                    'simplified': config.use_simplified_cycle
                }
                for tier, config in self.tier_configs.items()
            }
        }

    def get_settlement_agents(self, settlement_id: UUID) -> Set[UUID]:
        """Get all agents in a settlement."""
        return self._settlement_agents.get(settlement_id, set()).copy()

    def get_region_agents(self, region_id: str) -> Set[UUID]:
        """Get all agents in a region."""
        return self._region_agents.get(region_id, set()).copy()
