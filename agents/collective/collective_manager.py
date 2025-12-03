"""
Collective Manager
Manages all collective agents (settlements and kingdoms) and coordinates
their effects on individual agents.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Tuple
from uuid import UUID
import logging
import asyncio

from .settlement_agent import SettlementAgent, SettlementModifiers
from .kingdom_agent import KingdomAgent, KingdomModifiers

logger = logging.getLogger(__name__)


@dataclass
class CombinedModifiers:
    """Combined modifiers from settlement and kingdom."""
    # Settlement modifiers
    settlement_mood: float = 0.0
    settlement_safety: float = 0.0
    settlement_economy: float = 0.0

    # Kingdom modifiers
    tax_rate: float = 0.1
    military_levy: float = 0.0
    law_enforcement: float = 0.5

    # Combined effects
    total_mood_modifier: float = 0.0
    total_safety_modifier: float = 0.0
    total_economy_modifier: float = 0.0

    # Activity suggestions
    encouraged_activities: List[str] = None
    discouraged_activities: List[str] = None

    # Context for agent perception
    settlement_context: Dict[str, Any] = None
    kingdom_context: Dict[str, Any] = None

    def __post_init__(self):
        if self.encouraged_activities is None:
            self.encouraged_activities = []
        if self.discouraged_activities is None:
            self.discouraged_activities = []
        if self.settlement_context is None:
            self.settlement_context = {}
        if self.kingdom_context is None:
            self.kingdom_context = {}


class CollectiveManager:
    """
    Manages all collective agents and their effects on individual agents.

    Responsibilities:
    - Track all settlements and kingdoms
    - Execute collective ticks at appropriate intervals
    - Provide combined modifiers to individual agents
    - Coordinate settlement<->kingdom relationships
    """

    def __init__(
        self,
        settlement_tick_interval: int = 10,
        kingdom_tick_interval: int = 100
    ):
        self.settlement_tick_interval = settlement_tick_interval
        self.kingdom_tick_interval = kingdom_tick_interval

        # Collective agents
        self.settlements: Dict[UUID, SettlementAgent] = {}
        self.kingdoms: Dict[UUID, KingdomAgent] = {}

        # Mappings
        self.settlement_to_kingdom: Dict[UUID, UUID] = {}
        self.agent_to_settlement: Dict[UUID, UUID] = {}

        # Cached modifiers
        self._settlement_modifiers: Dict[UUID, SettlementModifiers] = {}
        self._kingdom_modifiers: Dict[UUID, KingdomModifiers] = {}

        # Tick tracking
        self.last_settlement_tick = 0
        self.last_kingdom_tick = 0

    def register_settlement(
        self,
        settlement_id: UUID,
        name: str,
        settlement_type: str,
        x: int,
        y: int,
        population: int = 100,
        faction_id: Optional[UUID] = None
    ) -> SettlementAgent:
        """Register a new settlement."""
        settlement = SettlementAgent(
            settlement_id=settlement_id,
            name=name,
            settlement_type=settlement_type,
            x=x,
            y=y,
            population=population,
            faction_id=faction_id
        )
        self.settlements[settlement_id] = settlement

        # Link to kingdom if faction exists
        if faction_id and faction_id in self.kingdoms:
            self.settlement_to_kingdom[settlement_id] = faction_id
            self.kingdoms[faction_id].add_settlement(settlement_id)

        logger.debug(f"Registered settlement: {name} ({settlement_id})")
        return settlement

    def register_kingdom(
        self,
        faction_id: UUID,
        name: str,
        faction_type: str,
        capital_settlement_id: Optional[UUID] = None
    ) -> KingdomAgent:
        """Register a new kingdom."""
        kingdom = KingdomAgent(
            faction_id=faction_id,
            name=name,
            faction_type=faction_type,
            capital_settlement_id=capital_settlement_id
        )
        self.kingdoms[faction_id] = kingdom

        # Link existing settlements to this kingdom
        for sid, settlement in self.settlements.items():
            if settlement.faction_id == faction_id:
                self.settlement_to_kingdom[sid] = faction_id
                kingdom.add_settlement(sid)

        logger.debug(f"Registered kingdom: {name} ({faction_id})")
        return kingdom

    def register_agent_in_settlement(self, agent_id: UUID, settlement_id: UUID, initial_mood: float = 0.5):
        """Register an agent as a resident of a settlement."""
        self.agent_to_settlement[agent_id] = settlement_id

        if settlement_id in self.settlements:
            self.settlements[settlement_id].add_resident(agent_id, initial_mood)

    def unregister_agent(self, agent_id: UUID):
        """Remove an agent from their settlement."""
        settlement_id = self.agent_to_settlement.pop(agent_id, None)
        if settlement_id and settlement_id in self.settlements:
            self.settlements[settlement_id].remove_resident(agent_id)

    def move_agent_to_settlement(self, agent_id: UUID, new_settlement_id: UUID):
        """Move an agent to a different settlement."""
        # Remove from old settlement
        old_settlement_id = self.agent_to_settlement.get(agent_id)
        if old_settlement_id and old_settlement_id in self.settlements:
            self.settlements[old_settlement_id].remove_resident(agent_id)

        # Add to new settlement
        self.agent_to_settlement[agent_id] = new_settlement_id
        if new_settlement_id in self.settlements:
            self.settlements[new_settlement_id].add_resident(agent_id)

    def update_agent_mood(self, agent_id: UUID, mood: float):
        """Update an agent's mood in their settlement."""
        settlement_id = self.agent_to_settlement.get(agent_id)
        if settlement_id and settlement_id in self.settlements:
            self.settlements[settlement_id].update_resident_mood(agent_id, mood)

    async def tick(self, current_tick: int, game_hour: int) -> Dict[str, Any]:
        """
        Execute collective agent ticks.

        Args:
            current_tick: Current simulation tick
            game_hour: Current game hour (0-23)

        Returns:
            Summary of collective actions taken
        """
        results = {
            'settlements_ticked': 0,
            'kingdoms_ticked': 0,
            'events_generated': [],
        }

        # Tick settlements if interval reached
        if current_tick - self.last_settlement_tick >= self.settlement_tick_interval:
            settlement_results = await self._tick_settlements(current_tick, game_hour)
            results['settlements_ticked'] = settlement_results['count']
            results['events_generated'].extend(settlement_results['events'])
            self.last_settlement_tick = current_tick

        # Tick kingdoms if interval reached
        if current_tick - self.last_kingdom_tick >= self.kingdom_tick_interval:
            kingdom_results = await self._tick_kingdoms(current_tick)
            results['kingdoms_ticked'] = kingdom_results['count']
            self.last_kingdom_tick = current_tick

        return results

    async def _tick_settlements(self, current_tick: int, game_hour: int) -> Dict[str, Any]:
        """Execute settlement ticks."""
        events = []

        # Run settlements in parallel
        async def tick_one(settlement: SettlementAgent):
            modifiers = await settlement.collective_tick(current_tick, game_hour)
            self._settlement_modifiers[settlement.settlement_id] = modifiers
            return modifiers

        tasks = [tick_one(s) for s in self.settlements.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect events
        for settlement in self.settlements.values():
            for event in settlement.active_events:
                events.append({
                    'settlement_id': str(settlement.settlement_id),
                    'settlement_name': settlement.name,
                    'event_type': event.event_type.value,
                    'description': event.description,
                })

        return {
            'count': len(self.settlements),
            'events': events,
        }

    async def _tick_kingdoms(self, current_tick: int) -> Dict[str, Any]:
        """Execute kingdom ticks."""
        async def tick_one(kingdom: KingdomAgent):
            modifiers = await kingdom.kingdom_tick(current_tick)
            self._kingdom_modifiers[kingdom.faction_id] = modifiers
            return modifiers

        tasks = [tick_one(k) for k in self.kingdoms.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'count': len(self.kingdoms),
        }

    def get_agent_modifiers(self, agent_id: UUID) -> CombinedModifiers:
        """
        Get combined modifiers for an agent based on their location.

        Args:
            agent_id: The agent to get modifiers for

        Returns:
            Combined settlement and kingdom modifiers
        """
        modifiers = CombinedModifiers()

        # Get settlement modifiers
        settlement_id = self.agent_to_settlement.get(agent_id)
        if settlement_id and settlement_id in self.settlements:
            settlement = self.settlements[settlement_id]
            s_mods = self._settlement_modifiers.get(settlement_id)

            if s_mods:
                modifiers.settlement_mood = s_mods.mood_modifier
                modifiers.settlement_safety = s_mods.safety_modifier
                modifiers.settlement_economy = s_mods.economy_modifier
                modifiers.encouraged_activities = s_mods.encouraged_activities.copy()
                modifiers.discouraged_activities = s_mods.discouraged_activities.copy()

            modifiers.settlement_context = settlement.get_resident_context(agent_id)

            # Get kingdom modifiers if in a kingdom
            kingdom_id = self.settlement_to_kingdom.get(settlement_id)
            if kingdom_id and kingdom_id in self.kingdoms:
                kingdom = self.kingdoms[kingdom_id]
                k_mods = self._kingdom_modifiers.get(kingdom_id)

                if k_mods:
                    modifiers.tax_rate = k_mods.tax_rate
                    modifiers.military_levy = k_mods.military_levy
                    modifiers.law_enforcement = k_mods.law_enforcement

                    # Kingdom modifiers affect totals
                    modifiers.settlement_mood += k_mods.morale_modifier
                    modifiers.settlement_economy += k_mods.trade_bonus

                modifiers.kingdom_context = kingdom.get_settlement_context(settlement_id)

        # Calculate combined effects
        modifiers.total_mood_modifier = modifiers.settlement_mood
        modifiers.total_safety_modifier = modifiers.settlement_safety + (modifiers.law_enforcement - 0.5) * 0.3
        modifiers.total_economy_modifier = modifiers.settlement_economy - modifiers.tax_rate * 0.5

        return modifiers

    def get_settlement(self, settlement_id: UUID) -> Optional[SettlementAgent]:
        """Get a settlement by ID."""
        return self.settlements.get(settlement_id)

    def get_kingdom(self, faction_id: UUID) -> Optional[KingdomAgent]:
        """Get a kingdom by ID."""
        return self.kingdoms.get(faction_id)

    def get_agent_settlement(self, agent_id: UUID) -> Optional[SettlementAgent]:
        """Get the settlement an agent belongs to."""
        settlement_id = self.agent_to_settlement.get(agent_id)
        if settlement_id:
            return self.settlements.get(settlement_id)
        return None

    def get_agent_kingdom(self, agent_id: UUID) -> Optional[KingdomAgent]:
        """Get the kingdom an agent's settlement belongs to."""
        settlement_id = self.agent_to_settlement.get(agent_id)
        if settlement_id:
            kingdom_id = self.settlement_to_kingdom.get(settlement_id)
            if kingdom_id:
                return self.kingdoms.get(kingdom_id)
        return None

    def get_nearby_settlements(self, x: int, y: int, radius: int) -> List[SettlementAgent]:
        """Get settlements within a radius of a position."""
        nearby = []
        for settlement in self.settlements.values():
            dx = abs(settlement.x - x)
            dy = abs(settlement.y - y)
            if dx <= radius and dy <= radius:
                nearby.append(settlement)
        return nearby

    def get_stats(self) -> Dict[str, Any]:
        """Get collective manager statistics."""
        total_population = sum(s.population for s in self.settlements.values())
        total_residents = sum(len(s.resident_ids) for s in self.settlements.values())

        return {
            'settlements': {
                'count': len(self.settlements),
                'total_population': total_population,
                'tracked_residents': total_residents,
            },
            'kingdoms': {
                'count': len(self.kingdoms),
                'total_territory': sum(k.territory_size for k in self.kingdoms.values()),
            },
            'agent_mappings': len(self.agent_to_settlement),
            'last_settlement_tick': self.last_settlement_tick,
            'last_kingdom_tick': self.last_kingdom_tick,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all collective agent state."""
        return {
            'settlements': {
                str(sid): s.to_dict()
                for sid, s in self.settlements.items()
            },
            'kingdoms': {
                str(kid): k.to_dict()
                for kid, k in self.kingdoms.items()
            },
            'stats': self.get_stats(),
        }
