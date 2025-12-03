"""
Kingdom Agent
A collective agent representing a faction/kingdom that makes political
and military decisions affecting all settlements and subjects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Types of kingdom policies."""
    TAXATION = "taxation"
    MILITARY = "military"
    TRADE = "trade"
    DIPLOMACY = "diplomacy"
    RELIGION = "religion"
    LAW = "law"


class PolicyLevel(Enum):
    """Intensity levels for policies."""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    EXTREME = 5


class DiplomaticRelation(Enum):
    """Relations between kingdoms."""
    WAR = "war"
    HOSTILE = "hostile"
    COLD = "cold"
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ALLIED = "allied"
    VASSAL = "vassal"
    OVERLORD = "overlord"


class KingdomEventType(Enum):
    """Kingdom-level events."""
    WAR_DECLARED = "war_declared"
    PEACE_TREATY = "peace_treaty"
    TRADE_AGREEMENT = "trade_agreement"
    TAX_CHANGE = "tax_change"
    MILITARY_LEVY = "military_levy"
    ROYAL_DECREE = "royal_decree"
    SUCCESSION_CRISIS = "succession_crisis"
    RELIGIOUS_EDICT = "religious_edict"


@dataclass
class KingdomPolicy:
    """A policy enacted by a kingdom."""
    policy_type: PolicyType
    level: PolicyLevel
    description: str
    enacted_tick: int
    duration_ticks: Optional[int] = None  # None = permanent
    affects_settlements: bool = True
    affects_subjects: bool = True


@dataclass
class KingdomEvent:
    """An event at the kingdom level."""
    event_id: UUID
    event_type: KingdomEventType
    description: str
    tick: int
    target_faction_id: Optional[UUID] = None
    is_resolved: bool = False


@dataclass
class KingdomModifiers:
    """Modifiers applied to all settlements in a kingdom."""
    tax_rate: float = 0.1          # 0-0.5, affects prosperity
    military_levy: float = 0.0     # 0-0.3, affects available labor
    trade_bonus: float = 0.0       # -0.2 to +0.3
    morale_modifier: float = 0.0   # -0.3 to +0.3
    law_enforcement: float = 0.5   # 0-1, affects safety


class KingdomAgent:
    """
    A collective agent representing a kingdom/faction.

    Makes high-level decisions that cascade down to settlements:
    - Taxation policies
    - Military mobilization
    - Diplomatic relations
    - Trade agreements
    - Religious/cultural policies
    """

    def __init__(
        self,
        faction_id: UUID,
        name: str,
        faction_type: str,
        capital_settlement_id: Optional[UUID] = None
    ):
        self.faction_id = faction_id
        self.name = name
        self.faction_type = faction_type
        self.capital_settlement_id = capital_settlement_id

        # Territory
        self.settlement_ids: Set[UUID] = set()
        self.territory_size: int = 0

        # Power metrics
        self.power_level: float = 0.5    # 0-1, overall strength
        self.military_strength: float = 0.5
        self.economic_strength: float = 0.5
        self.stability: float = 0.7      # 0-1, internal stability

        # Policies
        self.active_policies: Dict[PolicyType, KingdomPolicy] = {}
        self._initialize_default_policies()

        # Diplomacy
        self.relations: Dict[UUID, DiplomaticRelation] = {}

        # Events
        self.active_events: List[KingdomEvent] = []
        self.event_history: List[KingdomEvent] = []

        # Tick tracking
        self.last_tick = 0
        self.ticks_since_decision = 0
        self.decision_interval = 500  # Make decisions every 500 ticks

    def _initialize_default_policies(self):
        """Set up default policies."""
        self.active_policies = {
            PolicyType.TAXATION: KingdomPolicy(
                policy_type=PolicyType.TAXATION,
                level=PolicyLevel.MODERATE,
                description="Standard taxation on all subjects",
                enacted_tick=0
            ),
            PolicyType.MILITARY: KingdomPolicy(
                policy_type=PolicyType.MILITARY,
                level=PolicyLevel.MODERATE,
                description="Standing army maintained",
                enacted_tick=0
            ),
            PolicyType.LAW: KingdomPolicy(
                policy_type=PolicyType.LAW,
                level=PolicyLevel.MODERATE,
                description="Standard laws enforced",
                enacted_tick=0
            ),
        }

    def add_settlement(self, settlement_id: UUID):
        """Add a settlement to this kingdom."""
        self.settlement_ids.add(settlement_id)
        self.territory_size = len(self.settlement_ids)
        self._recalculate_power()

    def remove_settlement(self, settlement_id: UUID):
        """Remove a settlement from this kingdom."""
        self.settlement_ids.discard(settlement_id)
        self.territory_size = len(self.settlement_ids)
        self._recalculate_power()

    def set_relation(self, other_faction_id: UUID, relation: DiplomaticRelation):
        """Set diplomatic relation with another faction."""
        self.relations[other_faction_id] = relation

    def _recalculate_power(self):
        """Recalculate power level based on territory and policies."""
        # Base power from territory
        territory_power = min(1.0, self.territory_size / 50)

        # Military policy affects military strength
        military_policy = self.active_policies.get(PolicyType.MILITARY)
        if military_policy:
            self.military_strength = military_policy.level.value / 5

        # Economic strength from taxation (moderate is best)
        tax_policy = self.active_policies.get(PolicyType.TAXATION)
        if tax_policy:
            # Too high or too low taxation hurts economy
            tax_level = tax_policy.level.value
            self.economic_strength = 1.0 - abs(tax_level - 3) * 0.2

        # Overall power
        self.power_level = (
            territory_power * 0.3 +
            self.military_strength * 0.3 +
            self.economic_strength * 0.3 +
            self.stability * 0.1
        )

    async def kingdom_tick(self, current_tick: int) -> KingdomModifiers:
        """
        Kingdom-level thinking. Runs very infrequently.

        Args:
            current_tick: Current simulation tick

        Returns:
            Modifiers to apply to all settlements
        """
        self.last_tick = current_tick
        self.ticks_since_decision += 1

        # Check if it's time to make a major decision
        if self.ticks_since_decision >= self.decision_interval:
            await self._make_kingdom_decision(current_tick)
            self.ticks_since_decision = 0

        # Update stability based on policies and events
        self._update_stability()

        # Update active events
        self._update_events(current_tick)

        # Build modifiers
        return self._build_modifiers()

    async def _make_kingdom_decision(self, current_tick: int):
        """
        Make a major kingdom-level decision.
        This is where we could integrate LLM for complex decisions.
        """
        # For now, use rule-based decisions

        # Check if we should change taxation
        if self.economic_strength < 0.3:
            # Economy is weak, lower taxes
            self._adjust_policy(
                PolicyType.TAXATION,
                PolicyLevel.LOW,
                current_tick,
                "Taxes reduced to stimulate economy"
            )
        elif self.economic_strength > 0.7 and self.military_strength < 0.5:
            # Economy is strong but military weak, raise taxes for military
            self._adjust_policy(
                PolicyType.TAXATION,
                PolicyLevel.HIGH,
                current_tick,
                "Taxes raised to fund military"
            )

        # Check military needs
        at_war = any(
            rel == DiplomaticRelation.WAR
            for rel in self.relations.values()
        )
        if at_war and self.military_strength < 0.6:
            # At war with weak military, mobilize
            self._adjust_policy(
                PolicyType.MILITARY,
                PolicyLevel.HIGH,
                current_tick,
                "Military mobilization for war"
            )

        # Check stability
        if self.stability < 0.3:
            # Low stability, increase law enforcement
            self._adjust_policy(
                PolicyType.LAW,
                PolicyLevel.HIGH,
                current_tick,
                "Increased law enforcement to restore order"
            )

        logger.info(f"Kingdom {self.name} made decisions at tick {current_tick}")

    def _adjust_policy(
        self,
        policy_type: PolicyType,
        new_level: PolicyLevel,
        tick: int,
        description: str
    ):
        """Adjust a policy level."""
        old_policy = self.active_policies.get(policy_type)
        if old_policy and old_policy.level != new_level:
            self.active_policies[policy_type] = KingdomPolicy(
                policy_type=policy_type,
                level=new_level,
                description=description,
                enacted_tick=tick
            )
            logger.debug(f"Kingdom {self.name} changed {policy_type.value} to {new_level.value}")

    def _update_stability(self):
        """Update internal stability."""
        base_stability = 0.7

        # High taxes hurt stability
        tax_policy = self.active_policies.get(PolicyType.TAXATION)
        if tax_policy and tax_policy.level.value >= 4:
            base_stability -= 0.15

        # High military levy hurts stability
        military_policy = self.active_policies.get(PolicyType.MILITARY)
        if military_policy and military_policy.level.value >= 4:
            base_stability -= 0.1

        # War hurts stability
        at_war = any(
            rel == DiplomaticRelation.WAR
            for rel in self.relations.values()
        )
        if at_war:
            base_stability -= 0.2

        # Strong economy helps stability
        base_stability += (self.economic_strength - 0.5) * 0.2

        self.stability = max(0.0, min(1.0, base_stability))

    def _update_events(self, current_tick: int):
        """Update and resolve events."""
        # Events are typically resolved through the decision-making process
        # For now, just age out old events
        active = []
        for event in self.active_events:
            # Events last 1000 ticks unless resolved
            if current_tick - event.tick < 1000 and not event.is_resolved:
                active.append(event)
        self.active_events = active

    def _build_modifiers(self) -> KingdomModifiers:
        """Build modifiers to apply to settlements."""
        modifiers = KingdomModifiers()

        # Tax rate from taxation policy
        tax_policy = self.active_policies.get(PolicyType.TAXATION)
        if tax_policy:
            modifiers.tax_rate = tax_policy.level.value * 0.08  # 0.08 to 0.40

        # Military levy from military policy
        military_policy = self.active_policies.get(PolicyType.MILITARY)
        if military_policy and military_policy.level.value >= 4:
            modifiers.military_levy = (military_policy.level.value - 3) * 0.1

        # Trade bonus from trade agreements
        friendly_count = sum(
            1 for rel in self.relations.values()
            if rel in (DiplomaticRelation.FRIENDLY, DiplomaticRelation.ALLIED)
        )
        modifiers.trade_bonus = min(0.3, friendly_count * 0.05)

        # War penalty
        at_war = any(
            rel == DiplomaticRelation.WAR
            for rel in self.relations.values()
        )
        if at_war:
            modifiers.trade_bonus -= 0.2
            modifiers.morale_modifier -= 0.1

        # Law enforcement from law policy
        law_policy = self.active_policies.get(PolicyType.LAW)
        if law_policy:
            modifiers.law_enforcement = law_policy.level.value * 0.18

        # Stability affects morale
        modifiers.morale_modifier += (self.stability - 0.5) * 0.2

        return modifiers

    def declare_war(self, target_faction_id: UUID, tick: int) -> KingdomEvent:
        """Declare war on another faction."""
        self.relations[target_faction_id] = DiplomaticRelation.WAR

        event = KingdomEvent(
            event_id=uuid4(),
            event_type=KingdomEventType.WAR_DECLARED,
            description=f"{self.name} has declared war!",
            tick=tick,
            target_faction_id=target_faction_id
        )
        self.active_events.append(event)
        self.event_history.append(event)

        logger.info(f"Kingdom {self.name} declared war on faction {target_faction_id}")
        return event

    def make_peace(self, target_faction_id: UUID, tick: int) -> KingdomEvent:
        """Make peace with another faction."""
        self.relations[target_faction_id] = DiplomaticRelation.NEUTRAL

        event = KingdomEvent(
            event_id=uuid4(),
            event_type=KingdomEventType.PEACE_TREATY,
            description=f"{self.name} has signed a peace treaty.",
            tick=tick,
            target_faction_id=target_faction_id
        )
        self.active_events.append(event)
        self.event_history.append(event)

        logger.info(f"Kingdom {self.name} made peace with faction {target_faction_id}")
        return event

    def get_settlement_context(self, settlement_id: UUID) -> Dict[str, Any]:
        """Get kingdom context for a settlement."""
        modifiers = self._build_modifiers()

        return {
            'faction_id': str(self.faction_id),
            'faction_name': self.name,
            'faction_type': self.faction_type,
            'is_capital': settlement_id == self.capital_settlement_id,
            'tax_rate': modifiers.tax_rate,
            'military_levy': modifiers.military_levy,
            'at_war': any(rel == DiplomaticRelation.WAR for rel in self.relations.values()),
            'stability': self.stability,
            'power_level': self.power_level,
            'active_policies': [
                {
                    'type': p.policy_type.value,
                    'level': p.level.name,
                    'description': p.description
                }
                for p in self.active_policies.values()
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize kingdom agent state."""
        return {
            'faction_id': str(self.faction_id),
            'name': self.name,
            'faction_type': self.faction_type,
            'capital_settlement_id': str(self.capital_settlement_id) if self.capital_settlement_id else None,
            'territory_size': self.territory_size,
            'power_level': self.power_level,
            'military_strength': self.military_strength,
            'economic_strength': self.economic_strength,
            'stability': self.stability,
            'settlement_count': len(self.settlement_ids),
            'relations': {
                str(k): v.value for k, v in self.relations.items()
            },
            'active_policies': {
                p.policy_type.value: {
                    'level': p.level.name,
                    'description': p.description
                }
                for p in self.active_policies.values()
            },
        }
