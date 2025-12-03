"""
Settlement Agent
A collective agent that represents a settlement and makes decisions
affecting all residents without requiring individual LLM calls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)


class SettlementMood(Enum):
    """Overall mood of a settlement."""
    PROSPEROUS = "prosperous"      # 0.8+
    CONTENT = "content"            # 0.6-0.8
    NEUTRAL = "neutral"            # 0.4-0.6
    UNEASY = "uneasy"             # 0.2-0.4
    DISTRESSED = "distressed"      # 0.0-0.2


class SettlementEventType(Enum):
    """Types of events a settlement can generate."""
    FESTIVAL = "festival"
    MARKET_DAY = "market_day"
    PATROL_INCREASED = "patrol_increased"
    RESOURCE_SHORTAGE = "resource_shortage"
    VISITOR_ARRIVAL = "visitor_arrival"
    WEATHER_WARNING = "weather_warning"
    CRIME_WAVE = "crime_wave"
    CELEBRATION = "celebration"
    MOURNING = "mourning"
    CONSTRUCTION = "construction"
    HARVEST = "harvest"
    RELIGIOUS_CEREMONY = "religious_ceremony"


@dataclass
class SettlementEvent:
    """An event occurring at a settlement."""
    event_id: UUID
    event_type: SettlementEventType
    description: str
    start_tick: int
    duration_ticks: int
    affects_mood: float  # -0.2 to +0.2
    affects_safety: float  # -0.2 to +0.2
    affects_economy: float  # -0.2 to +0.2
    involved_residents: Set[UUID] = field(default_factory=set)
    is_active: bool = True


@dataclass
class SettlementModifiers:
    """
    Modifiers applied to all residents of a settlement.
    These affect agent behavior without individual LLM calls.
    """
    mood_modifier: float = 0.0      # -0.5 to +0.5, affects agent happiness
    safety_modifier: float = 0.0    # -0.5 to +0.5, affects anxiety/danger perception
    economy_modifier: float = 0.0   # -0.5 to +0.5, affects trade/work availability
    social_modifier: float = 0.0    # -0.5 to +0.5, affects social interaction frequency

    # Activity suggestions based on settlement state
    encouraged_activities: List[str] = field(default_factory=list)
    discouraged_activities: List[str] = field(default_factory=list)

    # Events residents should be aware of
    active_events: List[SettlementEvent] = field(default_factory=list)

    # Resource availability (affects what agents can do)
    available_resources: Dict[str, float] = field(default_factory=dict)


@dataclass
class SettlementResources:
    """Resources available in a settlement."""
    food: float = 1.0          # 0-1, multiplier for food availability
    water: float = 1.0
    trade_goods: float = 1.0
    raw_materials: float = 1.0
    labor: float = 1.0         # Available workforce
    security: float = 1.0      # Guard presence


class SettlementAgent:
    """
    A collective agent representing a settlement.

    Makes settlement-level decisions that affect all residents:
    - Generates events (festivals, market days, etc.)
    - Tracks aggregate mood and safety
    - Manages resource availability
    - Provides context to resident agents
    """

    def __init__(
        self,
        settlement_id: UUID,
        name: str,
        settlement_type: str,
        x: int,
        y: int,
        population: int = 100,
        faction_id: Optional[UUID] = None
    ):
        self.settlement_id = settlement_id
        self.name = name
        self.settlement_type = settlement_type
        self.x = x
        self.y = y
        self.population = population
        self.faction_id = faction_id

        # Resident tracking
        self.resident_ids: Set[UUID] = set()
        self._resident_moods: Dict[UUID, float] = {}  # Cached moods

        # Settlement state
        self.mood = 0.5  # 0-1, aggregate happiness
        self.safety = 0.7  # 0-1, how safe the settlement is
        self.prosperity = 0.5  # 0-1, economic health
        self.resources = SettlementResources()

        # Events
        self.active_events: List[SettlementEvent] = []
        self.event_history: List[SettlementEvent] = []

        # Tick tracking
        self.last_tick = 0
        self.ticks_since_event = 0

        # Configuration
        self.event_chance_per_tick = 0.05  # 5% chance per collective tick
        self.min_ticks_between_events = 10

    def add_resident(self, agent_id: UUID, initial_mood: float = 0.5):
        """Add a resident to this settlement."""
        self.resident_ids.add(agent_id)
        self._resident_moods[agent_id] = initial_mood
        self.population = len(self.resident_ids)

    def remove_resident(self, agent_id: UUID):
        """Remove a resident from this settlement."""
        self.resident_ids.discard(agent_id)
        self._resident_moods.pop(agent_id, None)
        self.population = len(self.resident_ids)

    def update_resident_mood(self, agent_id: UUID, mood: float):
        """Update cached mood for a resident."""
        if agent_id in self.resident_ids:
            self._resident_moods[agent_id] = max(0.0, min(1.0, mood))

    async def collective_tick(self, current_tick: int, game_hour: int) -> SettlementModifiers:
        """
        Execute settlement-level thinking.
        Runs less frequently than individual agents.

        Args:
            current_tick: Current simulation tick
            game_hour: Current hour in game time (0-23)

        Returns:
            Modifiers to apply to resident agents
        """
        self.last_tick = current_tick
        self.ticks_since_event += 1

        # 1. Update aggregate mood from residents
        self._update_aggregate_mood()

        # 2. Update safety based on various factors
        self._update_safety(game_hour)

        # 3. Update resources
        self._update_resources()

        # 4. Check for event generation
        if self._should_generate_event():
            event = self._generate_event(current_tick, game_hour)
            if event:
                self.active_events.append(event)
                self.event_history.append(event)
                self.ticks_since_event = 0
                logger.info(f"Settlement {self.name} generated event: {event.event_type.value}")

        # 5. Update active events
        self._update_active_events(current_tick)

        # 6. Build modifiers for residents
        return self._build_modifiers()

    def _update_aggregate_mood(self):
        """Calculate aggregate mood from resident moods."""
        if self._resident_moods:
            self.mood = sum(self._resident_moods.values()) / len(self._resident_moods)
        else:
            # Base mood affected by prosperity and safety
            self.mood = (self.prosperity + self.safety) / 2

        # Apply event effects
        for event in self.active_events:
            self.mood += event.affects_mood

        self.mood = max(0.0, min(1.0, self.mood))

    def _update_safety(self, game_hour: int):
        """Update safety level based on time and events."""
        base_safety = 0.7

        # Night is less safe
        if game_hour < 6 or game_hour > 21:
            base_safety -= 0.1

        # Settlement type affects safety
        type_modifiers = {
            'metropolis': 0.1,
            'city': 0.05,
            'town': 0.0,
            'village': -0.05,
            'hamlet': -0.1,
            'fortress': 0.2,
        }
        base_safety += type_modifiers.get(self.settlement_type.lower(), 0)

        # Events affect safety
        for event in self.active_events:
            base_safety += event.affects_safety

        self.safety = max(0.0, min(1.0, base_safety))

    def _update_resources(self):
        """Update resource availability."""
        # Base resource availability
        base = 0.8

        # Prosperity affects all resources
        prosperity_factor = 0.5 + self.prosperity * 0.5

        self.resources.food = base * prosperity_factor
        self.resources.water = base + 0.1  # Usually stable
        self.resources.trade_goods = base * prosperity_factor
        self.resources.raw_materials = base * prosperity_factor
        self.resources.labor = min(1.0, self.population / 100)
        self.resources.security = self.safety

        # Events affect resources
        for event in self.active_events:
            if event.affects_economy != 0:
                factor = 1.0 + event.affects_economy
                self.resources.food *= factor
                self.resources.trade_goods *= factor

    def _should_generate_event(self) -> bool:
        """Check if we should generate a new event."""
        if self.ticks_since_event < self.min_ticks_between_events:
            return False
        return random.random() < self.event_chance_per_tick

    def _generate_event(self, current_tick: int, game_hour: int) -> Optional[SettlementEvent]:
        """Generate a settlement event based on current state."""
        # Weight event types based on settlement state
        event_weights = self._get_event_weights(game_hour)

        if not event_weights:
            return None

        # Select event type
        total_weight = sum(event_weights.values())
        roll = random.random() * total_weight

        selected_type = None
        cumulative = 0
        for event_type, weight in event_weights.items():
            cumulative += weight
            if roll <= cumulative:
                selected_type = event_type
                break

        if not selected_type:
            return None

        # Generate event details
        return self._create_event(selected_type, current_tick)

    def _get_event_weights(self, game_hour: int) -> Dict[SettlementEventType, float]:
        """Get weighted probabilities for event types."""
        weights = {}

        # Positive events more likely when prosperous
        if self.prosperity > 0.6:
            weights[SettlementEventType.FESTIVAL] = 0.1
            weights[SettlementEventType.CELEBRATION] = 0.1
            weights[SettlementEventType.MARKET_DAY] = 0.2

        # Market day during daytime
        if 8 <= game_hour <= 18:
            weights[SettlementEventType.MARKET_DAY] = weights.get(SettlementEventType.MARKET_DAY, 0) + 0.15

        # Visitor arrivals
        weights[SettlementEventType.VISITOR_ARRIVAL] = 0.15

        # Negative events more likely when struggling
        if self.prosperity < 0.4:
            weights[SettlementEventType.RESOURCE_SHORTAGE] = 0.2

        if self.safety < 0.5:
            weights[SettlementEventType.PATROL_INCREASED] = 0.2
            weights[SettlementEventType.CRIME_WAVE] = 0.1

        # Religious ceremonies
        weights[SettlementEventType.RELIGIOUS_CEREMONY] = 0.05

        # Construction in growing settlements
        if self.prosperity > 0.5:
            weights[SettlementEventType.CONSTRUCTION] = 0.1

        return weights

    def _create_event(self, event_type: SettlementEventType, current_tick: int) -> SettlementEvent:
        """Create an event of the given type."""
        event_configs = {
            SettlementEventType.FESTIVAL: {
                'description': f"A festival is being held in {self.name}!",
                'duration': 50,
                'mood': 0.15,
                'safety': 0.0,
                'economy': 0.1,
            },
            SettlementEventType.MARKET_DAY: {
                'description': f"It's market day in {self.name}.",
                'duration': 20,
                'mood': 0.05,
                'safety': 0.0,
                'economy': 0.15,
            },
            SettlementEventType.PATROL_INCREASED: {
                'description': f"Guards are patrolling {self.name} more frequently.",
                'duration': 30,
                'mood': -0.05,
                'safety': 0.15,
                'economy': 0.0,
            },
            SettlementEventType.RESOURCE_SHORTAGE: {
                'description': f"{self.name} is experiencing a resource shortage.",
                'duration': 40,
                'mood': -0.15,
                'safety': -0.05,
                'economy': -0.2,
            },
            SettlementEventType.VISITOR_ARRIVAL: {
                'description': f"Travelers have arrived in {self.name}.",
                'duration': 15,
                'mood': 0.05,
                'safety': 0.0,
                'economy': 0.05,
            },
            SettlementEventType.CRIME_WAVE: {
                'description': f"Crime has increased in {self.name}.",
                'duration': 35,
                'mood': -0.1,
                'safety': -0.2,
                'economy': -0.1,
            },
            SettlementEventType.CELEBRATION: {
                'description': f"The people of {self.name} are celebrating!",
                'duration': 25,
                'mood': 0.2,
                'safety': 0.0,
                'economy': 0.05,
            },
            SettlementEventType.RELIGIOUS_CEREMONY: {
                'description': f"A religious ceremony is taking place in {self.name}.",
                'duration': 15,
                'mood': 0.1,
                'safety': 0.05,
                'economy': 0.0,
            },
            SettlementEventType.CONSTRUCTION: {
                'description': f"New construction is underway in {self.name}.",
                'duration': 100,
                'mood': 0.05,
                'safety': 0.0,
                'economy': 0.1,
            },
        }

        config = event_configs.get(event_type, {
            'description': f"Something is happening in {self.name}.",
            'duration': 20,
            'mood': 0.0,
            'safety': 0.0,
            'economy': 0.0,
        })

        return SettlementEvent(
            event_id=uuid4(),
            event_type=event_type,
            description=config['description'],
            start_tick=current_tick,
            duration_ticks=config['duration'],
            affects_mood=config['mood'],
            affects_safety=config['safety'],
            affects_economy=config['economy'],
        )

    def _update_active_events(self, current_tick: int):
        """Update active events, removing expired ones."""
        still_active = []
        for event in self.active_events:
            if current_tick - event.start_tick < event.duration_ticks:
                still_active.append(event)
            else:
                event.is_active = False
                logger.debug(f"Event ended in {self.name}: {event.event_type.value}")
        self.active_events = still_active

    def _build_modifiers(self) -> SettlementModifiers:
        """Build modifiers to apply to resident agents."""
        modifiers = SettlementModifiers(
            mood_modifier=self.mood - 0.5,  # Center around 0
            safety_modifier=self.safety - 0.5,
            economy_modifier=self.prosperity - 0.5,
            social_modifier=0.0,
            active_events=self.active_events.copy(),
            available_resources={
                'food': self.resources.food,
                'water': self.resources.water,
                'trade_goods': self.resources.trade_goods,
                'raw_materials': self.resources.raw_materials,
            }
        )

        # Determine encouraged/discouraged activities
        if self.mood > 0.7:
            modifiers.encouraged_activities.extend(['socialize', 'celebrate', 'trade'])
        elif self.mood < 0.3:
            modifiers.discouraged_activities.extend(['celebrate', 'party'])

        if self.safety < 0.4:
            modifiers.discouraged_activities.extend(['wander', 'travel'])
            modifiers.encouraged_activities.append('stay_home')

        # Event-specific activities
        for event in self.active_events:
            if event.event_type == SettlementEventType.MARKET_DAY:
                modifiers.encouraged_activities.append('trade')
                modifiers.encouraged_activities.append('shop')
            elif event.event_type == SettlementEventType.FESTIVAL:
                modifiers.encouraged_activities.append('celebrate')
                modifiers.encouraged_activities.append('socialize')
            elif event.event_type == SettlementEventType.RELIGIOUS_CEREMONY:
                modifiers.encouraged_activities.append('pray')
                modifiers.encouraged_activities.append('attend_ceremony')

        return modifiers

    def get_resident_context(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Get settlement context for a specific resident's decision-making.
        Injected into the agent's perception.
        """
        return {
            'settlement_id': str(self.settlement_id),
            'settlement_name': self.name,
            'settlement_type': self.settlement_type,
            'settlement_mood': self._get_mood_description(),
            'safety_level': self._get_safety_description(),
            'prosperity': self.prosperity,
            'population': self.population,
            'active_events': [
                {
                    'type': e.event_type.value,
                    'description': e.description
                }
                for e in self.active_events
            ],
            'available_resources': {
                'food': 'abundant' if self.resources.food > 0.7 else 'normal' if self.resources.food > 0.4 else 'scarce',
                'water': 'abundant' if self.resources.water > 0.7 else 'normal' if self.resources.water > 0.4 else 'scarce',
            }
        }

    def _get_mood_description(self) -> str:
        """Get human-readable mood description."""
        if self.mood >= 0.8:
            return SettlementMood.PROSPEROUS.value
        elif self.mood >= 0.6:
            return SettlementMood.CONTENT.value
        elif self.mood >= 0.4:
            return SettlementMood.NEUTRAL.value
        elif self.mood >= 0.2:
            return SettlementMood.UNEASY.value
        else:
            return SettlementMood.DISTRESSED.value

    def _get_safety_description(self) -> str:
        """Get human-readable safety description."""
        if self.safety >= 0.8:
            return "very safe"
        elif self.safety >= 0.6:
            return "safe"
        elif self.safety >= 0.4:
            return "somewhat safe"
        elif self.safety >= 0.2:
            return "dangerous"
        else:
            return "very dangerous"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize settlement agent state."""
        return {
            'settlement_id': str(self.settlement_id),
            'name': self.name,
            'settlement_type': self.settlement_type,
            'x': self.x,
            'y': self.y,
            'population': self.population,
            'faction_id': str(self.faction_id) if self.faction_id else None,
            'mood': self.mood,
            'safety': self.safety,
            'prosperity': self.prosperity,
            'resident_count': len(self.resident_ids),
            'active_events': [
                {
                    'event_id': str(e.event_id),
                    'type': e.event_type.value,
                    'description': e.description,
                }
                for e in self.active_events
            ],
        }
