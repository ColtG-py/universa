"""
Event System
World events, broadcasting, and historical tracking.
"""

from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import random


class EventType(str, Enum):
    """Types of world events"""
    # Natural
    WEATHER = "weather"
    NATURAL_DISASTER = "natural_disaster"
    SEASONAL = "seasonal"

    # Social
    GATHERING = "gathering"
    CELEBRATION = "celebration"
    CONFLICT = "conflict"
    TRADE_FAIR = "trade_fair"

    # Political
    POLITICAL = "political"
    WAR = "war"
    ALLIANCE = "alliance"
    SUCCESSION = "succession"

    # Economic
    ECONOMIC = "economic"
    SHORTAGE = "shortage"
    PROSPERITY = "prosperity"

    # Supernatural
    MAGICAL = "magical"
    OMEN = "omen"
    DIVINE = "divine"

    # Agent-driven
    ACHIEVEMENT = "achievement"
    DEATH = "death"
    BIRTH = "birth"
    DISCOVERY = "discovery"


class EventScope(str, Enum):
    """Scope of event impact"""
    LOCAL = "local"           # Single location
    REGIONAL = "regional"     # Multiple nearby locations
    FACTION = "faction"       # Entire faction
    WORLD = "world"           # Everyone


@dataclass
class WorldEvent:
    """A significant event in the world"""
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.POLITICAL
    scope: EventScope = EventScope.LOCAL

    # Content
    title: str = ""
    description: str = ""
    location: Optional[str] = None

    # Participants
    involved_agents: List[UUID] = field(default_factory=list)
    involved_factions: List[str] = field(default_factory=list)

    # Timing
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    duration_hours: float = 1.0
    is_ongoing: bool = False

    # Impact
    importance: float = 0.5  # 0-1
    effects: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    witnesses: Set[UUID] = field(default_factory=set)
    reactions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "scope": self.scope.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "involved_agents": [str(a) for a in self.involved_agents],
            "involved_factions": self.involved_factions,
            "occurred_at": self.occurred_at.isoformat(),
            "importance": self.importance,
            "is_ongoing": self.is_ongoing,
        }

    def get_news_text(self) -> str:
        """Get event as news text"""
        return f"{self.title}: {self.description}"


class EventSystem:
    """
    Manages world events.

    Features:
    - Event creation and tracking
    - Broadcasting to agents
    - Historical record
    - Emergent event generation
    """

    def __init__(self):
        """Initialize event system"""
        # Active events
        self._active_events: Dict[UUID, WorldEvent] = {}

        # Event history
        self._history: List[WorldEvent] = []
        self._max_history = 1000

        # Subscribers: scope -> list of (agent_id, callback)
        self._subscribers: Dict[EventScope, List[tuple]] = {
            scope: [] for scope in EventScope
        }

        # Location subscribers
        self._location_subscribers: Dict[str, List[tuple]] = {}

        # Event generation rules
        self._generation_rules: List[Callable] = []

    def create_event(
        self,
        event_type: EventType,
        title: str,
        description: str,
        scope: EventScope = EventScope.LOCAL,
        location: Optional[str] = None,
        importance: float = 0.5,
        involved_agents: Optional[List[UUID]] = None,
        involved_factions: Optional[List[str]] = None,
        duration_hours: float = 1.0,
        effects: Optional[Dict[str, Any]] = None,
    ) -> WorldEvent:
        """
        Create and broadcast a new event.

        Args:
            event_type: Type of event
            title: Short title
            description: Full description
            scope: How far the event reaches
            location: Where it happened
            importance: How important (0-1)
            involved_agents: Agents involved
            involved_factions: Factions involved
            duration_hours: How long event lasts
            effects: Event effects

        Returns:
            Created event
        """
        event = WorldEvent(
            event_type=event_type,
            scope=scope,
            title=title,
            description=description,
            location=location,
            importance=importance,
            involved_agents=involved_agents or [],
            involved_factions=involved_factions or [],
            duration_hours=duration_hours,
            effects=effects or {},
            is_ongoing=duration_hours > 0,
        )

        # Track active events
        if event.is_ongoing:
            self._active_events[event.event_id] = event

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Broadcast to subscribers
        self._broadcast(event)

        return event

    def subscribe(
        self,
        agent_id: UUID,
        callback: Callable[[WorldEvent], None],
        scope: EventScope = EventScope.LOCAL,
        location: Optional[str] = None,
    ) -> None:
        """
        Subscribe an agent to events.

        Args:
            agent_id: Agent subscribing
            callback: Function to call on event
            scope: Minimum scope to receive
            location: Specific location to watch
        """
        self._subscribers[scope].append((agent_id, callback))

        if location:
            if location not in self._location_subscribers:
                self._location_subscribers[location] = []
            self._location_subscribers[location].append((agent_id, callback))

    def unsubscribe(self, agent_id: UUID) -> None:
        """Remove all subscriptions for an agent"""
        for scope in self._subscribers:
            self._subscribers[scope] = [
                (aid, cb) for aid, cb in self._subscribers[scope]
                if aid != agent_id
            ]

        for location in self._location_subscribers:
            self._location_subscribers[location] = [
                (aid, cb) for aid, cb in self._location_subscribers[location]
                if aid != agent_id
            ]

    def _broadcast(self, event: WorldEvent) -> None:
        """Broadcast event to subscribers"""
        notified = set()

        # Scope-based notification
        for scope in EventScope:
            if self._should_notify_scope(event.scope, scope):
                for agent_id, callback in self._subscribers[scope]:
                    if agent_id not in notified:
                        try:
                            callback(event)
                            event.witnesses.add(agent_id)
                            notified.add(agent_id)
                        except Exception:
                            pass

        # Location-based notification
        if event.location and event.location in self._location_subscribers:
            for agent_id, callback in self._location_subscribers[event.location]:
                if agent_id not in notified:
                    try:
                        callback(event)
                        event.witnesses.add(agent_id)
                        notified.add(agent_id)
                    except Exception:
                        pass

    def _should_notify_scope(
        self,
        event_scope: EventScope,
        subscriber_scope: EventScope
    ) -> bool:
        """Check if subscriber should be notified based on scope"""
        scope_order = [EventScope.LOCAL, EventScope.REGIONAL, EventScope.FACTION, EventScope.WORLD]
        event_idx = scope_order.index(event_scope)
        sub_idx = scope_order.index(subscriber_scope)
        return event_idx >= sub_idx

    def complete_event(self, event_id: UUID) -> Optional[WorldEvent]:
        """Mark an ongoing event as complete"""
        event = self._active_events.pop(event_id, None)
        if event:
            event.is_ongoing = False
        return event

    def get_active_events(
        self,
        location: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> List[WorldEvent]:
        """Get currently active events"""
        events = list(self._active_events.values())

        if location:
            events = [e for e in events if e.location == location]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    def get_recent_events(
        self,
        limit: int = 50,
        event_type: Optional[EventType] = None,
        min_importance: float = 0.0,
    ) -> List[WorldEvent]:
        """Get recent historical events"""
        events = self._history[-limit:]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if min_importance > 0:
            events = [e for e in events if e.importance >= min_importance]

        return events

    def get_events_involving(self, agent_id: UUID) -> List[WorldEvent]:
        """Get events involving a specific agent"""
        return [
            e for e in self._history
            if agent_id in e.involved_agents
        ]

    def record_reaction(
        self,
        event_id: UUID,
        agent_id: UUID,
        reaction: str,
    ) -> None:
        """Record an agent's reaction to an event"""
        event = self._active_events.get(event_id)
        if not event:
            # Check history
            for e in reversed(self._history):
                if e.event_id == event_id:
                    event = e
                    break

        if event:
            event.reactions.append({
                "agent_id": str(agent_id),
                "reaction": reaction,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def generate_random_event(
        self,
        location: Optional[str] = None,
        allowed_types: Optional[List[EventType]] = None,
    ) -> Optional[WorldEvent]:
        """
        Generate a random event.

        Args:
            location: Where to generate event
            allowed_types: Types to choose from

        Returns:
            Generated event or None
        """
        event_templates = {
            EventType.WEATHER: [
                ("Storm Approaching", "Dark clouds gather on the horizon."),
                ("Clear Skies", "The weather is pleasant and clear."),
                ("Fog Rolls In", "A thick fog blankets the area."),
            ],
            EventType.GATHERING: [
                ("Market Day", "Merchants gather to sell their wares."),
                ("Town Meeting", "Citizens gather to discuss local matters."),
                ("Festival", "A celebration begins in the town."),
            ],
            EventType.ECONOMIC: [
                ("Price Increase", "The cost of goods has risen."),
                ("Good Harvest", "Farmers report excellent yields."),
                ("Trade Caravan", "A merchant caravan arrives."),
            ],
            EventType.POLITICAL: [
                ("New Decree", "The local lord issues a new proclamation."),
                ("Diplomatic Visit", "An envoy arrives from another region."),
                ("Tax Collection", "Tax collectors are making rounds."),
            ],
        }

        types = allowed_types or list(event_templates.keys())
        available = [t for t in types if t in event_templates]

        if not available:
            return None

        chosen_type = random.choice(available)
        title, description = random.choice(event_templates[chosen_type])

        return self.create_event(
            event_type=chosen_type,
            title=title,
            description=description,
            location=location,
            scope=EventScope.LOCAL,
            importance=random.uniform(0.3, 0.7),
        )

    def add_generation_rule(
        self,
        rule: Callable[["EventSystem", Dict[str, Any]], Optional[WorldEvent]]
    ) -> None:
        """Add a rule for generating emergent events"""
        self._generation_rules.append(rule)

    def check_generation_rules(
        self,
        context: Dict[str, Any]
    ) -> List[WorldEvent]:
        """Run all generation rules and return any created events"""
        events = []

        for rule in self._generation_rules:
            try:
                event = rule(self, context)
                if event:
                    events.append(event)
            except Exception:
                pass

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics"""
        by_type = {}
        for event in self._history:
            t = event.event_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_events": len(self._history),
            "active_events": len(self._active_events),
            "by_type": by_type,
        }
