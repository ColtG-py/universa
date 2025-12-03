"""
Information Diffusion System
Tracks how information spreads between agents.
"""

from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
import random


class InformationType(str, Enum):
    """Types of information that can spread"""
    RUMOR = "rumor"           # May or may not be true
    NEWS = "news"             # Factual events
    GOSSIP = "gossip"         # About other agents
    SECRET = "secret"         # Confidential information
    KNOWLEDGE = "knowledge"    # Skills, facts
    LOCATION = "location"     # Where something/someone is
    EVENT = "event"           # Something that happened


class InformationReliability(str, Enum):
    """How reliable the information is"""
    FIRSTHAND = "firsthand"    # Witnessed directly
    SECONDHAND = "secondhand"  # Heard from someone who witnessed
    RUMOR = "rumor"           # Passed through many people
    UNCERTAIN = "uncertain"    # Unknown reliability


@dataclass
class Information:
    """A piece of information that can spread"""
    info_id: UUID = field(default_factory=uuid4)
    info_type: InformationType = InformationType.NEWS
    content: str = ""
    subject: Optional[str] = None  # What/who this is about

    # Origin
    origin_agent_id: Optional[UUID] = None
    origin_location: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Truth and reliability
    is_true: bool = True
    reliability: InformationReliability = InformationReliability.FIRSTHAND
    distortion_count: int = 0  # How many times it's been passed on

    # Spread tracking
    importance: float = 0.5  # How important (affects spread rate)
    interest_tags: List[str] = field(default_factory=list)  # Who cares about this

    # Expiry
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if information has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def degrade(self) -> "Information":
        """
        Create a degraded copy (for passing along).

        Information degrades as it spreads.
        """
        degraded = Information(
            info_id=self.info_id,  # Same info, different knowledge
            info_type=self.info_type,
            content=self._distort_content(),
            subject=self.subject,
            origin_agent_id=self.origin_agent_id,
            origin_location=self.origin_location,
            created_at=self.created_at,
            is_true=self.is_true,  # Truth doesn't change
            reliability=self._degrade_reliability(),
            distortion_count=self.distortion_count + 1,
            importance=self.importance * 0.9,  # Slightly less important each time
            interest_tags=self.interest_tags.copy(),
            expires_at=self.expires_at,
        )
        return degraded

    def _distort_content(self) -> str:
        """Potentially distort content as it spreads"""
        # For now, content stays the same
        # Could add random distortions for rumors
        return self.content

    def _degrade_reliability(self) -> InformationReliability:
        """Degrade reliability based on spread"""
        if self.reliability == InformationReliability.FIRSTHAND:
            return InformationReliability.SECONDHAND
        elif self.reliability == InformationReliability.SECONDHAND:
            return InformationReliability.RUMOR
        return InformationReliability.UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "info_id": str(self.info_id),
            "info_type": self.info_type.value,
            "content": self.content,
            "subject": self.subject,
            "origin_agent_id": str(self.origin_agent_id) if self.origin_agent_id else None,
            "is_true": self.is_true,
            "reliability": self.reliability.value,
            "distortion_count": self.distortion_count,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AgentKnowledge:
    """Information an agent knows"""
    agent_id: UUID
    info_id: UUID
    info: Information
    learned_at: datetime = field(default_factory=datetime.utcnow)
    learned_from: Optional[UUID] = None  # Who told them
    has_shared: bool = False  # Have they told anyone
    belief_strength: float = 1.0  # How much they believe it (0-1)

    def should_share(self) -> bool:
        """Determine if agent should share this info"""
        if self.has_shared:
            return False

        # Higher importance = more likely to share
        share_chance = self.info.importance * 0.5

        # Secrets are less likely to be shared
        if self.info.info_type == InformationType.SECRET:
            share_chance *= 0.2

        return random.random() < share_chance


class InformationNetwork:
    """
    Manages information flow between agents.

    Tracks:
    - What each agent knows
    - How information spreads
    - Information reliability over time
    """

    def __init__(
        self,
        decay_rate: float = 0.1,  # How fast info loses importance
    ):
        """
        Initialize information network.

        Args:
            decay_rate: Daily importance decay rate
        """
        self.decay_rate = decay_rate

        # agent_id -> {info_id -> AgentKnowledge}
        self._knowledge: Dict[UUID, Dict[UUID, AgentKnowledge]] = {}

        # All information in the system
        self._all_info: Dict[UUID, Information] = {}

        # Track spread: info_id -> list of agent_ids who know
        self._spread_tracking: Dict[UUID, Set[UUID]] = {}

    def create_information(
        self,
        content: str,
        info_type: InformationType,
        origin_agent_id: UUID,
        subject: Optional[str] = None,
        importance: float = 0.5,
        interest_tags: Optional[List[str]] = None,
        is_true: bool = True,
        expires_hours: Optional[int] = None,
    ) -> Information:
        """
        Create new information in the network.

        Args:
            content: The information content
            info_type: Type of information
            origin_agent_id: Who generated this info
            subject: What/who it's about
            importance: How important (0-1)
            interest_tags: Who might care
            is_true: Is this true?
            expires_hours: When it expires (None = never)

        Returns:
            Created Information
        """
        expires_at = None
        if expires_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

        info = Information(
            info_type=info_type,
            content=content,
            subject=subject,
            origin_agent_id=origin_agent_id,
            importance=importance,
            interest_tags=interest_tags or [],
            is_true=is_true,
            expires_at=expires_at,
        )

        # Store in network
        self._all_info[info.info_id] = info
        self._spread_tracking[info.info_id] = set()

        # Origin agent knows this firsthand
        self.teach_agent(origin_agent_id, info)

        return info

    def teach_agent(
        self,
        agent_id: UUID,
        info: Information,
        source_agent_id: Optional[UUID] = None,
    ) -> bool:
        """
        Give an agent some information.

        Args:
            agent_id: Agent learning the info
            info: Information to learn
            source_agent_id: Who told them (None if firsthand)

        Returns:
            True if agent learned something new
        """
        if agent_id not in self._knowledge:
            self._knowledge[agent_id] = {}

        # Check if already knows
        if info.info_id in self._knowledge[agent_id]:
            return False

        knowledge = AgentKnowledge(
            agent_id=agent_id,
            info_id=info.info_id,
            info=info,
            learned_from=source_agent_id,
            belief_strength=1.0 if info.reliability == InformationReliability.FIRSTHAND else 0.8,
        )

        self._knowledge[agent_id][info.info_id] = knowledge
        self._spread_tracking[info.info_id].add(agent_id)

        return True

    def share_information(
        self,
        sender_id: UUID,
        receiver_id: UUID,
        info_id: Optional[UUID] = None,
    ) -> Optional[Information]:
        """
        Have one agent share information with another.

        Args:
            sender_id: Agent sharing
            receiver_id: Agent receiving
            info_id: Specific info to share (None = random shareable)

        Returns:
            Information shared, or None if nothing shared
        """
        sender_knowledge = self._knowledge.get(sender_id, {})

        if info_id:
            # Share specific info
            knowledge = sender_knowledge.get(info_id)
            if not knowledge:
                return None
        else:
            # Find something to share
            shareable = [
                k for k in sender_knowledge.values()
                if k.should_share() and not k.info.is_expired()
            ]

            if not shareable:
                return None

            # Pick most important
            knowledge = max(shareable, key=lambda k: k.info.importance)

        # Degrade the information as it spreads
        degraded_info = knowledge.info.degrade()

        # Teach receiver
        learned = self.teach_agent(receiver_id, degraded_info, sender_id)

        if learned:
            knowledge.has_shared = True
            return degraded_info

        return None

    def agent_knows(self, agent_id: UUID, info_id: UUID) -> bool:
        """Check if agent knows specific information"""
        return info_id in self._knowledge.get(agent_id, {})

    def get_agent_knowledge(
        self,
        agent_id: UUID,
        info_type: Optional[InformationType] = None,
    ) -> List[AgentKnowledge]:
        """Get all information an agent knows"""
        all_knowledge = list(self._knowledge.get(agent_id, {}).values())

        if info_type:
            all_knowledge = [k for k in all_knowledge if k.info.info_type == info_type]

        return all_knowledge

    def get_knowledge_about(
        self,
        agent_id: UUID,
        subject: str,
    ) -> List[AgentKnowledge]:
        """Get agent's knowledge about a specific subject"""
        all_knowledge = self._knowledge.get(agent_id, {})
        return [
            k for k in all_knowledge.values()
            if k.info.subject and subject.lower() in k.info.subject.lower()
        ]

    def get_spread_count(self, info_id: UUID) -> int:
        """Get number of agents who know this information"""
        return len(self._spread_tracking.get(info_id, set()))

    def get_who_knows(self, info_id: UUID) -> Set[UUID]:
        """Get set of agents who know this information"""
        return self._spread_tracking.get(info_id, set()).copy()

    def simulate_gossip_tick(
        self,
        agent_pairs: List[Tuple[UUID, UUID]],
        gossip_chance: float = 0.3,
    ) -> List[Tuple[UUID, UUID, Information]]:
        """
        Simulate gossip spreading in a tick.

        Args:
            agent_pairs: Pairs of agents who can gossip
            gossip_chance: Probability of gossip occurring

        Returns:
            List of (sender, receiver, info) for gossip that occurred
        """
        gossip_events = []

        for sender_id, receiver_id in agent_pairs:
            if random.random() > gossip_chance:
                continue

            info = self.share_information(sender_id, receiver_id)
            if info:
                gossip_events.append((sender_id, receiver_id, info))

        return gossip_events

    def decay_information(self, days: float = 1.0) -> int:
        """
        Apply time decay to all information.

        Args:
            days: Number of days passed

        Returns:
            Number of expired pieces of information
        """
        expired_count = 0
        to_remove = []

        for info_id, info in self._all_info.items():
            # Decay importance
            info.importance *= (1 - self.decay_rate * days)

            # Check expiry
            if info.is_expired():
                to_remove.append(info_id)
                expired_count += 1

        # Remove expired info
        for info_id in to_remove:
            del self._all_info[info_id]

            # Remove from agent knowledge
            for agent_knowledge in self._knowledge.values():
                agent_knowledge.pop(info_id, None)

            self._spread_tracking.pop(info_id, None)

        return expired_count

    def create_news_event(
        self,
        event_description: str,
        location: str,
        importance: float = 0.7,
        witness_agents: Optional[List[UUID]] = None,
    ) -> Information:
        """
        Create a news event that spreads.

        Args:
            event_description: What happened
            location: Where it happened
            importance: How important
            witness_agents: Agents who witnessed it

        Returns:
            Created information
        """
        # Use first witness as origin, or create without origin
        origin = witness_agents[0] if witness_agents else None

        info = Information(
            info_type=InformationType.NEWS,
            content=event_description,
            origin_agent_id=origin,
            origin_location=location,
            importance=importance,
            is_true=True,
            interest_tags=["news", location],
        )

        self._all_info[info.info_id] = info
        self._spread_tracking[info.info_id] = set()

        # All witnesses know firsthand
        if witness_agents:
            for agent_id in witness_agents:
                self.teach_agent(agent_id, info)

        return info

    def get_trending_info(self, limit: int = 10) -> List[Information]:
        """Get most widely known information"""
        sorted_info = sorted(
            self._all_info.values(),
            key=lambda i: len(self._spread_tracking.get(i.info_id, set())),
            reverse=True,
        )
        return sorted_info[:limit]

    def get_agent_summary(self, agent_id: UUID) -> Dict[str, Any]:
        """Get summary of agent's knowledge"""
        knowledge = self._knowledge.get(agent_id, {})

        by_type = {}
        for k in knowledge.values():
            t = k.info.info_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_knowledge": len(knowledge),
            "by_type": by_type,
            "unshared": sum(1 for k in knowledge.values() if not k.has_shared),
            "firsthand": sum(1 for k in knowledge.values() if k.info.reliability == InformationReliability.FIRSTHAND),
        }
