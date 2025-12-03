"""
Relationship System
Tracks and manages relationships between agents.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import math


class RelationshipType(str, Enum):
    """Types of relationships"""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    RIVAL = "rival"
    ENEMY = "enemy"
    FAMILY = "family"
    ROMANTIC = "romantic"
    MENTOR = "mentor"
    APPRENTICE = "apprentice"
    COLLEAGUE = "colleague"
    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass
class RelationshipMemory:
    """A memory associated with a relationship"""
    memory_id: UUID
    description: str
    sentiment: float  # -1 to 1, negative to positive
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Relationship:
    """
    A relationship between two agents.

    Tracks:
    - Familiarity: How well they know each other (0-1)
    - Trust: How much they trust each other (-1 to 1)
    - Affection: How they feel about each other (-1 to 1)
    - Respect: Professional/capability respect (-1 to 1)
    """
    relationship_id: UUID = field(default_factory=uuid4)
    agent_a_id: UUID = None
    agent_b_id: UUID = None

    # Core relationship values
    familiarity: float = 0.0  # 0 = stranger, 1 = knows everything
    trust: float = 0.0        # -1 = complete distrust, 1 = complete trust
    affection: float = 0.0    # -1 = hatred, 1 = love
    respect: float = 0.0      # -1 = contempt, 1 = admiration

    # Relationship metadata
    relationship_type: RelationshipType = RelationshipType.STRANGER
    first_met: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0

    # Shared history
    shared_memories: List[RelationshipMemory] = field(default_factory=list)

    # Special flags
    is_mutual: bool = True  # Relationship is symmetric
    is_hidden: bool = False  # One side doesn't know about the relationship

    def get_disposition(self) -> float:
        """
        Get overall disposition towards the other agent.

        Returns:
            Value from -1 (hostile) to 1 (friendly)
        """
        # Weighted average of relationship factors
        return (
            self.trust * 0.3 +
            self.affection * 0.4 +
            self.respect * 0.2 +
            self.familiarity * 0.1
        )

    def get_relationship_type(self) -> RelationshipType:
        """
        Determine relationship type based on current values.
        """
        disposition = self.get_disposition()

        if self.familiarity < 0.1:
            return RelationshipType.STRANGER

        if disposition < -0.6:
            return RelationshipType.ENEMY
        elif disposition < -0.2:
            return RelationshipType.RIVAL
        elif disposition < 0.2:
            return RelationshipType.ACQUAINTANCE
        elif disposition < 0.5:
            return RelationshipType.FRIEND
        elif disposition < 0.8:
            return RelationshipType.CLOSE_FRIEND
        else:
            # Check for special types
            if self.affection > 0.8:
                return RelationshipType.ROMANTIC
            return RelationshipType.CLOSE_FRIEND

    def update_from_interaction(
        self,
        sentiment: float,
        significance: float = 0.5,
    ) -> None:
        """
        Update relationship based on an interaction.

        Args:
            sentiment: How positive/negative the interaction was (-1 to 1)
            significance: How important the interaction was (0 to 1)
        """
        # Update familiarity (always increases with interaction)
        self.familiarity = min(1.0, self.familiarity + 0.02 * significance)

        # Update other values based on sentiment
        change = sentiment * significance * 0.1

        # Trust changes slowly
        self.trust = max(-1.0, min(1.0, self.trust + change * 0.5))

        # Affection changes more readily
        self.affection = max(-1.0, min(1.0, self.affection + change))

        # Respect based on positive achievements
        if sentiment > 0.5:
            self.respect = max(-1.0, min(1.0, self.respect + change * 0.3))

        self.interaction_count += 1
        self.last_interaction = datetime.utcnow()

        # Update type
        self.relationship_type = self.get_relationship_type()

    def add_memory(
        self,
        memory_id: UUID,
        description: str,
        sentiment: float,
    ) -> None:
        """Add a shared memory"""
        memory = RelationshipMemory(
            memory_id=memory_id,
            description=description,
            sentiment=sentiment,
        )
        self.shared_memories.append(memory)

        # Keep only recent memories
        if len(self.shared_memories) > 50:
            self.shared_memories = self.shared_memories[-50:]

    def decay(self, days_passed: float = 1.0) -> None:
        """
        Decay relationship over time without interaction.

        Args:
            days_passed: Number of days since last interaction
        """
        decay_rate = 0.01 * days_passed

        # Familiarity decays slowly
        self.familiarity = max(0.1, self.familiarity - decay_rate * 0.1)

        # Extreme values decay towards neutral
        if abs(self.trust) > 0.1:
            self.trust *= (1 - decay_rate * 0.05)

        if abs(self.affection) > 0.1:
            self.affection *= (1 - decay_rate * 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "relationship_id": str(self.relationship_id),
            "agent_a_id": str(self.agent_a_id),
            "agent_b_id": str(self.agent_b_id),
            "familiarity": self.familiarity,
            "trust": self.trust,
            "affection": self.affection,
            "respect": self.respect,
            "relationship_type": self.relationship_type.value,
            "first_met": self.first_met.isoformat() if self.first_met else None,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "interaction_count": self.interaction_count,
        }

    def describe(self, perspective_agent_id: UUID) -> str:
        """
        Get a natural language description of the relationship.

        Args:
            perspective_agent_id: From whose perspective

        Returns:
            Description string
        """
        disposition = self.get_disposition()
        rel_type = self.relationship_type.value

        if disposition > 0.6:
            feeling = "very positive feelings towards"
        elif disposition > 0.2:
            feeling = "positive feelings towards"
        elif disposition > -0.2:
            feeling = "neutral feelings towards"
        elif disposition > -0.6:
            feeling = "negative feelings towards"
        else:
            feeling = "very negative feelings towards"

        trust_desc = ""
        if self.trust > 0.5:
            trust_desc = " and trusts them deeply"
        elif self.trust < -0.5:
            trust_desc = " but does not trust them"

        return f"has {feeling} them ({rel_type}){trust_desc}"


class RelationshipManager:
    """
    Manages all relationships for agents.

    Handles:
    - Creating and tracking relationships
    - Updating relationships based on interactions
    - Querying relationship status
    - Relationship decay over time
    """

    def __init__(self):
        """Initialize relationship manager"""
        # agent_id -> {other_agent_id -> Relationship}
        self._relationships: Dict[UUID, Dict[UUID, Relationship]] = {}

    def get_or_create(
        self,
        agent_a_id: UUID,
        agent_b_id: UUID,
    ) -> Relationship:
        """
        Get existing relationship or create new one.

        Args:
            agent_a_id: First agent
            agent_b_id: Second agent

        Returns:
            Relationship between agents
        """
        # Ensure both directions exist
        if agent_a_id not in self._relationships:
            self._relationships[agent_a_id] = {}

        if agent_b_id not in self._relationships[agent_a_id]:
            relationship = Relationship(
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                first_met=datetime.utcnow(),
            )
            self._relationships[agent_a_id][agent_b_id] = relationship

            # Create symmetric relationship
            if agent_b_id not in self._relationships:
                self._relationships[agent_b_id] = {}
            self._relationships[agent_b_id][agent_a_id] = relationship

        return self._relationships[agent_a_id][agent_b_id]

    def get(
        self,
        agent_a_id: UUID,
        agent_b_id: UUID,
    ) -> Optional[Relationship]:
        """Get existing relationship or None"""
        if agent_a_id in self._relationships:
            return self._relationships[agent_a_id].get(agent_b_id)
        return None

    def record_interaction(
        self,
        agent_a_id: UUID,
        agent_b_id: UUID,
        sentiment: float,
        significance: float = 0.5,
        memory_description: Optional[str] = None,
    ) -> Relationship:
        """
        Record an interaction between agents.

        Args:
            agent_a_id: First agent
            agent_b_id: Second agent
            sentiment: How positive/negative (-1 to 1)
            significance: How important (0 to 1)
            memory_description: Optional memory to record

        Returns:
            Updated relationship
        """
        relationship = self.get_or_create(agent_a_id, agent_b_id)
        relationship.update_from_interaction(sentiment, significance)

        if memory_description:
            relationship.add_memory(
                memory_id=uuid4(),
                description=memory_description,
                sentiment=sentiment,
            )

        return relationship

    def get_all_relationships(
        self,
        agent_id: UUID,
    ) -> List[Relationship]:
        """Get all relationships for an agent"""
        if agent_id not in self._relationships:
            return []
        return list(self._relationships[agent_id].values())

    def get_friends(
        self,
        agent_id: UUID,
        min_disposition: float = 0.3,
    ) -> List[Tuple[UUID, Relationship]]:
        """Get agents with positive relationships"""
        friends = []

        for other_id, rel in self._relationships.get(agent_id, {}).items():
            if rel.get_disposition() >= min_disposition:
                friends.append((other_id, rel))

        # Sort by disposition
        friends.sort(key=lambda x: x[1].get_disposition(), reverse=True)
        return friends

    def get_enemies(
        self,
        agent_id: UUID,
        max_disposition: float = -0.3,
    ) -> List[Tuple[UUID, Relationship]]:
        """Get agents with negative relationships"""
        enemies = []

        for other_id, rel in self._relationships.get(agent_id, {}).items():
            if rel.get_disposition() <= max_disposition:
                enemies.append((other_id, rel))

        # Sort by disposition (most negative first)
        enemies.sort(key=lambda x: x[1].get_disposition())
        return enemies

    def get_relationship_context(
        self,
        agent_id: UUID,
        other_id: UUID,
    ) -> str:
        """
        Get context about a relationship for LLM prompts.

        Returns:
            Natural language description
        """
        rel = self.get(agent_id, other_id)
        if not rel:
            return "has never met this person"

        return rel.describe(agent_id)

    def decay_all(self, days_passed: float = 1.0) -> None:
        """Apply time decay to all relationships"""
        seen = set()

        for agent_id, relationships in self._relationships.items():
            for other_id, rel in relationships.items():
                # Only decay once per relationship
                pair = tuple(sorted([str(agent_id), str(other_id)]))
                if pair in seen:
                    continue
                seen.add(pair)

                rel.decay(days_passed)

    def get_mutual_friends(
        self,
        agent_a_id: UUID,
        agent_b_id: UUID,
    ) -> List[UUID]:
        """Find agents that both agents have positive relationships with"""
        a_friends = {uid for uid, _ in self.get_friends(agent_a_id)}
        b_friends = {uid for uid, _ in self.get_friends(agent_b_id)}

        return list(a_friends & b_friends)

    def get_social_network(
        self,
        agent_id: UUID,
        depth: int = 2,
    ) -> Dict[UUID, int]:
        """
        Get agent's extended social network.

        Args:
            agent_id: Starting agent
            depth: How many hops to include

        Returns:
            Dict of agent_id -> distance from starting agent
        """
        network = {agent_id: 0}
        frontier = [agent_id]

        for d in range(1, depth + 1):
            next_frontier = []

            for current_id in frontier:
                friends = self.get_friends(current_id, min_disposition=0.2)

                for friend_id, _ in friends:
                    if friend_id not in network:
                        network[friend_id] = d
                        next_frontier.append(friend_id)

            frontier = next_frontier

        return network

    def to_database_format(self) -> List[Dict[str, Any]]:
        """Convert all relationships to database format"""
        seen = set()
        result = []

        for agent_id, relationships in self._relationships.items():
            for other_id, rel in relationships.items():
                # Only include once per pair
                pair = tuple(sorted([str(agent_id), str(other_id)]))
                if pair in seen:
                    continue
                seen.add(pair)

                result.append(rel.to_dict())

        return result
