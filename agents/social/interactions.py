"""
Interaction System
Manages agent-to-agent interactions.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from agents.social.relationships import RelationshipManager, Relationship
from agents.memory.memory_stream import MemoryStream
from agents.reasoning.dialogue import DialogueSystem, Conversation


class InteractionType(str, Enum):
    """Types of interactions between agents"""
    GREETING = "greeting"
    CONVERSATION = "conversation"
    TRADE = "trade"
    COMBAT = "combat"
    COOPERATION = "cooperation"
    TEACHING = "teaching"
    OBSERVATION = "observation"
    GIFT = "gift"
    INSULT = "insult"
    HELP = "help"


@dataclass
class Interaction:
    """Record of an interaction between agents"""
    interaction_id: UUID = field(default_factory=uuid4)
    initiator_id: UUID = None
    target_id: UUID = None
    interaction_type: InteractionType = InteractionType.CONVERSATION
    description: str = ""
    sentiment: float = 0.0  # -1 to 1
    significance: float = 0.5  # 0 to 1
    location: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_minutes: float = 5.0
    outcome: Optional[str] = None
    witnesses: List[UUID] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "interaction_id": str(self.interaction_id),
            "initiator_id": str(self.initiator_id),
            "target_id": str(self.target_id),
            "interaction_type": self.interaction_type.value,
            "description": self.description,
            "sentiment": self.sentiment,
            "significance": self.significance,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "duration_minutes": self.duration_minutes,
            "outcome": self.outcome,
        }


class InteractionManager:
    """
    Manages interactions between agents.

    Handles:
    - Initiating interactions
    - Recording interaction history
    - Updating relationships based on interactions
    - Coordinating dialogue between agents
    """

    def __init__(
        self,
        relationship_manager: RelationshipManager,
    ):
        """
        Initialize interaction manager.

        Args:
            relationship_manager: Relationship tracking system
        """
        self.relationships = relationship_manager

        # Track active interactions
        self._active_interactions: Dict[UUID, Interaction] = {}

        # Interaction history
        self._history: List[Interaction] = []

        # Agent memory streams (for recording memories)
        self._memory_streams: Dict[UUID, MemoryStream] = {}

        # Dialogue systems
        self._dialogue_systems: Dict[UUID, DialogueSystem] = {}

    def register_agent(
        self,
        agent_id: UUID,
        memory_stream: MemoryStream,
        dialogue_system: Optional[DialogueSystem] = None,
    ) -> None:
        """Register an agent with the interaction system"""
        self._memory_streams[agent_id] = memory_stream
        if dialogue_system:
            self._dialogue_systems[agent_id] = dialogue_system

    async def initiate_interaction(
        self,
        initiator_id: UUID,
        target_id: UUID,
        interaction_type: InteractionType,
        location: Optional[str] = None,
    ) -> Interaction:
        """
        Start an interaction between two agents.

        Args:
            initiator_id: Agent starting the interaction
            target_id: Target agent
            interaction_type: Type of interaction
            location: Where it happens

        Returns:
            The interaction object
        """
        interaction = Interaction(
            initiator_id=initiator_id,
            target_id=target_id,
            interaction_type=interaction_type,
            location=location,
        )

        self._active_interactions[interaction.interaction_id] = interaction

        # Record initial observation for both agents
        await self._record_observation(
            initiator_id,
            f"Started {interaction_type.value} with {target_id}",
            location,
        )
        await self._record_observation(
            target_id,
            f"{initiator_id} initiated {interaction_type.value}",
            location,
        )

        return interaction

    async def complete_interaction(
        self,
        interaction_id: UUID,
        outcome: str,
        sentiment: float,
        significance: float = 0.5,
    ) -> Interaction:
        """
        Complete an ongoing interaction.

        Args:
            interaction_id: The interaction to complete
            outcome: What happened
            sentiment: How positive/negative (-1 to 1)
            significance: How important (0 to 1)

        Returns:
            Completed interaction
        """
        interaction = self._active_interactions.get(interaction_id)
        if not interaction:
            raise ValueError(f"Unknown interaction: {interaction_id}")

        # Update interaction
        interaction.outcome = outcome
        interaction.sentiment = sentiment
        interaction.significance = significance
        interaction.duration_minutes = (
            datetime.utcnow() - interaction.timestamp
        ).total_seconds() / 60.0

        # Update relationships
        self.relationships.record_interaction(
            interaction.initiator_id,
            interaction.target_id,
            sentiment,
            significance,
            memory_description=outcome,
        )

        # Record memories for both agents
        await self._record_observation(
            interaction.initiator_id,
            outcome,
            interaction.location,
            importance=significance,
        )
        await self._record_observation(
            interaction.target_id,
            outcome,
            interaction.location,
            importance=significance,
        )

        # Move to history
        del self._active_interactions[interaction_id]
        self._history.append(interaction)

        return interaction

    async def greet(
        self,
        greeter_id: UUID,
        greeter_name: str,
        target_id: UUID,
        target_name: str,
        location: Optional[str] = None,
    ) -> Interaction:
        """
        Have one agent greet another.

        Args:
            greeter_id: Agent doing the greeting
            greeter_name: Name of greeter
            target_id: Agent being greeted
            target_name: Name of target
            location: Where this happens

        Returns:
            Completed greeting interaction
        """
        interaction = await self.initiate_interaction(
            greeter_id,
            target_id,
            InteractionType.GREETING,
            location,
        )

        # Get relationship context
        relationship = self.relationships.get_or_create(greeter_id, target_id)
        disposition = relationship.get_disposition()

        # Determine greeting sentiment
        if disposition > 0.5:
            sentiment = 0.8
            outcome = f"{greeter_name} warmly greeted {target_name}"
        elif disposition > 0:
            sentiment = 0.5
            outcome = f"{greeter_name} greeted {target_name}"
        elif disposition > -0.3:
            sentiment = 0.2
            outcome = f"{greeter_name} nodded at {target_name}"
        else:
            sentiment = -0.2
            outcome = f"{greeter_name} coldly acknowledged {target_name}"

        return await self.complete_interaction(
            interaction.interaction_id,
            outcome,
            sentiment,
            significance=0.2,
        )

    async def start_conversation(
        self,
        initiator_id: UUID,
        initiator_name: str,
        target_id: UUID,
        target_name: str,
        topic: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Tuple[Interaction, Optional[Conversation]]:
        """
        Start a conversation between agents.

        Args:
            initiator_id: Agent starting conversation
            initiator_name: Name of initiator
            target_id: Target agent
            target_name: Name of target
            topic: Optional conversation topic
            location: Where this happens

        Returns:
            Tuple of (Interaction, Conversation)
        """
        interaction = await self.initiate_interaction(
            initiator_id,
            target_id,
            InteractionType.CONVERSATION,
            location,
        )

        interaction.description = f"Conversation about {topic or 'general matters'}"

        # Use dialogue system if available
        dialogue = self._dialogue_systems.get(initiator_id)
        conversation = None

        if dialogue:
            relationship = self.relationships.get_or_create(initiator_id, target_id)
            summary = f"Relationship: {relationship.describe(initiator_id)}"

            conversation, _ = await dialogue.start_conversation(
                other_agent_id=target_id,
                other_agent_name=target_name,
                agent_summary=summary,
                topic=topic,
                location=location,
            )

        return (interaction, conversation)

    async def help_agent(
        self,
        helper_id: UUID,
        helper_name: str,
        helpee_id: UUID,
        helpee_name: str,
        help_description: str,
        location: Optional[str] = None,
    ) -> Interaction:
        """
        Record one agent helping another.

        Args:
            helper_id: Agent providing help
            helper_name: Name of helper
            helpee_id: Agent receiving help
            helpee_name: Name of helpee
            help_description: What help was provided
            location: Where this happened

        Returns:
            Completed interaction
        """
        interaction = await self.initiate_interaction(
            helper_id,
            helpee_id,
            InteractionType.HELP,
            location,
        )

        outcome = f"{helper_name} helped {helpee_name}: {help_description}"

        return await self.complete_interaction(
            interaction.interaction_id,
            outcome,
            sentiment=0.7,
            significance=0.6,
        )

    async def record_conflict(
        self,
        attacker_id: UUID,
        attacker_name: str,
        defender_id: UUID,
        defender_name: str,
        outcome: str,
        winner_id: Optional[UUID] = None,
        location: Optional[str] = None,
    ) -> Interaction:
        """
        Record a conflict between agents.

        Args:
            attacker_id: Agent who initiated conflict
            attacker_name: Name of attacker
            defender_id: Agent who was attacked
            defender_name: Name of defender
            outcome: What happened
            winner_id: Who won (if applicable)
            location: Where this happened

        Returns:
            Completed interaction
        """
        interaction = await self.initiate_interaction(
            attacker_id,
            defender_id,
            InteractionType.COMBAT,
            location,
        )

        # Conflicts hurt relationships significantly
        return await self.complete_interaction(
            interaction.interaction_id,
            outcome,
            sentiment=-0.8,
            significance=0.9,
        )

    async def trade(
        self,
        trader_a_id: UUID,
        trader_a_name: str,
        trader_b_id: UUID,
        trader_b_name: str,
        trade_description: str,
        fair_trade: bool = True,
        location: Optional[str] = None,
    ) -> Interaction:
        """
        Record a trade between agents.

        Args:
            trader_a_id: First trader
            trader_a_name: Name of first trader
            trader_b_id: Second trader
            trader_b_name: Name of second trader
            trade_description: What was traded
            fair_trade: Whether trade was fair
            location: Where this happened

        Returns:
            Completed interaction
        """
        interaction = await self.initiate_interaction(
            trader_a_id,
            trader_b_id,
            InteractionType.TRADE,
            location,
        )

        outcome = f"{trader_a_name} and {trader_b_name} traded: {trade_description}"
        sentiment = 0.5 if fair_trade else -0.3

        return await self.complete_interaction(
            interaction.interaction_id,
            outcome,
            sentiment,
            significance=0.4,
        )

    def get_interaction_history(
        self,
        agent_id: UUID,
        limit: int = 50,
    ) -> List[Interaction]:
        """Get interaction history for an agent"""
        relevant = [
            i for i in self._history
            if i.initiator_id == agent_id or i.target_id == agent_id
        ]
        return relevant[-limit:]

    def get_recent_interactions_with(
        self,
        agent_id: UUID,
        other_id: UUID,
        limit: int = 10,
    ) -> List[Interaction]:
        """Get recent interactions between two specific agents"""
        relevant = [
            i for i in self._history
            if (i.initiator_id == agent_id and i.target_id == other_id) or
               (i.initiator_id == other_id and i.target_id == agent_id)
        ]
        return relevant[-limit:]

    async def _record_observation(
        self,
        agent_id: UUID,
        description: str,
        location: Optional[str] = None,
        importance: float = 0.4,
    ) -> None:
        """Record observation in agent's memory"""
        memory_stream = self._memory_streams.get(agent_id)
        if memory_stream:
            await memory_stream.add_observation(
                description=description,
                location=location,
                importance=importance,
            )

    def get_active_interactions(self) -> List[Interaction]:
        """Get all currently active interactions"""
        return list(self._active_interactions.values())

    def is_agent_busy(self, agent_id: UUID) -> bool:
        """Check if agent is in an active interaction"""
        for interaction in self._active_interactions.values():
            if interaction.initiator_id == agent_id or interaction.target_id == agent_id:
                return True
        return False


# Need to import this for type hint
from typing import Tuple
