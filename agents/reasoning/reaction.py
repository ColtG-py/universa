"""
Reaction System
Determines how agents respond to environmental changes.
Based on Stanford Generative Agents paper.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum

from agents.memory.memory_stream import MemoryStream
from agents.models.memory import Memory
from agents.config import MemoryType
from agents.llm.ollama_client import OllamaClient


class ReactionType(str, Enum):
    """Types of reactions an agent can have"""
    IGNORE = "ignore"          # Continue current activity
    ACKNOWLEDGE = "acknowledge" # Note but don't change behavior
    INTERRUPT = "interrupt"     # Stop current and respond
    URGENT = "urgent"          # Immediate priority response


@dataclass
class ReactionDecision:
    """Result of deciding how to react to a stimulus"""
    reaction_type: ReactionType
    should_react: bool
    priority: float  # 0-1, how urgent
    reasoning: str
    suggested_action: Optional[str] = None
    context_memories: List[UUID] = field(default_factory=list)


@dataclass
class EnvironmentChange:
    """Represents a change in the environment"""
    change_id: UUID
    description: str
    source_agent_id: Optional[UUID] = None
    source_agent_name: Optional[str] = None
    location: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    urgency: float = 0.5  # 0-1


class ReactionSystem:
    """
    Determines how agents react to environmental changes.

    From the Stanford paper:
    "When an agent perceives an event, it needs to decide whether
    to continue with its current plan or react to the event."

    The reaction decision considers:
    1. Relevance of the event to the agent
    2. Importance of current activity
    3. Relationship with source agent (if any)
    4. Past experiences with similar events
    """

    # Prompt for reaction decision
    REACTION_PROMPT = """{agent_summary}

{agent_name} is currently {current_activity}.

{agent_name} perceives: {event_description}

Given {agent_name}'s personality, current activity, and the nature of this event, should {agent_name} react to this?

Consider:
1. Is this relevant to {agent_name}?
2. Is the current activity more important?
3. What is {agent_name}'s relationship with anyone involved?

Response format:
React: [yes/no]
Priority: [low/medium/high/urgent]
Reasoning: [brief explanation]
Suggested action: [what should {agent_name} do if reacting]"""

    # Prompt for determining reaction type
    REACTION_TYPE_PROMPT = """{agent_summary}

{agent_name} has decided to react to: {event_description}

What type of reaction is appropriate?
- IGNORE: Note it but continue current activity
- ACKNOWLEDGE: Brief response, then continue
- INTERRUPT: Stop current activity and respond
- URGENT: Drop everything and respond immediately

Reaction type:"""

    def __init__(
        self,
        agent_id: UUID,
        agent_name: str,
        memory_stream: MemoryStream,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize reaction system.

        Args:
            agent_id: Agent's UUID
            agent_name: Agent's name
            memory_stream: Agent's memory stream
            ollama_client: LLM client
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.memory_stream = memory_stream
        self.client = ollama_client

        # Track recent reactions to avoid spam
        self._recent_reactions: List[Tuple[datetime, UUID]] = []
        self._reaction_cooldown_seconds = 30

    async def should_react(
        self,
        event: EnvironmentChange,
        current_activity: str,
        agent_summary: str,
    ) -> ReactionDecision:
        """
        Decide whether and how to react to an event.

        Args:
            event: The environmental change
            current_activity: What agent is currently doing
            agent_summary: Agent's description

        Returns:
            ReactionDecision with recommendation
        """
        # Check cooldown for this event source
        if event.source_agent_id and self._is_on_cooldown(event.source_agent_id):
            return ReactionDecision(
                reaction_type=ReactionType.IGNORE,
                should_react=False,
                priority=0.0,
                reasoning="Too many recent interactions with this agent"
            )

        # Get relevant memories for context
        relevant_memories = await self._get_relevant_memories(
            event.description,
            event.source_agent_name
        )

        if self.client:
            decision = await self._llm_decide(
                event, current_activity, agent_summary, relevant_memories
            )
        else:
            decision = self._heuristic_decide(
                event, current_activity, relevant_memories
            )

        # Add context memories
        decision.context_memories = [m.memory_id for m in relevant_memories]

        # Track this reaction
        if decision.should_react:
            self._record_reaction(event)

        return decision

    async def perceive(
        self,
        observation: str,
        source_agent_id: Optional[UUID] = None,
        source_agent_name: Optional[str] = None,
        location: Optional[str] = None,
    ) -> EnvironmentChange:
        """
        Create an environment change from a perception.

        Args:
            observation: What was perceived
            source_agent_id: ID of agent causing change
            source_agent_name: Name of agent causing change
            location: Where it happened

        Returns:
            EnvironmentChange object
        """
        from uuid import uuid4

        # Determine urgency heuristically
        urgency = self._estimate_urgency(observation)

        change = EnvironmentChange(
            change_id=uuid4(),
            description=observation,
            source_agent_id=source_agent_id,
            source_agent_name=source_agent_name,
            location=location,
            urgency=urgency
        )

        # Store as observation in memory
        await self.memory_stream.add_observation(
            description=observation,
            location=location
        )

        return change

    async def execute_reaction(
        self,
        decision: ReactionDecision,
        event: EnvironmentChange,
    ) -> str:
        """
        Execute the decided reaction.

        Args:
            decision: The reaction decision
            event: The triggering event

        Returns:
            Description of action taken
        """
        if not decision.should_react:
            return f"{self.agent_name} continues {self.agent_name}'s current activity."

        if decision.suggested_action:
            # Store reaction in memory
            await self.memory_stream.add_observation(
                description=f"Reacted to {event.description}: {decision.suggested_action}",
                importance=decision.priority
            )
            return decision.suggested_action

        # Default reactions by type
        if decision.reaction_type == ReactionType.ACKNOWLEDGE:
            action = f"{self.agent_name} acknowledges {event.description}"
        elif decision.reaction_type == ReactionType.INTERRUPT:
            action = f"{self.agent_name} stops to address {event.description}"
        elif decision.reaction_type == ReactionType.URGENT:
            action = f"{self.agent_name} immediately responds to {event.description}"
        else:
            action = f"{self.agent_name} notes {event.description}"

        await self.memory_stream.add_observation(
            description=action,
            importance=decision.priority
        )

        return action

    async def _get_relevant_memories(
        self,
        event_description: str,
        agent_name: Optional[str] = None
    ) -> List[Memory]:
        """Get memories relevant to the event"""
        memories = []

        # Search for event-related memories
        event_memories = await self.memory_stream.retrieve(
            query=event_description,
            limit=5,
            alpha_relevance=2.0
        )
        memories.extend(event_memories)

        # If another agent is involved, get relationship memories
        if agent_name:
            relationship_memories = await self.memory_stream.retrieve(
                query=agent_name,
                limit=5,
                alpha_relevance=1.5
            )
            memories.extend(relationship_memories)

        # Deduplicate
        seen_ids = set()
        unique_memories = []
        for m in memories:
            if m.memory_id not in seen_ids:
                seen_ids.add(m.memory_id)
                unique_memories.append(m)

        return unique_memories[:10]

    async def _llm_decide(
        self,
        event: EnvironmentChange,
        current_activity: str,
        agent_summary: str,
        relevant_memories: List[Memory]
    ) -> ReactionDecision:
        """Use LLM to decide reaction"""
        # Add memory context to prompt
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\nRelevant memories:\n" + "\n".join(
                f"- {m.description}" for m in relevant_memories[:5]
            )

        prompt = self.REACTION_PROMPT.format(
            agent_summary=agent_summary + memory_context,
            agent_name=self.agent_name,
            current_activity=current_activity,
            event_description=event.description
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200
            )

            return self._parse_decision(response.text, event)

        except Exception:
            return self._heuristic_decide(event, current_activity, relevant_memories)

    def _heuristic_decide(
        self,
        event: EnvironmentChange,
        current_activity: str,
        relevant_memories: List[Memory]
    ) -> ReactionDecision:
        """Decide reaction using heuristics"""
        desc_lower = event.description.lower()

        # Urgent keywords
        urgent_keywords = ["attack", "danger", "fire", "help", "emergency", "dying"]
        if any(kw in desc_lower for kw in urgent_keywords):
            return ReactionDecision(
                reaction_type=ReactionType.URGENT,
                should_react=True,
                priority=1.0,
                reasoning="Urgent situation detected",
                suggested_action=f"Immediately respond to {event.description}"
            )

        # Social keywords - interrupt for conversation
        social_keywords = ["greets", "says", "asks", "tells", "waves"]
        if any(kw in desc_lower for kw in social_keywords):
            # Check if we know this person
            knows_source = any(
                event.source_agent_name and event.source_agent_name.lower() in m.description.lower()
                for m in relevant_memories
            )

            if knows_source or event.source_agent_name:
                return ReactionDecision(
                    reaction_type=ReactionType.INTERRUPT,
                    should_react=True,
                    priority=0.7,
                    reasoning="Social interaction from known person",
                    suggested_action=f"Respond to {event.source_agent_name or 'the person'}"
                )
            else:
                return ReactionDecision(
                    reaction_type=ReactionType.ACKNOWLEDGE,
                    should_react=True,
                    priority=0.4,
                    reasoning="Social interaction from stranger",
                    suggested_action="Briefly acknowledge the interaction"
                )

        # Interest keywords - acknowledge but continue
        interest_keywords = ["interesting", "unusual", "strange", "new"]
        if any(kw in desc_lower for kw in interest_keywords):
            return ReactionDecision(
                reaction_type=ReactionType.ACKNOWLEDGE,
                should_react=True,
                priority=0.3,
                reasoning="Interesting but not urgent"
            )

        # Default: evaluate based on event urgency
        if event.urgency > 0.7:
            return ReactionDecision(
                reaction_type=ReactionType.INTERRUPT,
                should_react=True,
                priority=event.urgency,
                reasoning="High urgency event"
            )
        elif event.urgency > 0.4:
            return ReactionDecision(
                reaction_type=ReactionType.ACKNOWLEDGE,
                should_react=True,
                priority=event.urgency,
                reasoning="Moderate urgency"
            )
        else:
            return ReactionDecision(
                reaction_type=ReactionType.IGNORE,
                should_react=False,
                priority=event.urgency,
                reasoning="Low priority, continue current activity"
            )

    def _parse_decision(
        self,
        response: str,
        event: EnvironmentChange
    ) -> ReactionDecision:
        """Parse LLM response into decision"""
        lines = response.strip().split('\n')

        should_react = False
        priority = 0.5
        reasoning = ""
        suggested_action = None

        for line in lines:
            line_lower = line.lower()

            if line_lower.startswith("react:"):
                should_react = "yes" in line_lower

            elif line_lower.startswith("priority:"):
                if "urgent" in line_lower:
                    priority = 1.0
                elif "high" in line_lower:
                    priority = 0.8
                elif "medium" in line_lower:
                    priority = 0.5
                else:
                    priority = 0.3

            elif line_lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

            elif line_lower.startswith("suggested action:"):
                suggested_action = line.split(":", 1)[1].strip()

        # Determine reaction type from priority
        if priority >= 0.9:
            reaction_type = ReactionType.URGENT
        elif priority >= 0.6:
            reaction_type = ReactionType.INTERRUPT
        elif should_react:
            reaction_type = ReactionType.ACKNOWLEDGE
        else:
            reaction_type = ReactionType.IGNORE

        return ReactionDecision(
            reaction_type=reaction_type,
            should_react=should_react,
            priority=priority,
            reasoning=reasoning or "LLM decision",
            suggested_action=suggested_action
        )

    def _estimate_urgency(self, observation: str) -> float:
        """Estimate urgency of an observation"""
        obs_lower = observation.lower()

        # Urgency modifiers
        urgent_words = ["attack", "fire", "dying", "emergency", "help"]
        if any(w in obs_lower for w in urgent_words):
            return 0.95

        high_words = ["danger", "threat", "warning", "urgent"]
        if any(w in obs_lower for w in high_words):
            return 0.8

        social_words = ["greets", "says", "asks", "calls"]
        if any(w in obs_lower for w in social_words):
            return 0.6

        return 0.4  # Default moderate urgency

    def _is_on_cooldown(self, agent_id: UUID) -> bool:
        """Check if we've reacted to this agent recently"""
        now = datetime.utcnow()

        # Clean old entries
        self._recent_reactions = [
            (t, aid) for t, aid in self._recent_reactions
            if (now - t).total_seconds() < self._reaction_cooldown_seconds * 10
        ]

        # Check recent reactions to this agent
        recent_count = sum(
            1 for t, aid in self._recent_reactions
            if aid == agent_id and (now - t).total_seconds() < self._reaction_cooldown_seconds
        )

        return recent_count >= 3  # Max 3 reactions per cooldown period

    def _record_reaction(self, event: EnvironmentChange) -> None:
        """Record that we reacted to an event"""
        if event.source_agent_id:
            self._recent_reactions.append((datetime.utcnow(), event.source_agent_id))


class PerceptionFilter:
    """
    Filters what an agent perceives based on attention and context.
    """

    def __init__(
        self,
        attention_radius: float = 10.0,
        max_simultaneous: int = 5
    ):
        """
        Initialize filter.

        Args:
            attention_radius: How far agent can perceive
            max_simultaneous: Max events to process at once
        """
        self.attention_radius = attention_radius
        self.max_simultaneous = max_simultaneous

    def filter_events(
        self,
        events: List[EnvironmentChange],
        agent_location: Tuple[float, float],
        current_focus: Optional[str] = None
    ) -> List[EnvironmentChange]:
        """
        Filter events to those the agent would perceive.

        Args:
            events: All environmental changes
            agent_location: Agent's current position
            current_focus: What agent is focused on

        Returns:
            Filtered list of perceivable events
        """
        perceivable = []

        for event in events:
            # Check if in attention radius (simplified without location coords)
            # In production, would check actual distance

            # Prioritize events related to current focus
            if current_focus and current_focus.lower() in event.description.lower():
                event.urgency = min(1.0, event.urgency + 0.2)

            perceivable.append(event)

        # Sort by urgency and limit
        perceivable.sort(key=lambda e: e.urgency, reverse=True)
        return perceivable[:self.max_simultaneous]

    def should_notice(
        self,
        event: EnvironmentChange,
        agent_awareness: float = 0.5
    ) -> bool:
        """
        Probabilistically determine if agent notices an event.

        Args:
            event: The event
            agent_awareness: Agent's current awareness level (0-1)

        Returns:
            Whether agent notices the event
        """
        import random

        # Higher urgency = more likely to notice
        notice_chance = event.urgency * agent_awareness

        # Minimum chance for urgent events
        if event.urgency > 0.8:
            notice_chance = max(0.9, notice_chance)

        return random.random() < notice_chance
