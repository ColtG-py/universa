"""
Dialogue System
Generates contextual dialogue for agent interactions.
Based on Stanford Generative Agents paper.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from agents.memory.memory_stream import MemoryStream
from agents.models.memory import Memory
from agents.llm.ollama_client import OllamaClient


class ConversationTone(str, Enum):
    """Tone of conversation"""
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    FORMAL = "formal"
    HOSTILE = "hostile"
    CAUTIOUS = "cautious"
    INTIMATE = "intimate"


class DialogueIntent(str, Enum):
    """Intent behind a dialogue"""
    GREET = "greet"
    FAREWELL = "farewell"
    INFORM = "inform"
    ASK = "ask"
    REQUEST = "request"
    AGREE = "agree"
    DISAGREE = "disagree"
    THANK = "thank"
    APOLOGIZE = "apologize"
    GOSSIP = "gossip"
    NEGOTIATE = "negotiate"


@dataclass
class DialogueTurn:
    """A single turn in a conversation"""
    turn_id: UUID = field(default_factory=uuid4)
    speaker_id: UUID = None
    speaker_name: str = ""
    content: str = ""
    intent: DialogueIntent = DialogueIntent.INFORM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    emotional_tone: Optional[str] = None


@dataclass
class Conversation:
    """A complete conversation between agents"""
    conversation_id: UUID = field(default_factory=uuid4)
    participants: List[UUID] = field(default_factory=list)
    participant_names: List[str] = field(default_factory=list)
    turns: List[DialogueTurn] = field(default_factory=list)
    topic: Optional[str] = None
    location: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    is_active: bool = True

    def add_turn(self, turn: DialogueTurn) -> None:
        """Add a turn to the conversation"""
        self.turns.append(turn)

    def get_transcript(self, last_n: Optional[int] = None) -> str:
        """Get conversation as text"""
        turns = self.turns[-last_n:] if last_n else self.turns
        return "\n".join(
            f"{t.speaker_name}: {t.content}"
            for t in turns
        )

    def end(self) -> None:
        """End the conversation"""
        self.is_active = False
        self.ended_at = datetime.utcnow()


class DialogueSystem:
    """
    Generates contextual dialogue for agent interactions.

    From the Stanford paper:
    "Dialogue is generated based on the agents' memories of each other,
    their current context, and their ongoing plans."

    Dialogue considers:
    1. Relationship history between speakers
    2. Current emotional state
    3. Topic relevance to each speaker
    4. Speaking style based on personality
    """

    # Dialogue generation prompt
    DIALOGUE_PROMPT = """{speaker_summary}

{speaker_name} is talking to {listener_name}.
Location: {location}
Topic: {topic}

What {speaker_name} knows about {listener_name}:
{relationship_context}

Recent conversation:
{conversation_history}

What does {speaker_name} say next? Respond in character, in first person.
Keep the response natural and brief (1-3 sentences).

{speaker_name} says:"""

    # Greeting prompt
    GREETING_PROMPT = """{speaker_summary}

{speaker_name} encounters {listener_name} at {location}.

What {speaker_name} knows about {listener_name}:
{relationship_context}

How does {speaker_name} greet {listener_name}? Be natural and in character.

{speaker_name} says:"""

    # Response generation prompt
    RESPONSE_PROMPT = """{speaker_summary}

{listener_name} just said: "{last_utterance}"

Context:
{conversation_history}

What {speaker_name} knows about {listener_name}:
{relationship_context}

How does {speaker_name} respond? Be natural and in character (1-3 sentences).

{speaker_name} says:"""

    def __init__(
        self,
        agent_id: UUID,
        agent_name: str,
        memory_stream: MemoryStream,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize dialogue system.

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

        # Active conversations
        self._active_conversations: Dict[UUID, Conversation] = {}

    async def start_conversation(
        self,
        other_agent_id: UUID,
        other_agent_name: str,
        agent_summary: str,
        topic: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Tuple[Conversation, DialogueTurn]:
        """
        Start a new conversation with another agent.

        Args:
            other_agent_id: Other agent's ID
            other_agent_name: Other agent's name
            agent_summary: This agent's summary
            topic: Conversation topic
            location: Where conversation happens

        Returns:
            Tuple of (Conversation, opening DialogueTurn)
        """
        # Create conversation
        conversation = Conversation(
            participants=[self.agent_id, other_agent_id],
            participant_names=[self.agent_name, other_agent_name],
            topic=topic,
            location=location or "somewhere"
        )

        # Get relationship context
        relationship = await self._get_relationship_context(other_agent_name)

        # Generate greeting
        if self.client:
            greeting = await self._generate_greeting(
                agent_summary, other_agent_name, relationship, location or "here"
            )
        else:
            greeting = self._heuristic_greeting(other_agent_name, relationship)

        # Create opening turn
        turn = DialogueTurn(
            speaker_id=self.agent_id,
            speaker_name=self.agent_name,
            content=greeting,
            intent=DialogueIntent.GREET
        )

        conversation.add_turn(turn)
        self._active_conversations[conversation.conversation_id] = conversation

        # Store in memory
        await self.memory_stream.add_observation(
            description=f"Started conversation with {other_agent_name}: '{greeting}'",
            location=location
        )

        return conversation, turn

    async def respond(
        self,
        conversation: Conversation,
        last_utterance: str,
        speaker_name: str,
        agent_summary: str,
    ) -> DialogueTurn:
        """
        Generate a response in an ongoing conversation.

        Args:
            conversation: The conversation
            last_utterance: What the other agent just said
            speaker_name: Who spoke last
            agent_summary: This agent's summary

        Returns:
            Response DialogueTurn
        """
        # Get relationship context
        relationship = await self._get_relationship_context(speaker_name)

        # Get conversation history
        history = conversation.get_transcript(last_n=5)

        if self.client:
            response = await self._generate_response(
                agent_summary,
                speaker_name,
                last_utterance,
                history,
                relationship
            )
        else:
            response = self._heuristic_response(last_utterance, speaker_name)

        # Determine intent
        intent = self._classify_intent(response)

        turn = DialogueTurn(
            speaker_id=self.agent_id,
            speaker_name=self.agent_name,
            content=response,
            intent=intent
        )

        conversation.add_turn(turn)

        # Store in memory
        await self.memory_stream.add_observation(
            description=f"Said to {speaker_name}: '{response}'",
            location=conversation.location
        )

        return turn

    async def generate_dialogue(
        self,
        conversation: Conversation,
        agent_summary: str,
        topic: Optional[str] = None,
    ) -> DialogueTurn:
        """
        Generate the next dialogue turn on a topic.

        Args:
            conversation: The conversation
            agent_summary: This agent's summary
            topic: What to talk about

        Returns:
            DialogueTurn
        """
        # Find the other participant
        other_name = next(
            (name for name in conversation.participant_names
             if name != self.agent_name),
            "them"
        )

        relationship = await self._get_relationship_context(other_name)
        history = conversation.get_transcript(last_n=5)

        if self.client:
            content = await self._generate_dialogue(
                agent_summary,
                other_name,
                topic or conversation.topic or "general matters",
                history,
                relationship,
                conversation.location or "here"
            )
        else:
            content = self._heuristic_dialogue(topic, other_name)

        intent = self._classify_intent(content)

        turn = DialogueTurn(
            speaker_id=self.agent_id,
            speaker_name=self.agent_name,
            content=content,
            intent=intent
        )

        conversation.add_turn(turn)

        return turn

    async def end_conversation(
        self,
        conversation: Conversation,
        agent_summary: str,
    ) -> DialogueTurn:
        """
        Generate a farewell and end conversation.

        Args:
            conversation: The conversation
            agent_summary: This agent's summary

        Returns:
            Farewell DialogueTurn
        """
        other_name = next(
            (name for name in conversation.participant_names
             if name != self.agent_name),
            "them"
        )

        relationship = await self._get_relationship_context(other_name)

        if self.client:
            farewell = await self._generate_farewell(
                agent_summary, other_name, relationship, conversation
            )
        else:
            farewell = self._heuristic_farewell(other_name)

        turn = DialogueTurn(
            speaker_id=self.agent_id,
            speaker_name=self.agent_name,
            content=farewell,
            intent=DialogueIntent.FAREWELL
        )

        conversation.add_turn(turn)
        conversation.end()

        # Remove from active
        if conversation.conversation_id in self._active_conversations:
            del self._active_conversations[conversation.conversation_id]

        # Store summary in memory
        summary = self._summarize_conversation(conversation)
        await self.memory_stream.add_observation(
            description=f"Had conversation with {other_name}: {summary}",
            importance=0.6,
            location=conversation.location
        )

        return turn

    async def _get_relationship_context(self, other_name: str) -> str:
        """Get memories about the other agent"""
        memories = await self.memory_stream.retrieve(
            query=other_name,
            limit=10,
            alpha_relevance=1.5
        )

        if not memories:
            return f"I don't know much about {other_name}."

        context_lines = [m.description for m in memories[:5]]
        return "\n".join(f"- {line}" for line in context_lines)

    async def _generate_greeting(
        self,
        agent_summary: str,
        other_name: str,
        relationship: str,
        location: str
    ) -> str:
        """Generate a greeting using LLM"""
        prompt = self.GREETING_PROMPT.format(
            speaker_summary=agent_summary,
            speaker_name=self.agent_name,
            listener_name=other_name,
            location=location,
            relationship_context=relationship
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=100
            )
            return self._clean_dialogue(response.text)
        except Exception:
            return self._heuristic_greeting(other_name, relationship)

    async def _generate_response(
        self,
        agent_summary: str,
        speaker_name: str,
        last_utterance: str,
        history: str,
        relationship: str
    ) -> str:
        """Generate a response using LLM"""
        prompt = self.RESPONSE_PROMPT.format(
            speaker_summary=agent_summary,
            speaker_name=self.agent_name,
            listener_name=speaker_name,
            last_utterance=last_utterance,
            conversation_history=history,
            relationship_context=relationship
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=150
            )
            return self._clean_dialogue(response.text)
        except Exception:
            return self._heuristic_response(last_utterance, speaker_name)

    async def _generate_dialogue(
        self,
        agent_summary: str,
        other_name: str,
        topic: str,
        history: str,
        relationship: str,
        location: str
    ) -> str:
        """Generate dialogue on a topic"""
        prompt = self.DIALOGUE_PROMPT.format(
            speaker_summary=agent_summary,
            speaker_name=self.agent_name,
            listener_name=other_name,
            location=location,
            topic=topic,
            relationship_context=relationship,
            conversation_history=history or "Just started talking."
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=150
            )
            return self._clean_dialogue(response.text)
        except Exception:
            return self._heuristic_dialogue(topic, other_name)

    async def _generate_farewell(
        self,
        agent_summary: str,
        other_name: str,
        relationship: str,
        conversation: Conversation
    ) -> str:
        """Generate a farewell"""
        prompt = f"""{agent_summary}

{self.agent_name} has been talking to {other_name} about {conversation.topic or 'various things'}.
The conversation is ending.

What {self.agent_name} knows about {other_name}:
{relationship}

How does {self.agent_name} say goodbye? Be natural and in character.

{self.agent_name} says:"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=80
            )
            return self._clean_dialogue(response.text)
        except Exception:
            return self._heuristic_farewell(other_name)

    def _clean_dialogue(self, text: str) -> str:
        """Clean up LLM dialogue output"""
        # Remove any speaker prefixes the LLM might have added
        text = text.strip()

        if text.startswith(f"{self.agent_name}:"):
            text = text[len(self.agent_name) + 1:].strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Ensure not too long
        if len(text) > 300:
            # Find a good break point
            sentences = text.split('.')
            text = '. '.join(sentences[:3]) + '.'

        return text.strip()

    def _classify_intent(self, content: str) -> DialogueIntent:
        """Classify the intent of dialogue"""
        content_lower = content.lower()

        # Question detection
        if '?' in content:
            return DialogueIntent.ASK

        # Greeting detection
        greet_words = ["hello", "hi ", "good morning", "good evening", "greetings"]
        if any(w in content_lower for w in greet_words):
            return DialogueIntent.GREET

        # Farewell detection
        bye_words = ["goodbye", "bye", "farewell", "see you", "take care"]
        if any(w in content_lower for w in bye_words):
            return DialogueIntent.FAREWELL

        # Thank detection
        if "thank" in content_lower:
            return DialogueIntent.THANK

        # Apology detection
        if "sorry" in content_lower or "apologize" in content_lower:
            return DialogueIntent.APOLOGIZE

        # Request detection
        request_words = ["could you", "would you", "please", "can you"]
        if any(w in content_lower for w in request_words):
            return DialogueIntent.REQUEST

        # Agreement/disagreement
        if content_lower.startswith("yes") or "i agree" in content_lower:
            return DialogueIntent.AGREE
        if content_lower.startswith("no") or "i disagree" in content_lower:
            return DialogueIntent.DISAGREE

        return DialogueIntent.INFORM

    def _heuristic_greeting(self, other_name: str, relationship: str) -> str:
        """Generate greeting without LLM"""
        if "friend" in relationship.lower() or "know" in relationship.lower():
            greetings = [
                f"Good to see you, {other_name}!",
                f"Ah, {other_name}! How have you been?",
                f"Hello, {other_name}. It's nice to meet you again."
            ]
        else:
            greetings = [
                f"Hello there. I'm {self.agent_name}.",
                f"Greetings, traveler.",
                f"Good day to you."
            ]

        import random
        return random.choice(greetings)

    def _heuristic_response(self, last_utterance: str, speaker_name: str) -> str:
        """Generate response without LLM"""
        # Question response
        if '?' in last_utterance:
            responses = [
                "That's an interesting question. Let me think about it.",
                "Hmm, I'm not entirely sure about that.",
                "I believe so, yes.",
                "That depends on how you look at it."
            ]
        else:
            responses = [
                "I see what you mean.",
                "That's quite interesting.",
                "I hadn't thought about it that way.",
                "Indeed.",
                "Tell me more about that."
            ]

        import random
        return random.choice(responses)

    def _heuristic_dialogue(self, topic: Optional[str], other_name: str) -> str:
        """Generate dialogue without LLM"""
        if topic:
            dialogues = [
                f"Speaking of {topic}, have you heard anything interesting lately?",
                f"What do you think about {topic}?",
                f"I've been thinking about {topic} recently."
            ]
        else:
            dialogues = [
                "How have things been going for you?",
                "Any news from around here?",
                "What brings you here today?"
            ]

        import random
        return random.choice(dialogues)

    def _heuristic_farewell(self, other_name: str) -> str:
        """Generate farewell without LLM"""
        farewells = [
            f"It was good talking to you, {other_name}. Take care.",
            f"I should get going. Until next time, {other_name}.",
            f"Farewell, {other_name}. Safe travels.",
            "I must be on my way. Goodbye for now."
        ]

        import random
        return random.choice(farewells)

    def _summarize_conversation(self, conversation: Conversation) -> str:
        """Create a brief summary of the conversation"""
        if len(conversation.turns) == 0:
            return "Brief encounter"

        if len(conversation.turns) <= 2:
            return f"Brief exchange about {conversation.topic or 'greetings'}"

        topics_mentioned = set()
        if conversation.topic:
            topics_mentioned.add(conversation.topic)

        # Extract any other topics from the conversation
        all_text = " ".join(t.content for t in conversation.turns)

        if len(topics_mentioned) > 0:
            return f"Discussed {', '.join(topics_mentioned)}"

        return f"Had a conversation ({len(conversation.turns)} exchanges)"

    def get_active_conversation(
        self,
        other_agent_id: UUID
    ) -> Optional[Conversation]:
        """Get active conversation with another agent"""
        for conv in self._active_conversations.values():
            if other_agent_id in conv.participants:
                return conv
        return None

    def get_all_active_conversations(self) -> List[Conversation]:
        """Get all active conversations"""
        return list(self._active_conversations.values())


class DialogueManager:
    """
    Manages multiple dialogue systems and conversations.
    """

    def __init__(self):
        """Initialize manager"""
        self._systems: Dict[UUID, DialogueSystem] = {}
        self._global_conversations: Dict[UUID, Conversation] = {}

    def register_agent(
        self,
        agent_id: UUID,
        dialogue_system: DialogueSystem
    ) -> None:
        """Register an agent's dialogue system"""
        self._systems[agent_id] = dialogue_system

    def get_system(self, agent_id: UUID) -> Optional[DialogueSystem]:
        """Get an agent's dialogue system"""
        return self._systems.get(agent_id)

    async def facilitate_conversation(
        self,
        agent1_id: UUID,
        agent2_id: UUID,
        agent1_summary: str,
        agent2_summary: str,
        topic: Optional[str] = None,
        location: Optional[str] = None,
        max_turns: int = 10
    ) -> Optional[Conversation]:
        """
        Facilitate a full conversation between two agents.

        Args:
            agent1_id: First agent
            agent2_id: Second agent
            agent1_summary: First agent's summary
            agent2_summary: Second agent's summary
            topic: Conversation topic
            location: Where it happens
            max_turns: Maximum turns

        Returns:
            The completed conversation
        """
        system1 = self._systems.get(agent1_id)
        system2 = self._systems.get(agent2_id)

        if not system1 or not system2:
            return None

        # Agent 1 starts
        conversation, opening = await system1.start_conversation(
            agent2_id,
            system2.agent_name,
            agent1_summary,
            topic,
            location
        )

        # Alternate turns
        current_system = system2
        current_summary = agent2_summary
        last_speaker_name = system1.agent_name
        turns = 1

        while turns < max_turns:
            # Get last utterance
            last_turn = conversation.turns[-1]

            # Generate response
            turn = await current_system.respond(
                conversation,
                last_turn.content,
                last_speaker_name,
                current_summary
            )

            turns += 1

            # Check for natural ending
            if turn.intent == DialogueIntent.FAREWELL:
                break

            # Swap speakers
            if current_system == system1:
                current_system = system2
                current_summary = agent2_summary
                last_speaker_name = system1.agent_name
            else:
                current_system = system1
                current_summary = agent1_summary
                last_speaker_name = system2.agent_name

        # End if not already ended
        if conversation.is_active:
            await current_system.end_conversation(conversation, current_summary)

        self._global_conversations[conversation.conversation_id] = conversation
        return conversation
