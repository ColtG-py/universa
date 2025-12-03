"""
Dialogue Service
Handles conversations between player and agents with memory and relationship updates.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """State of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class DialogueTurn:
    """A single turn in a conversation."""
    turn_id: str
    speaker_id: str
    speaker_name: str
    message: str
    speaker_type: str  # 'player' or 'agent'
    timestamp: datetime = field(default_factory=datetime.utcnow)
    emotion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "message": self.message,
            "speaker_type": self.speaker_type,
            "timestamp": self.timestamp.isoformat(),
            "emotion": self.emotion,
            "metadata": self.metadata
        }


@dataclass
class Conversation:
    """A conversation between participants."""
    conversation_id: str
    session_id: str
    participants: List[str]  # List of participant IDs
    participant_names: Dict[str, str]  # ID -> name mapping
    turns: List[DialogueTurn] = field(default_factory=list)
    state: ConversationState = ConversationState.ACTIVE
    location_x: int = 0
    location_y: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        speaker_id: str,
        speaker_name: str,
        message: str,
        speaker_type: str,
        emotion: Optional[str] = None
    ) -> DialogueTurn:
        """Add a turn to the conversation."""
        turn = DialogueTurn(
            turn_id=str(uuid4()),
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            message=message,
            speaker_type=speaker_type,
            emotion=emotion
        )
        self.turns.append(turn)
        return turn

    def get_recent_turns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent turns for context."""
        return [t.to_dict() for t in self.turns[-limit:]]

    @property
    def last_player_message(self) -> Optional[str]:
        """Get the last player message."""
        for turn in reversed(self.turns):
            if turn.speaker_type == 'player':
                return turn.message
        return None

    @property
    def last_agent_message(self) -> Optional[str]:
        """Get the last agent message."""
        for turn in reversed(self.turns):
            if turn.speaker_type == 'agent':
                return turn.message
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "participants": self.participants,
            "participant_names": self.participant_names,
            "turns": [t.to_dict() for t in self.turns],
            "state": self.state.value,
            "location": {"x": self.location_x, "y": self.location_y},
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turn_count": len(self.turns)
        }


@dataclass
class DialogueResponse:
    """Response from an agent in dialogue."""
    text: str
    emotion: Optional[str] = None
    should_end: bool = False
    action_taken: Optional[str] = None
    relationship_change: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "should_end": self.should_end,
            "action_taken": self.action_taken,
            "relationship_change": self.relationship_change
        }


class DialogueService:
    """
    Service for handling conversations between player and agents.

    Features:
    - Start/continue/end conversations
    - Generate contextual agent responses
    - Update agent memories based on conversations
    - Update relationships based on interactions
    """

    def __init__(self):
        # Active conversations (conversation_id -> Conversation)
        self._conversations: Dict[str, Conversation] = {}
        # Agent's active conversation (agent_id -> conversation_id)
        self._agent_conversations: Dict[str, str] = {}
        # Game service reference (set by dependency injection)
        self._game_service = None
        # LLM service reference
        self._llm_service = None

    def set_game_service(self, game_service):
        """Set the game service reference."""
        self._game_service = game_service

    def set_llm_service(self, llm_service):
        """Set the LLM service reference."""
        self._llm_service = llm_service

    async def start_conversation(
        self,
        session_id: str,
        player_id: str,
        player_name: str,
        agent_id: str,
        agent_name: str,
        player_x: int,
        player_y: int,
        opening_message: Optional[str] = None
    ) -> Conversation:
        """
        Start a new conversation between player and agent.

        Args:
            session_id: Game session ID
            player_id: Player's ID
            player_name: Player's name
            agent_id: Agent's ID
            agent_name: Agent's name
            player_x: X location
            player_y: Y location
            opening_message: Optional opening message from player

        Returns:
            New Conversation object
        """
        # Check if agent is already in conversation
        if agent_id in self._agent_conversations:
            existing_id = self._agent_conversations[agent_id]
            existing = self._conversations.get(existing_id)
            if existing and existing.state == ConversationState.ACTIVE:
                logger.warning(f"Agent {agent_id} already in conversation {existing_id}")
                return existing

        # Create new conversation
        conversation = Conversation(
            conversation_id=str(uuid4()),
            session_id=session_id,
            participants=[player_id, agent_id],
            participant_names={
                player_id: player_name,
                agent_id: agent_name
            },
            location_x=player_x,
            location_y=player_y
        )

        # Store conversation
        self._conversations[conversation.conversation_id] = conversation
        self._agent_conversations[agent_id] = conversation.conversation_id

        # Notify game service that agent is in interaction (for tier upgrade)
        if self._game_service:
            try:
                await self._game_service.start_interaction(session_id, agent_id)
            except Exception as e:
                logger.warning(f"Failed to notify game service of interaction: {e}")

        if opening_message:
            # Player initiated with a message
            conversation.add_turn(
                speaker_id=player_id,
                speaker_name=player_name,
                message=opening_message,
                speaker_type='player'
            )

            # Generate agent response
            response = await self.generate_agent_response(
                conversation=conversation,
                agent_id=agent_id,
                agent_name=agent_name
            )

            conversation.add_turn(
                speaker_id=agent_id,
                speaker_name=agent_name,
                message=response.text,
                speaker_type='agent',
                emotion=response.emotion
            )
        else:
            # Agent initiates (player just approached)
            greeting = await self.generate_greeting(
                agent_id=agent_id,
                agent_name=agent_name,
                player_name=player_name
            )

            conversation.add_turn(
                speaker_id=agent_id,
                speaker_name=agent_name,
                message=greeting.text,
                speaker_type='agent',
                emotion=greeting.emotion
            )

        logger.info(f"Started conversation {conversation.conversation_id} between {player_name} and {agent_name}")
        return conversation

    async def continue_conversation(
        self,
        conversation_id: str,
        speaker_id: str,
        speaker_name: str,
        message: str,
        speaker_type: str = 'player'
    ) -> Optional[DialogueTurn]:
        """
        Continue an existing conversation with a new message.

        Args:
            conversation_id: ID of the conversation
            speaker_id: ID of the speaker
            speaker_name: Name of the speaker
            message: The message content
            speaker_type: 'player' or 'agent'

        Returns:
            The new DialogueTurn or None if conversation not found
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return None

        if conversation.state != ConversationState.ACTIVE:
            logger.warning(f"Conversation {conversation_id} is not active")
            return None

        # Add the player's message
        turn = conversation.add_turn(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            message=message,
            speaker_type=speaker_type
        )

        return turn

    async def get_agent_response(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str
    ) -> Optional[DialogueResponse]:
        """
        Get an agent's response in a conversation.

        Args:
            conversation_id: ID of the conversation
            agent_id: ID of the agent
            agent_name: Name of the agent

        Returns:
            DialogueResponse with agent's reply
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return None

        # Generate response
        response = await self.generate_agent_response(
            conversation=conversation,
            agent_id=agent_id,
            agent_name=agent_name
        )

        # Add to conversation
        conversation.add_turn(
            speaker_id=agent_id,
            speaker_name=agent_name,
            message=response.text,
            speaker_type='agent',
            emotion=response.emotion
        )

        # Check if conversation should end
        if response.should_end:
            await self.end_conversation(conversation_id)

        return response

    async def end_conversation(
        self,
        conversation_id: str,
        reason: str = "ended"
    ) -> Optional[Conversation]:
        """
        End an active conversation.

        Args:
            conversation_id: ID of the conversation to end
            reason: Reason for ending

        Returns:
            The ended Conversation
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None

        conversation.state = ConversationState.ENDED
        conversation.ended_at = datetime.utcnow()
        conversation.context['end_reason'] = reason

        # Remove from active agent conversations
        for agent_id in conversation.participants:
            if agent_id in self._agent_conversations:
                if self._agent_conversations[agent_id] == conversation_id:
                    del self._agent_conversations[agent_id]

                    # Notify game service that interaction ended
                    if self._game_service:
                        try:
                            await self._game_service.end_interaction(
                                conversation.session_id,
                                agent_id
                            )
                        except Exception as e:
                            logger.warning(f"Failed to notify game service: {e}")

        # Create memories for participants
        await self._create_conversation_memories(conversation)

        logger.info(f"Ended conversation {conversation_id}: {reason}")
        return conversation

    async def generate_greeting(
        self,
        agent_id: str,
        agent_name: str,
        player_name: str
    ) -> DialogueResponse:
        """
        Generate an agent's greeting when player approaches.

        Args:
            agent_id: Agent's ID
            agent_name: Agent's name
            player_name: Player's name

        Returns:
            DialogueResponse with greeting
        """
        # Get agent context for personalized greeting
        agent_context = await self._get_agent_context(agent_id, player_name)

        # Build prompt
        prompt = self._build_greeting_prompt(
            agent_name=agent_name,
            player_name=player_name,
            context=agent_context
        )

        # Generate via LLM if available
        if self._llm_service:
            try:
                response_text = await self._llm_service.generate(
                    prompt=prompt,
                    system=DIALOGUE_SYSTEM_PROMPT,
                    max_tokens=150
                )
                emotion = self._detect_emotion(response_text)
                return DialogueResponse(
                    text=response_text,
                    emotion=emotion
                )
            except Exception as e:
                logger.error(f"LLM greeting generation failed: {e}")

        # Fallback to template
        greeting = self._get_template_greeting(agent_context, agent_name, player_name)
        return DialogueResponse(text=greeting, emotion='neutral')

    async def generate_agent_response(
        self,
        conversation: Conversation,
        agent_id: str,
        agent_name: str
    ) -> DialogueResponse:
        """
        Generate an agent's response in conversation.

        Args:
            conversation: The conversation context
            agent_id: Agent's ID
            agent_name: Agent's name

        Returns:
            DialogueResponse with agent's reply
        """
        # Get player name
        player_name = None
        for pid, name in conversation.participant_names.items():
            if pid != agent_id:
                player_name = name
                break

        # Get agent context
        agent_context = await self._get_agent_context(agent_id, player_name)

        # Build dialogue prompt
        prompt = self._build_dialogue_prompt(
            agent_name=agent_name,
            player_name=player_name,
            conversation=conversation,
            context=agent_context
        )

        # Generate via LLM if available
        if self._llm_service:
            try:
                response_text = await self._llm_service.generate(
                    prompt=prompt,
                    system=DIALOGUE_SYSTEM_PROMPT,
                    max_tokens=200
                )
                emotion = self._detect_emotion(response_text)
                should_end = self._should_end_conversation(response_text)
                return DialogueResponse(
                    text=response_text,
                    emotion=emotion,
                    should_end=should_end
                )
            except Exception as e:
                logger.error(f"LLM dialogue generation failed: {e}")

        # Fallback to template response
        response = self._get_template_response(
            agent_context,
            conversation.last_player_message
        )
        return DialogueResponse(text=response, emotion='neutral')

    async def _get_agent_context(
        self,
        agent_id: str,
        player_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get agent context for dialogue generation.

        Includes:
        - Agent personality/traits
        - Relevant memories
        - Relationship with player
        - Current state (needs, mood)
        """
        context = {
            'agent_id': agent_id,
            'memories': [],
            'relationship': None,
            'personality': {},
            'occupation': 'unknown',
            'mood': 'neutral'
        }

        # Try to get agent data from game service
        if self._game_service:
            try:
                # Get agent from session
                for session in self._game_service._sessions.values():
                    agent = session.get('agents', {}).get(agent_id)
                    if agent:
                        context['personality'] = agent.get('traits', {})
                        context['occupation'] = agent.get('occupation', 'unknown')
                        break
            except Exception as e:
                logger.warning(f"Failed to get agent context: {e}")

        # Try to get memories about player
        if player_name:
            try:
                from agents.debug.inspector import get_inspector
                inspector = get_inspector()
                memories = await inspector.search_memories(
                    agent_id=UUID(agent_id),
                    query=f"interactions with {player_name}",
                    limit=5
                )
                context['memories'] = memories
            except Exception as e:
                logger.warning(f"Failed to get agent memories: {e}")

        return context

    async def _create_conversation_memories(
        self,
        conversation: Conversation
    ) -> None:
        """
        Create memories for conversation participants.
        """
        if len(conversation.turns) < 2:
            return

        # Summarize conversation
        summary = self._summarize_conversation(conversation)

        # Create memory for each agent participant
        for participant_id in conversation.participants:
            # Skip if player (players don't have memory streams)
            if any(name.lower() == 'player' for name in conversation.participant_names.values()):
                continue

            try:
                from agents.debug.inspector import get_inspector
                inspector = get_inspector()

                # Get memory stream
                memory_stream = inspector._memory_streams.get(UUID(participant_id))
                if memory_stream:
                    # Add conversation memory
                    other_name = [
                        name for pid, name in conversation.participant_names.items()
                        if pid != participant_id
                    ][0]

                    description = f"Had a conversation with {other_name}: {summary}"
                    # Would call memory_stream.add_observation here
                    logger.info(f"Would create memory for {participant_id}: {description[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to create conversation memory: {e}")

    def _build_greeting_prompt(
        self,
        agent_name: str,
        player_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for greeting generation."""
        occupation = context.get('occupation', 'person')
        mood = context.get('mood', 'neutral')

        prompt = f"""You are {agent_name}, a {occupation}.
Your current mood is {mood}.

{player_name} has just approached you.

Generate a brief, natural greeting (1-2 sentences) that reflects your personality and current mood.
Speak in first person as {agent_name}.
"""
        return prompt

    def _build_dialogue_prompt(
        self,
        agent_name: str,
        player_name: str,
        conversation: Conversation,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for dialogue response."""
        occupation = context.get('occupation', 'person')
        personality = context.get('personality', {})
        memories = context.get('memories', [])

        # Format conversation history
        history = "\n".join([
            f"{t.speaker_name}: {t.message}"
            for t in conversation.turns[-5:]
        ])

        # Format memories
        memory_text = ""
        if memories:
            memory_text = "Relevant memories:\n" + "\n".join([
                f"- {m.get('description', m)}"
                for m in memories[:3]
            ])

        prompt = f"""You are {agent_name}, a {occupation}.
Personality: {personality}

{memory_text}

Conversation:
{history}

Generate a natural, in-character response (1-3 sentences).
Speak in first person as {agent_name}.
If the conversation seems to be ending naturally, you may say goodbye.
"""
        return prompt

    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from response text."""
        text_lower = text.lower()

        if any(word in text_lower for word in ['happy', 'glad', 'pleased', 'wonderful', 'delighted']):
            return 'happy'
        elif any(word in text_lower for word in ['angry', 'furious', 'annoyed', 'frustrated']):
            return 'angry'
        elif any(word in text_lower for word in ['sad', 'sorry', 'unfortunate', 'regret']):
            return 'sad'
        elif any(word in text_lower for word in ['afraid', 'worried', 'nervous', 'scared']):
            return 'fearful'
        elif any(word in text_lower for word in ['surprised', 'amazed', 'shocked']):
            return 'surprised'
        else:
            return 'neutral'

    def _should_end_conversation(self, text: str) -> bool:
        """Check if response indicates conversation should end."""
        text_lower = text.lower()
        end_phrases = [
            'goodbye', 'farewell', 'i must go', 'have to leave',
            'i should get back', 'need to continue', 'nice talking',
            'take care', 'be seeing you', 'until next time'
        ]
        return any(phrase in text_lower for phrase in end_phrases)

    def _summarize_conversation(self, conversation: Conversation) -> str:
        """Create a brief summary of the conversation."""
        if len(conversation.turns) <= 2:
            return conversation.turns[-1].message if conversation.turns else "Brief exchange"

        # Simple summary from first and last meaningful exchanges
        first_exchange = conversation.turns[0].message[:50]
        last_exchange = conversation.turns[-1].message[:50]
        return f"Discussed: {first_exchange}... ended with: {last_exchange}..."

    def _get_template_greeting(
        self,
        context: Dict[str, Any],
        agent_name: str,
        player_name: str
    ) -> str:
        """Get a template greeting when LLM is unavailable."""
        occupation = context.get('occupation', 'person')
        templates = [
            f"Greetings, traveler. I am {agent_name}.",
            f"Hello there. What brings you to speak with a {occupation}?",
            f"Well met. I'm {agent_name}. How can I help you?",
            f"Ah, hello. I don't believe we've met. I'm {agent_name}."
        ]
        import random
        return random.choice(templates)

    def _get_template_response(
        self,
        context: Dict[str, Any],
        player_message: Optional[str]
    ) -> str:
        """Get a template response when LLM is unavailable."""
        if not player_message:
            return "I see. Is there something you wanted to discuss?"

        message_lower = player_message.lower()

        if any(q in message_lower for q in ['how are', 'how do you']):
            return "I'm doing well enough, thank you for asking."
        elif any(q in message_lower for q in ['what do you', 'what are you']):
            return "I keep myself busy with my work, as one does."
        elif any(q in message_lower for q in ['where', 'location']):
            return "I'm afraid I don't know much about that area."
        elif any(q in message_lower for q in ['help', 'need']):
            return "I'll do what I can to assist you."
        elif any(q in message_lower for q in ['goodbye', 'farewell', 'bye']):
            return "Safe travels to you. Goodbye."
        else:
            return "That's interesting. Tell me more."

    # Conversation retrieval methods

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def get_agent_conversation(self, agent_id: str) -> Optional[Conversation]:
        """Get an agent's active conversation."""
        conv_id = self._agent_conversations.get(agent_id)
        if conv_id:
            return self._conversations.get(conv_id)
        return None

    def get_session_conversations(
        self,
        session_id: str,
        active_only: bool = True
    ) -> List[Conversation]:
        """Get all conversations in a session."""
        conversations = [
            c for c in self._conversations.values()
            if c.session_id == session_id
        ]

        if active_only:
            conversations = [
                c for c in conversations
                if c.state == ConversationState.ACTIVE
            ]

        return conversations


# System prompt for dialogue generation
DIALOGUE_SYSTEM_PROMPT = """You are an NPC in a fantasy world. You have your own personality,
occupation, and history. Respond naturally and in-character to the player.

Guidelines:
- Stay in character at all times
- Reference your memories and relationships when relevant
- Express emotions appropriate to the conversation
- Keep responses concise (1-3 sentences usually)
- If asked about things you don't know, admit it naturally
- You may initiate ending conversations naturally when appropriate
"""
