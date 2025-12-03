"""
Agent Interview System
Evaluates agent believability through structured interviews.
Based on Stanford Generative Agents paper methodology.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from agents.llm.ollama_client import OllamaClient
from agents.memory.memory_stream import MemoryStream


class QuestionCategory(str, Enum):
    """Categories of interview questions"""
    SELF_KNOWLEDGE = "self_knowledge"
    MEMORY_RECALL = "memory_recall"
    PLANNING = "planning"
    SOCIAL_AWARENESS = "social_awareness"
    REACTION = "reaction"
    REFLECTION = "reflection"


@dataclass
class InterviewQuestion:
    """A question for the agent interview"""
    question_id: UUID = field(default_factory=uuid4)
    category: QuestionCategory = QuestionCategory.SELF_KNOWLEDGE
    question: str = ""
    expected_topics: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    difficulty: float = 0.5  # 0-1


@dataclass
class InterviewResponse:
    """Agent's response to an interview question"""
    question_id: UUID = None
    response: str = ""
    response_time_ms: float = 0.0
    memories_retrieved: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InterviewResult:
    """Result of a complete agent interview"""
    interview_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    agent_name: str = ""

    # Responses
    responses: List[InterviewResponse] = field(default_factory=list)

    # Scores per category
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Overall score
    overall_score: float = 0.0

    # Metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "interview_id": str(self.interview_id),
            "agent_id": str(self.agent_id),
            "agent_name": self.agent_name,
            "category_scores": self.category_scores,
            "overall_score": self.overall_score,
            "duration_seconds": self.duration_seconds,
            "response_count": len(self.responses),
        }


class AgentInterviewer:
    """
    Conducts structured interviews with agents to evaluate believability.

    From the Stanford paper:
    "We conducted interviews with the agents to assess their
    self-knowledge, memory, plans, reactions, and reflections."

    Features:
    - Structured question sets per category
    - Memory-grounded response generation
    - Automatic scoring based on relevance and accuracy
    """

    # Standard interview questions by category
    STANDARD_QUESTIONS = {
        QuestionCategory.SELF_KNOWLEDGE: [
            "Can you tell me about yourself?",
            "What are your main personality traits?",
            "What do you do for a living?",
            "What are your goals in life?",
            "What are you most proud of?",
        ],
        QuestionCategory.MEMORY_RECALL: [
            "What did you do yesterday?",
            "Who have you talked to recently?",
            "What was the last significant thing that happened to you?",
            "Do you remember anything about {other_agent}?",
            "What happened at {location} recently?",
        ],
        QuestionCategory.PLANNING: [
            "What are your plans for today?",
            "What do you want to accomplish this week?",
            "How do you usually spend your mornings?",
            "What would you do if you had free time right now?",
        ],
        QuestionCategory.SOCIAL_AWARENESS: [
            "Who are your closest friends?",
            "How would you describe your relationship with {other_agent}?",
            "What do you think {other_agent} thinks of you?",
            "Have you heard any interesting news lately?",
        ],
        QuestionCategory.REACTION: [
            "How would you react if someone insulted you?",
            "What would you do if you saw someone in danger?",
            "How do you handle disagreements?",
            "What would you do if you found a valuable item on the ground?",
        ],
        QuestionCategory.REFLECTION: [
            "What have you learned recently?",
            "How have you changed over time?",
            "What patterns have you noticed in your life?",
            "What insights have you gained from your experiences?",
        ],
    }

    # Response evaluation prompt
    EVALUATION_PROMPT = """Evaluate this agent response for believability and accuracy.

Agent Description: {agent_summary}

Question Category: {category}
Question: {question}

Agent Response: {response}

Expected Topics: {expected_topics}
Ground Truth (if any): {ground_truth}

Rate the response on a scale of 0-10 for:
1. Relevance: Does it address the question?
2. Consistency: Is it consistent with the agent's character?
3. Memory Grounding: Does it reference specific memories/events?
4. Coherence: Is the response logical and well-formed?

Provide scores in format:
relevance: X
consistency: X
memory_grounding: X
coherence: X
overall: X
explanation: [brief explanation]"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize the interviewer.

        Args:
            ollama_client: LLM client for response generation and evaluation
        """
        self.client = ollama_client
        self._interview_history: List[InterviewResult] = []

    async def conduct_interview(
        self,
        agent_id: UUID,
        agent_name: str,
        agent_summary: str,
        memory_stream: MemoryStream,
        categories: Optional[List[QuestionCategory]] = None,
        questions_per_category: int = 3,
        context: Optional[Dict[str, str]] = None,
    ) -> InterviewResult:
        """
        Conduct a full interview with an agent.

        Args:
            agent_id: Agent's ID
            agent_name: Agent's name
            agent_summary: Agent's summary description
            memory_stream: Agent's memory stream
            categories: Categories to include (all by default)
            questions_per_category: Number of questions per category
            context: Context variables for question templates

        Returns:
            Interview result with scores
        """
        start_time = datetime.utcnow()

        result = InterviewResult(
            agent_id=agent_id,
            agent_name=agent_name,
            started_at=start_time,
        )

        # Select categories
        categories = categories or list(QuestionCategory)
        context = context or {}

        for category in categories:
            questions = self._get_questions(category, questions_per_category, context)
            category_responses = []

            for question in questions:
                # Generate agent response
                response = await self._generate_agent_response(
                    agent_name=agent_name,
                    agent_summary=agent_summary,
                    memory_stream=memory_stream,
                    question=question,
                )
                category_responses.append(response)
                result.responses.append(response)

            # Score the category
            if self.client:
                category_score = await self._evaluate_category(
                    agent_summary=agent_summary,
                    category=category,
                    questions=questions,
                    responses=category_responses,
                )
            else:
                category_score = self._heuristic_category_score(
                    questions, category_responses
                )

            result.category_scores[category.value] = category_score

        # Calculate overall score
        if result.category_scores:
            result.overall_score = sum(result.category_scores.values()) / len(
                result.category_scores
            )

        result.completed_at = datetime.utcnow()
        result.duration_seconds = (
            result.completed_at - start_time
        ).total_seconds()

        self._interview_history.append(result)
        return result

    async def ask_question(
        self,
        agent_name: str,
        agent_summary: str,
        memory_stream: MemoryStream,
        question: str,
        category: QuestionCategory = QuestionCategory.SELF_KNOWLEDGE,
    ) -> InterviewResponse:
        """
        Ask a single question to an agent.

        Args:
            agent_name: Agent's name
            agent_summary: Agent's description
            memory_stream: Agent's memory stream
            question: Question to ask
            category: Question category

        Returns:
            Agent's response
        """
        q = InterviewQuestion(
            category=category,
            question=question,
        )
        return await self._generate_agent_response(
            agent_name=agent_name,
            agent_summary=agent_summary,
            memory_stream=memory_stream,
            question=q,
        )

    def _get_questions(
        self,
        category: QuestionCategory,
        count: int,
        context: Dict[str, str],
    ) -> List[InterviewQuestion]:
        """Get questions for a category"""
        templates = self.STANDARD_QUESTIONS.get(category, [])
        questions = []

        for i, template in enumerate(templates[:count]):
            # Fill in context variables
            question_text = template
            for key, value in context.items():
                question_text = question_text.replace(f"{{{key}}}", value)

            # Skip if template variables not filled
            if "{" in question_text:
                continue

            questions.append(
                InterviewQuestion(
                    category=category,
                    question=question_text,
                    difficulty=0.5 + (i * 0.1),
                )
            )

        return questions

    async def _generate_agent_response(
        self,
        agent_name: str,
        agent_summary: str,
        memory_stream: MemoryStream,
        question: InterviewQuestion,
    ) -> InterviewResponse:
        """Generate agent's response to a question"""
        start_time = datetime.utcnow()

        # Retrieve relevant memories
        memories = await memory_stream.retrieve(
            query=question.question,
            limit=5,
        )
        memory_texts = [m.description for m in memories]

        # Generate response
        if self.client:
            response_text = await self._llm_generate_response(
                agent_name=agent_name,
                agent_summary=agent_summary,
                question=question.question,
                memories=memory_texts,
            )
        else:
            response_text = self._default_response(
                agent_name, question.category
            )

        end_time = datetime.utcnow()

        return InterviewResponse(
            question_id=question.question_id,
            response=response_text,
            response_time_ms=(end_time - start_time).total_seconds() * 1000,
            memories_retrieved=memory_texts[:3],
            timestamp=end_time,
        )

    async def _llm_generate_response(
        self,
        agent_name: str,
        agent_summary: str,
        question: str,
        memories: List[str],
    ) -> str:
        """Generate response using LLM"""
        memory_context = "\n".join(f"- {m}" for m in memories) if memories else "No specific memories retrieved."

        prompt = f"""{agent_summary}

You are {agent_name}. Answer the following question in character, drawing on your memories and personality.

Relevant memories:
{memory_context}

Question: {question}

{agent_name}:"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )
            return response.text.strip()
        except Exception as e:
            return f"I... I'm not sure how to answer that. ({str(e)})"

    async def _evaluate_category(
        self,
        agent_summary: str,
        category: QuestionCategory,
        questions: List[InterviewQuestion],
        responses: List[InterviewResponse],
    ) -> float:
        """Evaluate responses for a category using LLM"""
        scores = []

        for q, r in zip(questions, responses):
            prompt = self.EVALUATION_PROMPT.format(
                agent_summary=agent_summary,
                category=category.value,
                question=q.question,
                response=r.response,
                expected_topics=", ".join(q.expected_topics) or "N/A",
                ground_truth=q.ground_truth or "N/A",
            )

            try:
                eval_response = await self.client.generate(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=200,
                )
                score = self._parse_evaluation_score(eval_response.text)
                scores.append(score)
            except Exception:
                scores.append(5.0)  # Default middle score

        return sum(scores) / len(scores) if scores else 5.0

    def _parse_evaluation_score(self, response: str) -> float:
        """Parse evaluation score from LLM response"""
        import re

        # Look for "overall: X" pattern
        match = re.search(r"overall:\s*(\d+(?:\.\d+)?)", response.lower())
        if match:
            return float(match.group(1))

        # Fallback: look for any number
        numbers = re.findall(r"\d+(?:\.\d+)?", response)
        if numbers:
            return float(numbers[-1])

        return 5.0

    def _heuristic_category_score(
        self,
        questions: List[InterviewQuestion],
        responses: List[InterviewResponse],
    ) -> float:
        """Heuristic scoring when LLM not available"""
        scores = []

        for q, r in zip(questions, responses):
            score = 5.0  # Base score

            # Check response length (too short or too long is bad)
            words = len(r.response.split())
            if 10 <= words <= 100:
                score += 1.0
            elif words < 5:
                score -= 2.0

            # Check if memories were used
            if r.memories_retrieved:
                score += 1.0

            # Check response time (very fast might be cached/default)
            if 100 < r.response_time_ms < 10000:
                score += 0.5

            scores.append(min(10.0, max(0.0, score)))

        return sum(scores) / len(scores) if scores else 5.0

    def _default_response(
        self,
        agent_name: str,
        category: QuestionCategory
    ) -> str:
        """Default response when LLM not available"""
        defaults = {
            QuestionCategory.SELF_KNOWLEDGE: f"I am {agent_name}. I try to live each day with purpose.",
            QuestionCategory.MEMORY_RECALL: "I remember various events, but nothing specific comes to mind right now.",
            QuestionCategory.PLANNING: "I have plans to continue my work and maintain my relationships.",
            QuestionCategory.SOCIAL_AWARENESS: "I value the people in my life and try to understand them.",
            QuestionCategory.REACTION: "I would carefully consider the situation before acting.",
            QuestionCategory.REFLECTION: "I've learned many things from my experiences.",
        }
        return defaults.get(category, "I'm not sure how to answer that.")

    def get_interview_history(
        self,
        agent_id: Optional[UUID] = None
    ) -> List[InterviewResult]:
        """Get interview history, optionally filtered by agent"""
        if agent_id:
            return [r for r in self._interview_history if r.agent_id == agent_id]
        return self._interview_history

    def compare_agents(
        self,
        results: List[InterviewResult]
    ) -> Dict[str, Any]:
        """Compare interview results across multiple agents"""
        if not results:
            return {}

        comparison = {
            "agents": [],
            "category_averages": {},
            "best_performer": None,
            "worst_performer": None,
        }

        # Calculate category averages
        category_totals: Dict[str, List[float]] = {}

        for result in results:
            comparison["agents"].append({
                "agent_id": str(result.agent_id),
                "agent_name": result.agent_name,
                "overall_score": result.overall_score,
                "category_scores": result.category_scores,
            })

            for cat, score in result.category_scores.items():
                if cat not in category_totals:
                    category_totals[cat] = []
                category_totals[cat].append(score)

        for cat, scores in category_totals.items():
            comparison["category_averages"][cat] = sum(scores) / len(scores)

        # Find best/worst
        if comparison["agents"]:
            sorted_agents = sorted(
                comparison["agents"],
                key=lambda x: x["overall_score"],
                reverse=True
            )
            comparison["best_performer"] = sorted_agents[0]["agent_name"]
            comparison["worst_performer"] = sorted_agents[-1]["agent_name"]

        return comparison
