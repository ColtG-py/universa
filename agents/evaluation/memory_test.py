"""
Memory Accuracy Testing
Tests agent memory recall accuracy against ground truth.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from agents.memory.memory_stream import MemoryStream
from agents.llm.ollama_client import OllamaClient


class MemoryTestType(str, Enum):
    """Types of memory tests"""
    RECENT_RECALL = "recent_recall"
    SPECIFIC_EVENT = "specific_event"
    PERSON_RECALL = "person_recall"
    LOCATION_RECALL = "location_recall"
    TEMPORAL_ORDER = "temporal_order"
    DETAIL_ACCURACY = "detail_accuracy"


@dataclass
class MemoryTestCase:
    """A single memory test case"""
    test_id: UUID = field(default_factory=uuid4)
    test_type: MemoryTestType = MemoryTestType.RECENT_RECALL
    description: str = ""
    query: str = ""
    expected_content: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    difficulty: float = 0.5


@dataclass
class MemoryTestResult:
    """Result of a memory test"""
    test_id: UUID = field(default_factory=uuid4)
    test_case: Optional[MemoryTestCase] = None
    agent_id: UUID = None

    # Retrieved memories
    retrieved_memories: List[str] = field(default_factory=list)

    # Scoring
    recall_score: float = 0.0  # Did they remember?
    accuracy_score: float = 0.0  # Was it accurate?
    relevance_score: float = 0.0  # Was it relevant?
    overall_score: float = 0.0

    # Metadata
    retrieval_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": str(self.test_id),
            "test_type": self.test_case.test_type.value if self.test_case else None,
            "recall_score": self.recall_score,
            "accuracy_score": self.accuracy_score,
            "relevance_score": self.relevance_score,
            "overall_score": self.overall_score,
            "memories_retrieved": len(self.retrieved_memories),
        }


class MemoryAccuracyTester:
    """
    Tests agent memory accuracy against ground truth.

    Features:
    - Injects known events and tests recall
    - Measures retrieval accuracy
    - Tests temporal ordering
    - Evaluates detail preservation
    """

    # Accuracy evaluation prompt
    ACCURACY_PROMPT = """Evaluate memory recall accuracy.

Test Query: {query}

Expected Content (ground truth):
{expected}

Retrieved Memories:
{retrieved}

Rate on 0-10:
1. Recall Score: Did they retrieve relevant memories? (0=nothing, 10=perfect)
2. Accuracy Score: Is the content accurate? (0=wrong, 10=perfect)
3. Relevance Score: Are memories relevant to query? (0=irrelevant, 10=perfect)

Format:
recall: X
accuracy: X
relevance: X
explanation: [brief explanation]"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize memory tester.

        Args:
            ollama_client: LLM client for accuracy evaluation
        """
        self.client = ollama_client
        self._test_history: List[MemoryTestResult] = []

    async def inject_test_memories(
        self,
        memory_stream: MemoryStream,
        memories: List[Dict[str, Any]],
    ) -> List[UUID]:
        """
        Inject known memories for testing.

        Args:
            memory_stream: Agent's memory stream
            memories: Memories to inject

        Returns:
            List of created memory IDs
        """
        created_ids = []

        for mem in memories:
            observation = await memory_stream.add_observation(
                description=mem.get("description", ""),
                location_x=mem.get("location_x"),
                location_y=mem.get("location_y"),
                importance=mem.get("importance", 0.5),
            )
            created_ids.append(observation.memory_id)

        return created_ids

    async def run_test(
        self,
        memory_stream: MemoryStream,
        test_case: MemoryTestCase,
    ) -> MemoryTestResult:
        """
        Run a single memory test.

        Args:
            memory_stream: Agent's memory stream
            test_case: Test case to run

        Returns:
            Test result
        """
        start_time = datetime.utcnow()

        # Retrieve memories using the query
        memories = await memory_stream.retrieve(
            query=test_case.query,
            limit=10,
        )
        memory_texts = [m.description for m in memories]

        end_time = datetime.utcnow()

        # Evaluate accuracy
        if self.client and test_case.expected_content:
            scores = await self._evaluate_accuracy(
                query=test_case.query,
                expected=test_case.expected_content,
                retrieved=memory_texts,
            )
        else:
            scores = self._heuristic_scores(
                expected=test_case.expected_content,
                retrieved=memory_texts,
            )

        result = MemoryTestResult(
            test_case=test_case,
            agent_id=memory_stream.agent_id,
            retrieved_memories=memory_texts,
            recall_score=scores.get("recall", 5.0),
            accuracy_score=scores.get("accuracy", 5.0),
            relevance_score=scores.get("relevance", 5.0),
            retrieval_time_ms=(end_time - start_time).total_seconds() * 1000,
        )

        # Calculate overall score
        result.overall_score = (
            result.recall_score * 0.4 +
            result.accuracy_score * 0.4 +
            result.relevance_score * 0.2
        )

        self._test_history.append(result)
        return result

    async def run_test_suite(
        self,
        memory_stream: MemoryStream,
        test_cases: Optional[List[MemoryTestCase]] = None,
    ) -> List[MemoryTestResult]:
        """
        Run a suite of memory tests.

        Args:
            memory_stream: Agent's memory stream
            test_cases: Test cases to run (uses defaults if None)

        Returns:
            List of test results
        """
        if test_cases is None:
            test_cases = self._generate_default_tests()

        results = []
        for test_case in test_cases:
            result = await self.run_test(memory_stream, test_case)
            results.append(result)

        return results

    async def test_recent_recall(
        self,
        memory_stream: MemoryStream,
        hours: int = 24,
    ) -> MemoryTestResult:
        """
        Test recall of recent memories.

        Args:
            memory_stream: Agent's memory stream
            hours: How many hours back to test

        Returns:
            Test result
        """
        # Get actual recent memories
        recent = await memory_stream.retrieve_recent(hours=hours)
        expected = [m.description for m in recent[:5]]

        test_case = MemoryTestCase(
            test_type=MemoryTestType.RECENT_RECALL,
            description=f"Recall events from the last {hours} hours",
            query="What happened recently?",
            expected_content=expected,
        )

        return await self.run_test(memory_stream, test_case)

    async def test_specific_event(
        self,
        memory_stream: MemoryStream,
        event_keywords: List[str],
        expected_details: List[str],
    ) -> MemoryTestResult:
        """
        Test recall of a specific event.

        Args:
            memory_stream: Agent's memory stream
            event_keywords: Keywords to search for
            expected_details: Expected details to be recalled

        Returns:
            Test result
        """
        query = " ".join(event_keywords)

        test_case = MemoryTestCase(
            test_type=MemoryTestType.SPECIFIC_EVENT,
            description=f"Recall event: {query}",
            query=query,
            expected_content=expected_details,
        )

        return await self.run_test(memory_stream, test_case)

    async def test_person_recall(
        self,
        memory_stream: MemoryStream,
        person_name: str,
        expected_facts: List[str],
    ) -> MemoryTestResult:
        """
        Test recall of information about a person.

        Args:
            memory_stream: Agent's memory stream
            person_name: Name of person to recall
            expected_facts: Expected facts about the person

        Returns:
            Test result
        """
        test_case = MemoryTestCase(
            test_type=MemoryTestType.PERSON_RECALL,
            description=f"Recall information about {person_name}",
            query=f"What do I know about {person_name}?",
            expected_content=expected_facts,
        )

        return await self.run_test(memory_stream, test_case)

    async def test_temporal_order(
        self,
        memory_stream: MemoryStream,
        events_in_order: List[str],
    ) -> MemoryTestResult:
        """
        Test if agent recalls events in correct temporal order.

        Args:
            memory_stream: Agent's memory stream
            events_in_order: Events in chronological order

        Returns:
            Test result
        """
        test_case = MemoryTestCase(
            test_type=MemoryTestType.TEMPORAL_ORDER,
            description="Recall events in chronological order",
            query="What happened today, in order?",
            expected_content=events_in_order,
        )

        result = await self.run_test(memory_stream, test_case)

        # Additional check for order
        if result.retrieved_memories and events_in_order:
            order_score = self._check_temporal_order(
                expected=events_in_order,
                retrieved=result.retrieved_memories,
            )
            # Adjust overall score based on order accuracy
            result.overall_score = (result.overall_score + order_score) / 2

        return result

    async def _evaluate_accuracy(
        self,
        query: str,
        expected: List[str],
        retrieved: List[str],
    ) -> Dict[str, float]:
        """Evaluate accuracy using LLM"""
        expected_text = "\n".join(f"- {e}" for e in expected) or "N/A"
        retrieved_text = "\n".join(f"- {r}" for r in retrieved) or "No memories retrieved"

        prompt = self.ACCURACY_PROMPT.format(
            query=query,
            expected=expected_text,
            retrieved=retrieved_text,
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150,
            )
            return self._parse_scores(response.text)
        except Exception:
            return self._heuristic_scores(expected, retrieved)

    def _parse_scores(self, response: str) -> Dict[str, float]:
        """Parse scores from LLM response"""
        import re

        scores = {"recall": 5.0, "accuracy": 5.0, "relevance": 5.0}

        for key in scores.keys():
            match = re.search(rf"{key}:\s*(\d+(?:\.\d+)?)", response.lower())
            if match:
                scores[key] = min(10.0, float(match.group(1)))

        return scores

    def _heuristic_scores(
        self,
        expected: List[str],
        retrieved: List[str],
    ) -> Dict[str, float]:
        """Calculate heuristic scores"""
        if not retrieved:
            return {"recall": 0.0, "accuracy": 5.0, "relevance": 0.0}

        # Recall: did we get memories?
        recall = min(10.0, len(retrieved) * 2)

        # Accuracy: word overlap with expected
        if expected:
            expected_words = set(" ".join(expected).lower().split())
            retrieved_words = set(" ".join(retrieved).lower().split())
            overlap = len(expected_words & retrieved_words)
            accuracy = min(10.0, overlap / max(1, len(expected_words)) * 10)
        else:
            accuracy = 5.0

        # Relevance: based on memory count and length
        avg_length = sum(len(r.split()) for r in retrieved) / len(retrieved)
        relevance = min(10.0, avg_length / 5)

        return {
            "recall": recall,
            "accuracy": accuracy,
            "relevance": relevance,
        }

    def _check_temporal_order(
        self,
        expected: List[str],
        retrieved: List[str],
    ) -> float:
        """Check if retrieved memories are in expected order"""
        if len(expected) < 2 or len(retrieved) < 2:
            return 5.0

        # Find positions of expected items in retrieved
        positions = []
        for exp in expected:
            for i, ret in enumerate(retrieved):
                if exp.lower() in ret.lower() or ret.lower() in exp.lower():
                    positions.append(i)
                    break

        if len(positions) < 2:
            return 5.0

        # Check if positions are increasing (correct order)
        in_order = sum(1 for i in range(len(positions) - 1)
                       if positions[i] < positions[i + 1])
        total_pairs = len(positions) - 1

        return (in_order / total_pairs) * 10 if total_pairs > 0 else 5.0

    def _generate_default_tests(self) -> List[MemoryTestCase]:
        """Generate default test cases"""
        return [
            MemoryTestCase(
                test_type=MemoryTestType.RECENT_RECALL,
                description="Recent event recall",
                query="What happened recently?",
                difficulty=0.3,
            ),
            MemoryTestCase(
                test_type=MemoryTestType.LOCATION_RECALL,
                description="Location-based recall",
                query="What have I done at my usual locations?",
                difficulty=0.5,
            ),
            MemoryTestCase(
                test_type=MemoryTestType.DETAIL_ACCURACY,
                description="Detail recall",
                query="What specific details do I remember?",
                difficulty=0.7,
            ),
        ]

    def get_test_history(
        self,
        agent_id: Optional[UUID] = None
    ) -> List[MemoryTestResult]:
        """Get test history, optionally filtered by agent"""
        if agent_id:
            return [r for r in self._test_history if r.agent_id == agent_id]
        return self._test_history

    def get_summary_statistics(
        self,
        results: Optional[List[MemoryTestResult]] = None
    ) -> Dict[str, Any]:
        """Get summary statistics from test results"""
        results = results or self._test_history

        if not results:
            return {"total_tests": 0}

        return {
            "total_tests": len(results),
            "avg_recall": sum(r.recall_score for r in results) / len(results),
            "avg_accuracy": sum(r.accuracy_score for r in results) / len(results),
            "avg_relevance": sum(r.relevance_score for r in results) / len(results),
            "avg_overall": sum(r.overall_score for r in results) / len(results),
            "avg_retrieval_time_ms": sum(r.retrieval_time_ms for r in results) / len(results),
        }
