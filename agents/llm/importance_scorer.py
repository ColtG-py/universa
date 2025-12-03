"""
Importance Scorer
LLM-based scoring of memory importance.
Based on the Stanford Generative Agents paper methodology.
"""

from typing import Optional, Callable
import re
import asyncio

from agents.llm.ollama_client import OllamaClient, get_ollama_client


class ImportanceScorer:
    """
    Scores the importance/poignancy of memories using an LLM.

    From the Stanford paper:
    "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth)
    and 10 is extremely poignant (e.g., a break up, college acceptance),
    rate the likely poignancy of the following memory."
    """

    # Prompt template from Stanford paper
    IMPORTANCE_PROMPT = """On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, major discovery, death of a loved one), rate the likely poignancy of the following piece of memory.

Memory: {memory}

Respond with only a single number between 1 and 10.
Rating:"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model: str = "llama3.2",
        use_cache: bool = True,
    ):
        """
        Initialize importance scorer.

        Args:
            ollama_client: Ollama client (creates default if None)
            model: Model to use for scoring
            use_cache: Whether to cache scores
        """
        self.client = ollama_client or get_ollama_client()
        self.model = model
        self.use_cache = use_cache
        self._cache: dict = {}

    async def score(self, memory_description: str) -> float:
        """
        Score the importance of a memory description.

        Args:
            memory_description: The memory text to score

        Returns:
            Importance score normalized to 0-1 range
        """
        # Check cache
        if self.use_cache and memory_description in self._cache:
            return self._cache[memory_description]

        prompt = self.IMPORTANCE_PROMPT.format(memory=memory_description)

        try:
            response = await self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=10,
            )

            # Parse the response
            score = self._parse_score(response.text)

            # Cache result
            if self.use_cache:
                self._manage_cache()
                self._cache[memory_description] = score

            return score

        except Exception as e:
            # Fallback to heuristic scoring
            return self._heuristic_score(memory_description)

    def score_sync(self, memory_description: str) -> float:
        """Synchronous version of score"""
        return asyncio.get_event_loop().run_until_complete(
            self.score(memory_description)
        )

    async def score_batch(self, memories: list) -> list:
        """
        Score multiple memories in parallel.

        Args:
            memories: List of memory descriptions

        Returns:
            List of importance scores
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def score_one(memory: str) -> float:
            async with semaphore:
                return await self.score(memory)

        tasks = [score_one(m) for m in memories]
        return await asyncio.gather(*tasks)

    def _parse_score(self, response: str) -> float:
        """
        Parse LLM response to extract score.

        Args:
            response: Raw LLM response

        Returns:
            Normalized score (0-1)
        """
        # Extract first number from response
        response = response.strip()

        # Try to find a number 1-10
        match = re.search(r'\b([1-9]|10)\b', response)
        if match:
            raw_score = int(match.group(1))
            # Normalize to 0-1
            return raw_score / 10.0

        # Try floating point
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            raw_score = float(match.group(1))
            if raw_score <= 1:
                return raw_score
            elif raw_score <= 10:
                return raw_score / 10.0

        # Default to medium importance
        return 0.5

    def _heuristic_score(self, memory: str) -> float:
        """
        Fallback heuristic scoring based on keywords.

        Args:
            memory: Memory description

        Returns:
            Estimated importance (0-1)
        """
        memory_lower = memory.lower()

        # High importance keywords (score ~0.8-1.0)
        high_keywords = [
            "died", "death", "killed", "murder", "born", "birth",
            "married", "wedding", "divorce", "engagement",
            "discovered", "revelation", "secret",
            "betrayed", "betrayal", "confession",
            "war", "battle", "victory", "defeat",
            "love", "heartbreak", "proposal"
        ]

        # Medium-high keywords (score ~0.6-0.8)
        medium_high_keywords = [
            "fought", "argument", "conflict",
            "learned", "realized", "understood",
            "promoted", "fired", "hired",
            "moved", "traveled far", "journey",
            "sick", "illness", "recovered",
            "promise", "vow", "oath"
        ]

        # Medium keywords (score ~0.4-0.6)
        medium_keywords = [
            "met", "introduced", "greeted",
            "bought", "sold", "traded",
            "finished", "completed", "started",
            "told", "said", "heard",
            "decided", "chose", "planned"
        ]

        # Check keywords
        for kw in high_keywords:
            if kw in memory_lower:
                return 0.85

        for kw in medium_high_keywords:
            if kw in memory_lower:
                return 0.65

        for kw in medium_keywords:
            if kw in memory_lower:
                return 0.45

        # Default to low importance for mundane observations
        return 0.25

    def _manage_cache(self) -> None:
        """Remove old cache entries if cache is too large"""
        max_cache_size = 1000
        if len(self._cache) >= max_cache_size:
            # Remove half the cache (oldest entries would require timestamps)
            keys = list(self._cache.keys())
            for key in keys[:len(keys)//2]:
                del self._cache[key]

    def get_scorer_function(self) -> Callable[[str], float]:
        """
        Get a callable scorer function for use with MemoryStream.

        Returns:
            Function that takes memory text and returns importance score
        """
        def scorer(memory: str) -> float:
            return self.score_sync(memory)
        return scorer


# Convenience function for quick scoring
async def score_importance(
    memory: str,
    ollama_client: Optional[OllamaClient] = None
) -> float:
    """
    Quick importance scoring function.

    Args:
        memory: Memory description
        ollama_client: Optional client

    Returns:
        Importance score (0-1)
    """
    scorer = ImportanceScorer(ollama_client)
    return await scorer.score(memory)
