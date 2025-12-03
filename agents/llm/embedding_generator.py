"""
Embedding Generator
Generates vector embeddings for semantic memory search.
"""

from typing import Optional, List, Callable, Dict
import asyncio

from agents.llm.ollama_client import OllamaClient, get_ollama_client
from agents.config import EMBEDDING_DIMENSION


class EmbeddingGenerator:
    """
    Generates embeddings for memory text using local LLMs.
    Supports caching and batch processing.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model: str = "nomic-embed-text",
        use_cache: bool = True,
        cache_max_size: int = 5000,
    ):
        """
        Initialize embedding generator.

        Args:
            ollama_client: Ollama client (creates default if None)
            model: Embedding model name
            use_cache: Whether to cache embeddings
            cache_max_size: Maximum cache entries
        """
        self.client = ollama_client or get_ollama_client()
        self.model = model
        self.use_cache = use_cache
        self.cache_max_size = cache_max_size
        self._cache: Dict[str, List[float]] = {}
        self._dimension: Optional[int] = None

    async def generate(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Normalize text
        text = self._normalize_text(text)

        # Check cache
        if self.use_cache and text in self._cache:
            return self._cache[text]

        try:
            embedding = await self.client.embed(text, model=self.model)

            # Track dimension
            if self._dimension is None and embedding:
                self._dimension = len(embedding)

            # Cache result
            if self.use_cache and embedding:
                self._manage_cache()
                self._cache[text] = embedding

            return embedding

        except Exception as e:
            # Return zero vector on error
            return self._zero_vector()

    def generate_sync(self, text: str) -> List[float]:
        """Synchronous version of generate"""
        return asyncio.get_event_loop().run_until_complete(
            self.generate(text)
        )

    async def generate_batch(
        self,
        texts: List[str],
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel.

        Args:
            texts: List of texts to embed
            max_concurrent: Maximum concurrent requests

        Returns:
            List of embedding vectors
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_one(text: str) -> List[float]:
            async with semaphore:
                return await self.generate(text)

        tasks = [embed_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    def get_dimension(self) -> int:
        """Get embedding dimension (may require generating one first)"""
        if self._dimension is not None:
            return self._dimension
        return EMBEDDING_DIMENSION

    def get_generator_function(self) -> Callable[[str], List[float]]:
        """
        Get a callable generator function for use with MemoryStream.

        Returns:
            Function that takes text and returns embedding
        """
        def generator(text: str) -> List[float]:
            return self.generate_sync(text)
        return generator

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent embedding"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Truncate very long texts
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    def _zero_vector(self) -> List[float]:
        """Return zero vector of correct dimension"""
        dim = self._dimension or EMBEDDING_DIMENSION
        return [0.0] * dim

    def _manage_cache(self) -> None:
        """Remove old cache entries if cache is too large"""
        if len(self._cache) >= self.cache_max_size:
            # Remove half the cache
            keys = list(self._cache.keys())
            for key in keys[:len(keys)//2]:
                del self._cache[key]


class EmbeddingSimilarity:
    """
    Utility class for computing embedding similarity.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def normalized_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute normalized similarity (0 to 1 range).

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        sim = EmbeddingSimilarity.cosine_similarity(vec1, vec2)
        return (sim + 1) / 2

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Distance (0 = identical)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return float('inf')

        return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5

    @staticmethod
    def find_most_similar(
        query: List[float],
        candidates: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar vectors to a query.

        Args:
            query: Query vector
            candidates: List of candidate vectors
            top_k: Number of top results to return

        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        similarities = []
        for i, candidate in enumerate(candidates):
            sim = EmbeddingSimilarity.cosine_similarity(query, candidate)
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Convenience function
async def generate_embedding(
    text: str,
    ollama_client: Optional[OllamaClient] = None,
    model: str = "nomic-embed-text"
) -> List[float]:
    """
    Quick embedding generation function.

    Args:
        text: Text to embed
        ollama_client: Optional client
        model: Embedding model

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator(ollama_client, model)
    return await generator.generate(text)
