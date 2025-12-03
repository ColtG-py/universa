"""
Caching Systems
LRU caches for LLM responses, embeddings, and memory retrieval.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json


@dataclass
class CacheEntry:
    """A single cache entry"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class LRUCache:
    """
    Generic LRU cache implementation.

    Features:
    - Configurable max size
    - Optional TTL per entry
    - Access statistics
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            self._cache.pop(key)
            self._stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._stats["hits"] += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache"""
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self._stats["evictions"] += 1

        # Add entry
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl or self.default_ttl,
        )
        self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self._cache:
            self._cache.pop(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all entries"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self._stats,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            self._cache.pop(key)
        return len(expired_keys)


class ResponseCache(LRUCache):
    """
    Cache for LLM responses.

    Caches responses based on prompt hash to avoid redundant LLM calls.
    """

    def __init__(
        self,
        max_size: int = 500,
        default_ttl: float = 3600.0,  # 1 hour default
    ):
        """
        Initialize response cache.

        Args:
            max_size: Maximum cached responses
            default_ttl: Default TTL in seconds
        """
        super().__init__(max_size=max_size, default_ttl=default_ttl)

    def _make_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
    ) -> str:
        """Create cache key from prompt parameters"""
        # Include temperature in key (different temp = different response)
        temp_bucket = round(temperature, 1)  # Round to reduce key variations
        key_data = f"{model}:{temp_bucket}:{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Get cached response.

        Args:
            prompt: The prompt text
            model: Model used
            temperature: Temperature setting

        Returns:
            Cached response or None
        """
        # Only cache for low temperatures (deterministic)
        if temperature > 0.5:
            return None

        key = self._make_key(prompt, model, temperature)
        return self.get(key)

    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.7,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache a response.

        Args:
            prompt: The prompt text
            model: Model used
            response: The response to cache
            temperature: Temperature setting
            ttl: Optional custom TTL
        """
        # Only cache for low temperatures
        if temperature > 0.5:
            return

        key = self._make_key(prompt, model, temperature)
        self.set(key, response, ttl)


class EmbeddingCache(LRUCache):
    """
    Cache for embeddings.

    Embeddings are deterministic so can be cached indefinitely.
    """

    def __init__(
        self,
        max_size: int = 2000,
        default_ttl: Optional[float] = None,  # No expiry for embeddings
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum cached embeddings
            default_ttl: Default TTL (None = no expiry)
        """
        super().__init__(max_size=max_size, default_ttl=default_ttl)

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model"""
        key_data = f"{model}:{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get_embedding(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> Optional[List[float]]:
        """
        Get cached embedding.

        Args:
            text: Text to embed
            model: Embedding model

        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model)
        return self.get(key)

    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "nomic-embed-text",
    ) -> None:
        """
        Cache an embedding.

        Args:
            text: Text that was embedded
            embedding: The embedding vector
            model: Embedding model
        """
        key = self._make_key(text, model)
        self.set(key, embedding)

    def get_batch(
        self,
        texts: List[str],
        model: str = "nomic-embed-text",
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Get cached embeddings for batch, return uncached texts.

        Args:
            texts: Texts to lookup
            model: Embedding model

        Returns:
            Tuple of (cached_embeddings, uncached_texts)
        """
        cached = []
        uncached = []

        for text in texts:
            embedding = self.get_embedding(text, model)
            if embedding is not None:
                cached.append(embedding)
            else:
                uncached.append(text)

        return cached, uncached


class MemoryCache:
    """
    Cache for memory retrieval results.

    Caches query results to avoid redundant database queries.
    Short TTL since memories change frequently.
    """

    def __init__(
        self,
        max_size: int = 200,
        default_ttl: float = 60.0,  # 1 minute TTL
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum cached queries
            default_ttl: TTL in seconds
        """
        self._cache = LRUCache(max_size=max_size, default_ttl=default_ttl)

    def _make_key(
        self,
        agent_id: str,
        query: str,
        limit: int,
    ) -> str:
        """Create cache key"""
        key_data = f"{agent_id}:{limit}:{query}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
    ) -> Optional[List[Any]]:
        """
        Get cached memories.

        Args:
            agent_id: Agent's ID
            query: Search query
            limit: Result limit

        Returns:
            Cached memories or None
        """
        key = self._make_key(agent_id, query, limit)
        return self._cache.get(key)

    def cache_memories(
        self,
        agent_id: str,
        query: str,
        memories: List[Any],
        limit: int = 10,
    ) -> None:
        """
        Cache memory results.

        Args:
            agent_id: Agent's ID
            query: Search query
            memories: Retrieved memories
            limit: Result limit
        """
        key = self._make_key(agent_id, query, limit)
        self._cache.set(key, memories)

    def invalidate_agent(self, agent_id: str) -> int:
        """
        Invalidate all cached memories for an agent.

        Call this when agent's memories change.

        Args:
            agent_id: Agent's ID

        Returns:
            Number of entries invalidated
        """
        # Find and delete all keys for this agent
        keys_to_delete = [
            key for key in self._cache._cache.keys()
            if agent_id in key
        ]
        for key in keys_to_delete:
            self._cache.delete(key)
        return len(keys_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()
