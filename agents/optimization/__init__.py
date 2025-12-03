"""
Performance Optimization Module
Caching, batching, and optimization utilities for agent simulation.
"""

from agents.optimization.cache import (
    ResponseCache,
    EmbeddingCache,
    MemoryCache,
    LRUCache,
)
from agents.optimization.batching import (
    LLMBatcher,
    EmbeddingBatcher,
    RequestCoalescer,
)
from agents.optimization.profiler import (
    SimulationProfiler,
    ProfileResult,
    ProfileCategory,
    TickProfiler,
)

__all__ = [
    # Caching
    "LRUCache",
    "ResponseCache",
    "EmbeddingCache",
    "MemoryCache",
    # Batching
    "LLMBatcher",
    "EmbeddingBatcher",
    "RequestCoalescer",
    # Profiling
    "SimulationProfiler",
    "ProfileResult",
    "ProfileCategory",
    "TickProfiler",
]
