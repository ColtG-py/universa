"""
Integration Tests
End-to-end tests for the agent simulation system.

Run with: python3.10 -m pytest tests/test_integration.py -v
"""

import pytest
import sys
import os
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Agent components
from agents.memory.memory_stream import MemoryStream
from agents.llm.ollama_client import OllamaClient, get_ollama_client
from agents.graph.state import AgentGraphState, PerceptionData
from agents.simulation.orchestrator import SimulationOrchestrator
from agents.simulation.time_manager import TimeManager, SimulationTime
from agents.world.interface import WorldInterface, LocationData

# Optimization components
from agents.optimization.cache import ResponseCache, EmbeddingCache, MemoryCache, LRUCache
from agents.optimization.batching import LLMBatcher, EmbeddingBatcher
from agents.optimization.profiler import SimulationProfiler, ProfileCategory, TickProfiler

# Evaluation components
from agents.evaluation.metrics import BelievabilityMetrics
from agents.evaluation.interview import AgentInterviewer
from agents.evaluation.behavior import BehaviorConsistencyChecker
from agents.evaluation.memory_test import MemoryAccuracyTester


class MockWorldInterface(WorldInterface):
    """Mock world for integration testing"""

    def __init__(self):
        super().__init__(world_state=None)
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._locations: Dict[str, LocationData] = {}
        self.action_log: List[Dict] = []
        self._setup_test_world()

    def _setup_test_world(self):
        """Setup a simple test world"""
        self._locations["village_center"] = LocationData(
            x=100, y=100, chunk_x=0, chunk_y=0,
            biome_type="temperate_grassland",
            temperature_c=18.0,
            has_road=True,
            settlement_id=1,
            settlement_type="village",
            faction_name="Riverside",
        )

    def query_location(self, x: int, y: int, world_state=None) -> Optional[LocationData]:
        """Return test location data"""
        for name, loc in self._locations.items():
            if abs(loc.x - x) <= 10 and abs(loc.y - y) <= 10:
                return loc
        return LocationData(
            x=x, y=y, chunk_x=x // 256, chunk_y=y // 256,
            biome_type="temperate_grassland",
            temperature_c=18.0,
        )


@pytest.fixture
def ollama_client():
    """Create Ollama client for tests"""
    return get_ollama_client()


@pytest.fixture
def world():
    """Create mock world"""
    return MockWorldInterface()


@pytest.fixture
def profiler():
    """Create profiler"""
    return SimulationProfiler()


class TestMemoryStream:
    """Tests for memory stream operations"""

    def test_create_memory_stream(self):
        """Test creating a memory stream"""
        agent_id = uuid4()
        memory = MemoryStream(agent_id=agent_id)
        assert memory.agent_id == agent_id


class TestOptimization:
    """Tests for optimization components"""

    def test_lru_cache_basic(self):
        """Test basic LRU cache operations"""
        cache = LRUCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lru_cache_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 to make it recent
        cache.get("key1")

        # Add key3, should evict key2
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"

    def test_response_cache(self):
        """Test LLM response caching"""
        cache = ResponseCache(max_size=100)

        cache.cache_response(
            prompt="What is the weather?",
            model="qwen2.5:7b",
            response="The weather is sunny.",
            temperature=0.3,
        )

        cached = cache.get_response(
            prompt="What is the weather?",
            model="qwen2.5:7b",
            temperature=0.3,
        )
        assert cached == "The weather is sunny."

    def test_response_cache_high_temperature(self):
        """Test that high temperature responses aren't cached"""
        cache = ResponseCache()

        cache.cache_response(
            prompt="Tell me a story",
            model="qwen2.5:7b",
            response="Once upon a time...",
            temperature=0.8,
        )

        cached = cache.get_response(
            prompt="Tell me a story",
            model="qwen2.5:7b",
            temperature=0.8,
        )
        assert cached is None

    def test_embedding_cache(self):
        """Test embedding caching"""
        cache = EmbeddingCache(max_size=100)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.cache_embedding(
            text="Hello world",
            embedding=embedding,
            model="nomic-embed-text",
        )

        cached = cache.get_embedding(
            text="Hello world",
            model="nomic-embed-text",
        )
        assert cached == embedding

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=10)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_ttl(self):
        """Test cache TTL expiration"""
        import time
        cache = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("key1") is None  # Expired


class TestProfiler:
    """Tests for profiler components"""

    def test_profiler_session(self):
        """Test profiler session lifecycle"""
        profiler = SimulationProfiler()

        profiler.start_session()
        profiler.record_event(ProfileCategory.LLM_CALL, "test", 100.0)
        result = profiler.end_session()

        assert result.total_operations == 1
        assert ProfileCategory.LLM_CALL.value in result.category_stats

    def test_profiler_timing_context(self, profiler):
        """Test profiler timing with context manager"""
        import time

        profiler.start_session()

        with profiler.time_operation(ProfileCategory.LLM_CALL, "test_call"):
            time.sleep(0.01)

        result = profiler.end_session()

        assert result.total_operations == 1
        assert result.llm_calls == 1
        stats = result.category_stats[ProfileCategory.LLM_CALL.value]
        assert stats["avg_ms"] >= 10  # At least 10ms

    def test_tick_profiler(self):
        """Test tick profiler"""
        tick_profiler = TickProfiler()

        tick_profiler.start_tick()
        import time
        time.sleep(0.01)
        duration = tick_profiler.end_tick()

        assert duration >= 10  # At least 10ms

        stats = tick_profiler.get_tick_stats()
        assert stats["count"] == 1

    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        profiler = SimulationProfiler()
        profiler.start_session()

        # Add some operations with varying durations
        profiler.record_event(ProfileCategory.LLM_CALL, "fast", 50.0)
        profiler.record_event(ProfileCategory.LLM_CALL, "slow", 200.0)
        profiler.record_event(ProfileCategory.LLM_CALL, "medium", 100.0)

        bottlenecks = profiler.get_bottlenecks(threshold_ms=100.0, top_n=2)

        assert len(bottlenecks) == 2
        assert bottlenecks[0].duration_ms == 200.0

    def test_category_breakdown(self):
        """Test category breakdown calculation"""
        profiler = SimulationProfiler()
        profiler.start_session()

        profiler.record_event(ProfileCategory.LLM_CALL, "llm1", 100.0)
        profiler.record_event(ProfileCategory.LLM_CALL, "llm2", 100.0)
        profiler.record_event(ProfileCategory.EMBEDDING, "emb1", 100.0)
        profiler.record_event(ProfileCategory.MEMORY_RETRIEVAL, "mem1", 100.0)

        breakdown = profiler.get_category_breakdown()

        # Should have even distribution
        assert ProfileCategory.LLM_CALL.value in breakdown
        assert breakdown[ProfileCategory.LLM_CALL.value] == 50.0  # 200/400 = 50%


class TestEvaluation:
    """Tests for evaluation components"""

    def test_behavior_checker_init(self):
        """Test behavior checker initialization"""
        checker = BehaviorConsistencyChecker()
        assert checker is not None

    def test_anomaly_detection_normal(self):
        """Test that normal behavior doesn't trigger anomalies"""
        checker = BehaviorConsistencyChecker()

        history = [
            "Worked at the forge",
            "Had lunch at the tavern",
            "Continued crafting",
        ]

        anomalies = checker.detect_anomalies(
            agent_id=uuid4(),
            current_action="Sold a horseshoe",
            action_history=history,
        )
        assert len(anomalies) == 0

    def test_anomaly_detection_loop(self):
        """Test detection of repetitive behavior"""
        checker = BehaviorConsistencyChecker()

        history = [
            "Walked to the market",
            "Walked to the market",
            "Walked to the market",
        ]

        anomalies = checker.detect_anomalies(
            agent_id=uuid4(),
            current_action="Walked to the market",
            action_history=history,
        )
        assert len(anomalies) > 0
        assert "Repetitive" in anomalies[0]

    def test_pattern_detection(self):
        """Test behavior pattern detection"""
        checker = BehaviorConsistencyChecker()

        actions = [
            {"description": "Ate breakfast", "timestamp": datetime.utcnow()},
            {"description": "Worked at forge", "timestamp": datetime.utcnow()},
            {"description": "Ate lunch", "timestamp": datetime.utcnow()},
            {"description": "Worked more", "timestamp": datetime.utcnow()},
            {"description": "Ate dinner", "timestamp": datetime.utcnow()},
        ]

        patterns = checker._detect_patterns(actions)
        assert len(patterns) > 0


class TestMemoryTester:
    """Tests for memory accuracy testing"""

    def test_memory_tester_init(self):
        """Test memory tester initialization"""
        tester = MemoryAccuracyTester()
        assert tester is not None

    def test_heuristic_scores_no_memories(self):
        """Test heuristic scoring with no retrieved memories"""
        tester = MemoryAccuracyTester()

        scores = tester._heuristic_scores(
            expected=["some event"],
            retrieved=[],
        )

        assert scores["recall"] == 0.0
        assert scores["relevance"] == 0.0

    def test_heuristic_scores_with_overlap(self):
        """Test heuristic scoring with overlapping content"""
        tester = MemoryAccuracyTester()

        scores = tester._heuristic_scores(
            expected=["The blacksmith forged a sword"],
            retrieved=["Elena forged a new sword at the forge"],
        )

        assert scores["recall"] > 0
        assert scores["accuracy"] > 0


class TestInterview:
    """Tests for agent interview system"""

    def test_interviewer_init(self, ollama_client):
        """Test interviewer initialization"""
        interviewer = AgentInterviewer(ollama_client=ollama_client)
        assert interviewer is not None


class TestBelievabilityMetrics:
    """Tests for believability metrics"""

    def test_metrics_init(self):
        """Test metrics initialization"""
        metrics = BelievabilityMetrics()
        assert metrics is not None


class TestSimulationComponents:
    """Tests for simulation components"""

    def test_orchestrator_creation(self):
        """Test creating orchestrator"""
        time_manager = TimeManager(
            start_time=SimulationTime(year=1, month=6, day=15, hour=8, minute=0)
        )
        orchestrator = SimulationOrchestrator(time_manager=time_manager)
        assert orchestrator is not None

    def test_time_manager_tick(self):
        """Test time manager tick operation"""
        start = SimulationTime(year=1, month=6, day=15, hour=8, minute=0)
        manager = TimeManager(start_time=start, tick_minutes=15)

        # Get initial state
        initial = start.to_dict()
        assert initial["hour"] == 8
        assert initial["minute"] == 0

        # Advance time
        result = manager.tick()
        assert "current" in result
        assert "tick" in result
        assert result["current"]["minute"] == 15

    def test_simulation_time_advance(self):
        """Test SimulationTime advancement"""
        time = SimulationTime(year=1, month=6, day=15, hour=23, minute=45)

        # Advance 30 minutes - should wrap to next day
        time.advance(30)

        assert time.hour == 0
        assert time.minute == 15
        assert time.day == 16

    def test_time_of_day_detection(self):
        """Test time of day detection"""
        morning = SimulationTime(year=1, month=1, day=1, hour=8, minute=0)
        night = SimulationTime(year=1, month=1, day=1, hour=23, minute=0)

        assert morning.is_daytime()
        assert not morning.is_nighttime()
        assert night.is_nighttime()
        assert not night.is_daytime()


class TestWorldInterface:
    """Tests for world interface"""

    def test_mock_world_creation(self, world):
        """Test creating mock world"""
        assert world is not None
        assert len(world._locations) > 0

    def test_location_query(self, world):
        """Test querying location data"""
        loc = world.query_location(100, 100)
        assert loc is not None
        assert loc.biome_type == "temperate_grassland"

    def test_unknown_location(self, world):
        """Test querying unknown location returns default"""
        loc = world.query_location(9999, 9999)
        assert loc is not None
        assert loc.x == 9999


# Summary test
class TestIntegrationSummary:
    """Summary tests to verify all major components can be instantiated"""

    def test_all_cache_types(self):
        """Test all cache types can be created"""
        lru = LRUCache(max_size=10)
        response = ResponseCache(max_size=10)
        embedding = EmbeddingCache(max_size=10)
        memory = MemoryCache(max_size=10)

        assert lru is not None
        assert response is not None
        assert embedding is not None
        assert memory is not None

    def test_all_profiler_types(self):
        """Test all profiler types can be created"""
        sim_profiler = SimulationProfiler()
        tick_profiler = TickProfiler()

        assert sim_profiler is not None
        assert tick_profiler is not None

    def test_all_evaluation_types(self):
        """Test all evaluation types can be created"""
        behavior = BehaviorConsistencyChecker()
        memory_tester = MemoryAccuracyTester()
        metrics = BelievabilityMetrics()

        assert behavior is not None
        assert memory_tester is not None
        assert metrics is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
