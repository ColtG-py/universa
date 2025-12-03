"""
Simulation Profiler
Performance profiling for agent simulations.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
from contextlib import contextmanager
import time
import statistics


class ProfileCategory(str, Enum):
    """Categories of profiled operations"""
    LLM_CALL = "llm_call"
    EMBEDDING = "embedding"
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_WRITE = "memory_write"
    REFLECTION = "reflection"
    PLANNING = "planning"
    PERCEPTION = "perception"
    ACTION = "action"
    TICK = "tick"
    AGENT_CYCLE = "agent_cycle"


@dataclass
class TimingRecord:
    """A single timing record"""
    record_id: UUID = field(default_factory=uuid4)
    category: ProfileCategory = ProfileCategory.TICK
    operation: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProfileResult:
    """Result of profiling a simulation"""
    result_id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Timing records
    records: List[TimingRecord] = field(default_factory=list)

    # Aggregated stats by category
    category_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Overall stats
    total_duration_ms: float = 0.0
    total_operations: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    llm_calls: int = 0
    embedding_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "result_id": str(self.result_id),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "total_operations": self.total_operations,
            "category_stats": self.category_stats,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            "Simulation Profile Summary",
            "=" * 40,
            f"Duration: {self.total_duration_ms:.1f}ms",
            f"Operations: {self.total_operations}",
            f"LLM Calls: {self.llm_calls}",
            f"Embedding Calls: {self.embedding_calls}",
            "",
            "Breakdown by Category:",
        ]

        for cat, stats in sorted(self.category_stats.items()):
            lines.append(f"  {cat}:")
            lines.append(f"    Count: {stats.get('count', 0)}")
            lines.append(f"    Total: {stats.get('total_ms', 0):.1f}ms")
            lines.append(f"    Avg: {stats.get('avg_ms', 0):.1f}ms")
            if stats.get('max_ms'):
                lines.append(f"    Max: {stats.get('max_ms', 0):.1f}ms")

        return "\n".join(lines)


class SimulationProfiler:
    """
    Profiles simulation performance.

    Features:
    - Timing of all major operations
    - Category-based aggregation
    - Bottleneck identification
    - Resource usage tracking
    """

    def __init__(self):
        """Initialize profiler"""
        self._records: List[TimingRecord] = []
        self._active_timers: Dict[str, float] = {}
        self._start_time: Optional[datetime] = None
        self._enabled = True

        # Counters
        self._llm_calls = 0
        self._embedding_calls = 0

    def enable(self) -> None:
        """Enable profiling"""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling"""
        self._enabled = False

    def start_session(self) -> None:
        """Start a profiling session"""
        self._records = []
        self._active_timers = {}
        self._start_time = datetime.utcnow()
        self._llm_calls = 0
        self._embedding_calls = 0

    def end_session(self) -> ProfileResult:
        """End session and return results"""
        end_time = datetime.utcnow()

        result = ProfileResult(
            start_time=self._start_time or end_time,
            end_time=end_time,
            records=self._records.copy(),
            llm_calls=self._llm_calls,
            embedding_calls=self._embedding_calls,
        )

        # Calculate total duration
        if self._start_time:
            result.total_duration_ms = (
                (end_time - self._start_time).total_seconds() * 1000
            )

        result.total_operations = len(self._records)

        # Aggregate stats by category
        result.category_stats = self._aggregate_stats()

        return result

    @contextmanager
    def time_operation(
        self,
        category: ProfileCategory,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for timing an operation.

        Usage:
            with profiler.time_operation(ProfileCategory.LLM_CALL, "generate"):
                result = await llm.generate(prompt)
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            record = TimingRecord(
                category=category,
                operation=operation,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
            self._records.append(record)

            # Track specific counters
            if category == ProfileCategory.LLM_CALL:
                self._llm_calls += 1
            elif category == ProfileCategory.EMBEDDING:
                self._embedding_calls += 1

    def start_timer(self, timer_id: str) -> None:
        """Start a named timer"""
        if self._enabled:
            self._active_timers[timer_id] = time.perf_counter()

    def stop_timer(
        self,
        timer_id: str,
        category: ProfileCategory,
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Stop a named timer and record the duration"""
        if not self._enabled or timer_id not in self._active_timers:
            return 0.0

        start = self._active_timers.pop(timer_id)
        duration_ms = (time.perf_counter() - start) * 1000

        record = TimingRecord(
            category=category,
            operation=operation or timer_id,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._records.append(record)

        if category == ProfileCategory.LLM_CALL:
            self._llm_calls += 1
        elif category == ProfileCategory.EMBEDDING:
            self._embedding_calls += 1

        return duration_ms

    def record_event(
        self,
        category: ProfileCategory,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a timing event directly"""
        if not self._enabled:
            return

        record = TimingRecord(
            category=category,
            operation=operation,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._records.append(record)

    def _aggregate_stats(self) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics by category"""
        stats: Dict[str, Dict[str, float]] = {}

        # Group by category
        by_category: Dict[str, List[float]] = {}
        for record in self._records:
            cat = record.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(record.duration_ms)

        # Calculate stats for each category
        for cat, durations in by_category.items():
            stats[cat] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": statistics.mean(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
            }
            if len(durations) > 1:
                stats[cat]["std_ms"] = statistics.stdev(durations)

        return stats

    def get_bottlenecks(
        self,
        threshold_ms: float = 100.0,
        top_n: int = 10,
    ) -> List[TimingRecord]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_ms: Minimum duration to consider
            top_n: Number of top bottlenecks to return

        Returns:
            List of slowest operations
        """
        slow_operations = [
            r for r in self._records
            if r.duration_ms >= threshold_ms
        ]

        return sorted(
            slow_operations,
            key=lambda r: r.duration_ms,
            reverse=True,
        )[:top_n]

    def get_category_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown by category"""
        stats = self._aggregate_stats()
        total = sum(s.get("total_ms", 0) for s in stats.values())

        if total == 0:
            return {}

        return {
            cat: (s.get("total_ms", 0) / total) * 100
            for cat, s in stats.items()
        }

    def compare_runs(
        self,
        other: "ProfileResult",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare current stats with another profile result.

        Args:
            other: Another profile result to compare against

        Returns:
            Dict of category differences
        """
        current = self._aggregate_stats()
        comparison = {}

        all_categories = set(current.keys()) | set(other.category_stats.keys())

        for cat in all_categories:
            curr_stats = current.get(cat, {})
            other_stats = other.category_stats.get(cat, {})

            comparison[cat] = {
                "count_diff": curr_stats.get("count", 0) - other_stats.get("count", 0),
                "avg_ms_diff": curr_stats.get("avg_ms", 0) - other_stats.get("avg_ms", 0),
                "total_ms_diff": curr_stats.get("total_ms", 0) - other_stats.get("total_ms", 0),
            }

        return comparison

    def get_records(
        self,
        category: Optional[ProfileCategory] = None,
        min_duration_ms: float = 0.0,
    ) -> List[TimingRecord]:
        """Get filtered timing records"""
        records = self._records

        if category:
            records = [r for r in records if r.category == category]

        if min_duration_ms > 0:
            records = [r for r in records if r.duration_ms >= min_duration_ms]

        return records

    def reset(self) -> None:
        """Reset profiler state"""
        self._records = []
        self._active_timers = {}
        self._start_time = None
        self._llm_calls = 0
        self._embedding_calls = 0


class TickProfiler:
    """
    Specialized profiler for simulation ticks.

    Tracks per-tick performance and agent-level metrics.
    """

    def __init__(self):
        """Initialize tick profiler"""
        self._tick_durations: List[float] = []
        self._agent_durations: Dict[UUID, List[float]] = {}
        self._current_tick_start: Optional[float] = None

    def start_tick(self) -> None:
        """Mark the start of a tick"""
        self._current_tick_start = time.perf_counter()

    def end_tick(self) -> float:
        """Mark the end of a tick, return duration"""
        if self._current_tick_start is None:
            return 0.0

        duration = (time.perf_counter() - self._current_tick_start) * 1000
        self._tick_durations.append(duration)
        self._current_tick_start = None
        return duration

    def record_agent_cycle(
        self,
        agent_id: UUID,
        duration_ms: float,
    ) -> None:
        """Record an agent's cycle duration"""
        if agent_id not in self._agent_durations:
            self._agent_durations[agent_id] = []
        self._agent_durations[agent_id].append(duration_ms)

    def get_tick_stats(self) -> Dict[str, float]:
        """Get tick timing statistics"""
        if not self._tick_durations:
            return {"count": 0}

        return {
            "count": len(self._tick_durations),
            "total_ms": sum(self._tick_durations),
            "avg_ms": statistics.mean(self._tick_durations),
            "min_ms": min(self._tick_durations),
            "max_ms": max(self._tick_durations),
            "std_ms": statistics.stdev(self._tick_durations) if len(self._tick_durations) > 1 else 0,
        }

    def get_agent_stats(self, agent_id: UUID) -> Dict[str, float]:
        """Get stats for a specific agent"""
        durations = self._agent_durations.get(agent_id, [])

        if not durations:
            return {"cycles": 0}

        return {
            "cycles": len(durations),
            "total_ms": sum(durations),
            "avg_ms": statistics.mean(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }

    def get_slowest_agents(self, top_n: int = 5) -> List[tuple]:
        """Get agents with highest average cycle time"""
        agent_avgs = []

        for agent_id, durations in self._agent_durations.items():
            if durations:
                avg = statistics.mean(durations)
                agent_avgs.append((agent_id, avg))

        return sorted(agent_avgs, key=lambda x: x[1], reverse=True)[:top_n]

    def reset(self) -> None:
        """Reset tick profiler"""
        self._tick_durations = []
        self._agent_durations = {}
        self._current_tick_start = None
