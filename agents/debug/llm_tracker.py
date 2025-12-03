"""
LLM Call Tracker
Tracks all LLM calls for debugging and optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMCall:
    """Record of a single LLM call."""
    call_id: str
    agent_id: UUID
    timestamp: datetime
    purpose: str  # e.g., "importance_scoring", "reflection", "planning", "reaction"
    model: str
    prompt: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "agent_id": str(self.agent_id),
            "timestamp": self.timestamp.isoformat(),
            "purpose": self.purpose,
            "model": self.model,
            "prompt_summary": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "response_summary": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


class LLMCallTracker:
    """
    Tracks LLM calls across all agents.

    Provides:
    - Per-agent call history
    - Global call statistics
    - Token usage tracking
    - Performance metrics
    """

    def __init__(self, max_history_per_agent: int = 100):
        self.max_history = max_history_per_agent

        # Per-agent call history (agent_id -> deque of LLMCalls)
        self._agent_history: Dict[UUID, deque] = {}

        # Global statistics
        self._total_calls = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_duration_ms = 0.0
        self._calls_by_purpose: Dict[str, int] = {}
        self._errors = 0

        # Recent calls (global)
        self._recent_calls: deque = deque(maxlen=500)

    def record_call(
        self,
        agent_id: UUID,
        purpose: str,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMCall:
        """
        Record an LLM call.

        Args:
            agent_id: Agent that made the call
            purpose: What the call was for
            model: Model used
            prompt: Input prompt
            response: LLM response
            tokens_in: Input token count
            tokens_out: Output token count
            duration_ms: Call duration
            success: Whether call succeeded
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            The recorded LLMCall
        """
        import uuid

        call = LLMCall(
            call_id=str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            purpose=purpose,
            model=model,
            prompt=prompt,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        # Add to agent history
        if agent_id not in self._agent_history:
            self._agent_history[agent_id] = deque(maxlen=self.max_history)
        self._agent_history[agent_id].append(call)

        # Add to recent calls
        self._recent_calls.append(call)

        # Update statistics
        self._total_calls += 1
        self._total_tokens_in += tokens_in
        self._total_tokens_out += tokens_out
        self._total_duration_ms += duration_ms
        self._calls_by_purpose[purpose] = self._calls_by_purpose.get(purpose, 0) + 1
        if not success:
            self._errors += 1

        return call

    def get_agent_history(
        self,
        agent_id: UUID,
        limit: int = 50,
        purpose: Optional[str] = None
    ) -> List[LLMCall]:
        """
        Get LLM call history for an agent.

        Args:
            agent_id: Agent to get history for
            limit: Maximum calls to return
            purpose: Filter by purpose

        Returns:
            List of LLMCalls, most recent first
        """
        history = self._agent_history.get(agent_id, deque())
        calls = list(history)

        # Filter by purpose if specified
        if purpose:
            calls = [c for c in calls if c.purpose == purpose]

        # Return most recent first
        return list(reversed(calls))[:limit]

    def get_recent_calls(
        self,
        limit: int = 50,
        agent_id: Optional[UUID] = None,
        purpose: Optional[str] = None
    ) -> List[LLMCall]:
        """
        Get recent LLM calls.

        Args:
            limit: Maximum calls to return
            agent_id: Filter by agent
            purpose: Filter by purpose

        Returns:
            List of recent LLMCalls
        """
        calls = list(self._recent_calls)

        if agent_id:
            calls = [c for c in calls if c.agent_id == agent_id]
        if purpose:
            calls = [c for c in calls if c.purpose == purpose]

        return list(reversed(calls))[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get LLM usage statistics.

        Returns:
            Statistics dictionary
        """
        avg_duration = (
            self._total_duration_ms / self._total_calls
            if self._total_calls > 0 else 0
        )

        return {
            "total_calls": self._total_calls,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "total_tokens": self._total_tokens_in + self._total_tokens_out,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": avg_duration,
            "error_count": self._errors,
            "error_rate": self._errors / self._total_calls if self._total_calls > 0 else 0,
            "calls_by_purpose": dict(self._calls_by_purpose),
            "agents_tracked": len(self._agent_history),
        }

    def get_agent_stats(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Get LLM statistics for a specific agent.

        Args:
            agent_id: Agent to get stats for

        Returns:
            Agent-specific statistics
        """
        history = self._agent_history.get(agent_id, deque())
        calls = list(history)

        if not calls:
            return {
                "total_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "avg_duration_ms": 0,
                "calls_by_purpose": {},
            }

        tokens_in = sum(c.tokens_in for c in calls)
        tokens_out = sum(c.tokens_out for c in calls)
        total_duration = sum(c.duration_ms for c in calls)

        purpose_counts = {}
        for call in calls:
            purpose_counts[call.purpose] = purpose_counts.get(call.purpose, 0) + 1

        return {
            "total_calls": len(calls),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(calls),
            "calls_by_purpose": purpose_counts,
        }

    def clear_agent_history(self, agent_id: UUID) -> None:
        """Clear history for an agent."""
        if agent_id in self._agent_history:
            self._agent_history[agent_id].clear()

    def clear_all(self) -> None:
        """Clear all tracking data."""
        self._agent_history.clear()
        self._recent_calls.clear()
        self._total_calls = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_duration_ms = 0.0
        self._calls_by_purpose.clear()
        self._errors = 0


# Global tracker instance
_tracker: Optional[LLMCallTracker] = None


def get_tracker() -> LLMCallTracker:
    """Get the global LLM call tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LLMCallTracker()
    return _tracker


def track_llm_call(
    agent_id: UUID,
    purpose: str,
    model: str,
    prompt: str,
    response: str,
    **kwargs
) -> LLMCall:
    """Convenience function to track an LLM call."""
    return get_tracker().record_call(
        agent_id=agent_id,
        purpose=purpose,
        model=model,
        prompt=prompt,
        response=response,
        **kwargs
    )
