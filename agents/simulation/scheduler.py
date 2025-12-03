"""
Agent Scheduler
Priority-based agent action scheduling.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import heapq
import asyncio


class ActionPriority(int, Enum):
    """Priority levels for actions"""
    CRITICAL = 0     # Must happen immediately (danger response)
    HIGH = 1         # Important (social interaction, combat)
    NORMAL = 2       # Regular activities
    LOW = 3          # Can be delayed (idle activities)
    BACKGROUND = 4   # Happens when nothing else to do


class ActionStatus(str, Enum):
    """Status of a scheduled action"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledAction:
    """An action scheduled for execution"""
    action_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    priority: ActionPriority = ActionPriority.NORMAL
    action_type: str = ""
    description: str = ""
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    execute_at: Optional[datetime] = None  # None = ASAP
    status: ActionStatus = ActionStatus.PENDING

    # Execution
    executor: Optional[Callable[..., Awaitable[Any]]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None

    # Timing
    timeout_seconds: float = 60.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __lt__(self, other: "ScheduledAction") -> bool:
        """Compare for priority queue (lower priority value = higher priority)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Same priority: earlier scheduled goes first
        return self.scheduled_at < other.scheduled_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_id": str(self.action_id),
            "agent_id": str(self.agent_id),
            "priority": self.priority.value,
            "action_type": self.action_type,
            "description": self.description,
            "status": self.status.value,
            "scheduled_at": self.scheduled_at.isoformat(),
        }


class AgentScheduler:
    """
    Schedules and executes agent actions.

    Features:
    - Priority queue for actions
    - Per-agent concurrency limits
    - Timeout handling
    - Parallel execution with resource limits
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_per_agent: int = 1,
    ):
        """
        Initialize scheduler.

        Args:
            max_concurrent: Maximum total concurrent actions
            max_per_agent: Maximum concurrent actions per agent
        """
        self.max_concurrent = max_concurrent
        self.max_per_agent = max_per_agent

        # Priority queue of pending actions
        self._queue: List[ScheduledAction] = []
        heapq.heapify(self._queue)

        # Currently running actions
        self._running: Dict[UUID, ScheduledAction] = {}

        # Per-agent tracking
        self._agent_running: Dict[UUID, int] = {}

        # Completed actions (limited history)
        self._completed: List[ScheduledAction] = []
        self._max_history = 1000

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running_lock = asyncio.Lock()

        # Stats
        self._stats = {
            "total_scheduled": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
        }

    def schedule(
        self,
        agent_id: UUID,
        action_type: str,
        description: str,
        priority: ActionPriority = ActionPriority.NORMAL,
        executor: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        execute_at: Optional[datetime] = None,
        timeout: float = 60.0,
    ) -> ScheduledAction:
        """
        Schedule an action for execution.

        Args:
            agent_id: Agent performing the action
            action_type: Type of action
            description: What the action does
            priority: Execution priority
            executor: Async function to execute
            parameters: Parameters for executor
            execute_at: When to execute (None = ASAP)
            timeout: Timeout in seconds

        Returns:
            Scheduled action
        """
        action = ScheduledAction(
            agent_id=agent_id,
            action_type=action_type,
            description=description,
            priority=priority,
            executor=executor,
            parameters=parameters or {},
            execute_at=execute_at,
            timeout_seconds=timeout,
        )

        heapq.heappush(self._queue, action)
        self._stats["total_scheduled"] += 1

        return action

    def cancel(self, action_id: UUID) -> bool:
        """
        Cancel a scheduled action.

        Args:
            action_id: Action to cancel

        Returns:
            True if cancelled
        """
        # Check queue
        for i, action in enumerate(self._queue):
            if action.action_id == action_id:
                action.status = ActionStatus.CANCELLED
                self._queue.pop(i)
                heapq.heapify(self._queue)
                self._stats["total_cancelled"] += 1
                return True

        return False

    async def execute_next(self) -> Optional[ScheduledAction]:
        """
        Execute the next pending action.

        Returns:
            Executed action or None if queue empty
        """
        if not self._queue:
            return None

        # Get next action (respecting time constraints)
        action = self._get_ready_action()
        if not action:
            return None

        # Check agent concurrency
        async with self._running_lock:
            agent_running = self._agent_running.get(action.agent_id, 0)
            if agent_running >= self.max_per_agent:
                # Re-queue with slight delay
                heapq.heappush(self._queue, action)
                return None

            # Mark as running
            self._running[action.action_id] = action
            self._agent_running[action.agent_id] = agent_running + 1
            action.status = ActionStatus.RUNNING
            action.started_at = datetime.utcnow()

        try:
            async with self._semaphore:
                # Execute with timeout
                if action.executor:
                    try:
                        result = await asyncio.wait_for(
                            action.executor(**action.parameters),
                            timeout=action.timeout_seconds
                        )
                        action.result = result
                        action.status = ActionStatus.COMPLETED
                        self._stats["total_completed"] += 1
                    except asyncio.TimeoutError:
                        action.error = "Timeout"
                        action.status = ActionStatus.FAILED
                        self._stats["total_failed"] += 1
                    except Exception as e:
                        action.error = str(e)
                        action.status = ActionStatus.FAILED
                        self._stats["total_failed"] += 1
                else:
                    # No executor, just mark complete
                    action.status = ActionStatus.COMPLETED
                    self._stats["total_completed"] += 1

        finally:
            action.completed_at = datetime.utcnow()

            # Clean up tracking
            async with self._running_lock:
                self._running.pop(action.action_id, None)
                self._agent_running[action.agent_id] = max(
                    0, self._agent_running.get(action.agent_id, 1) - 1
                )

            # Add to history
            self._completed.append(action)
            if len(self._completed) > self._max_history:
                self._completed.pop(0)

        return action

    async def execute_batch(
        self,
        max_actions: int = 10,
    ) -> List[ScheduledAction]:
        """
        Execute a batch of actions in parallel.

        Args:
            max_actions: Maximum actions to execute

        Returns:
            List of executed actions
        """
        tasks = []

        for _ in range(max_actions):
            if not self._queue:
                break

            action = self._get_ready_action()
            if action:
                tasks.append(self._execute_action(action))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        actions = [r for r in results if isinstance(r, ScheduledAction)]
        return actions

    async def _execute_action(self, action: ScheduledAction) -> ScheduledAction:
        """Execute a single action"""
        # Similar to execute_next but returns the action
        async with self._running_lock:
            agent_running = self._agent_running.get(action.agent_id, 0)
            if agent_running >= self.max_per_agent:
                heapq.heappush(self._queue, action)
                return action

            self._running[action.action_id] = action
            self._agent_running[action.agent_id] = agent_running + 1
            action.status = ActionStatus.RUNNING
            action.started_at = datetime.utcnow()

        try:
            async with self._semaphore:
                if action.executor:
                    try:
                        result = await asyncio.wait_for(
                            action.executor(**action.parameters),
                            timeout=action.timeout_seconds
                        )
                        action.result = result
                        action.status = ActionStatus.COMPLETED
                        self._stats["total_completed"] += 1
                    except Exception as e:
                        action.error = str(e)
                        action.status = ActionStatus.FAILED
                        self._stats["total_failed"] += 1
                else:
                    action.status = ActionStatus.COMPLETED
                    self._stats["total_completed"] += 1
        finally:
            action.completed_at = datetime.utcnow()

            async with self._running_lock:
                self._running.pop(action.action_id, None)
                self._agent_running[action.agent_id] = max(
                    0, self._agent_running.get(action.agent_id, 1) - 1
                )

            self._completed.append(action)
            if len(self._completed) > self._max_history:
                self._completed.pop(0)

        return action

    def _get_ready_action(self) -> Optional[ScheduledAction]:
        """Get next action that's ready to execute"""
        now = datetime.utcnow()

        for i, action in enumerate(self._queue):
            if action.execute_at is None or action.execute_at <= now:
                self._queue.pop(i)
                heapq.heapify(self._queue)
                return action

        return None

    def get_pending_count(self) -> int:
        """Get number of pending actions"""
        return len(self._queue)

    def get_running_count(self) -> int:
        """Get number of running actions"""
        return len(self._running)

    def get_agent_pending(self, agent_id: UUID) -> List[ScheduledAction]:
        """Get pending actions for an agent"""
        return [a for a in self._queue if a.agent_id == agent_id]

    def get_agent_running(self, agent_id: UUID) -> List[ScheduledAction]:
        """Get running actions for an agent"""
        return [a for a in self._running.values() if a.agent_id == agent_id]

    def is_agent_busy(self, agent_id: UUID) -> bool:
        """Check if agent has actions running"""
        return self._agent_running.get(agent_id, 0) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            **self._stats,
            "pending": len(self._queue),
            "running": len(self._running),
        }

    def clear_queue(self) -> int:
        """Clear all pending actions"""
        count = len(self._queue)
        self._queue.clear()
        return count

    async def wait_for_agent(self, agent_id: UUID, timeout: float = 30.0) -> bool:
        """
        Wait for agent to finish all actions.

        Args:
            agent_id: Agent to wait for
            timeout: Maximum wait time

        Returns:
            True if agent finished, False if timeout
        """
        start = datetime.utcnow()

        while (datetime.utcnow() - start).total_seconds() < timeout:
            if not self.is_agent_busy(agent_id):
                return True
            await asyncio.sleep(0.1)

        return False
