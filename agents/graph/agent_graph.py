"""
Agent Graph
Main agent execution graph implementing perceive → retrieve → plan → act cycle.
"""

from typing import Optional, Dict, Any, Callable, Awaitable, List
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from agents.graph.state import AgentGraphState, AgentPhase
from agents.graph.nodes import (
    perceive_node,
    retrieve_node,
    reflect_node,
    plan_node,
    react_node,
    act_node,
    get_next_node,
)
from agents.memory.memory_stream import MemoryStream
from agents.reasoning.reflection import ReflectionSystem
from agents.reasoning.planning import PlanningSystem
from agents.reasoning.reaction import ReactionSystem
from agents.world.interface import WorldInterface
from agents.llm.ollama_client import OllamaClient


class GraphStatus(str, Enum):
    """Execution status of the graph"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class ExecutionResult:
    """Result of a graph execution"""
    status: GraphStatus
    state: AgentGraphState
    action_taken: Optional[str] = None
    duration_ms: float = 0.0
    nodes_executed: List[str] = None
    error: Optional[str] = None


class AgentGraph:
    """
    Agent execution graph.

    Implements the Stanford Generative Agents architecture:
    Perceive → Retrieve → (Reflect) → Plan → Act

    The graph is stateless - all state is carried in AgentGraphState.
    This allows for easy checkpointing, debugging, and replay.
    """

    def __init__(
        self,
        agent_id: UUID,
        agent_name: str,
        memory_stream: MemoryStream,
        world: WorldInterface,
        ollama_client: Optional[OllamaClient] = None,
        agent_summary: str = "",
    ):
        """
        Initialize agent graph.

        Args:
            agent_id: Agent's UUID
            agent_name: Agent's name
            memory_stream: Agent's memory stream
            world: World interface for perception and action
            ollama_client: LLM client for generation
            agent_summary: Description of the agent
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.memory_stream = memory_stream
        self.world = world
        self.client = ollama_client
        self.agent_summary = agent_summary

        # Initialize reasoning systems
        self.reflection_system = ReflectionSystem(
            agent_id=agent_id,
            agent_name=agent_name,
            memory_stream=memory_stream,
            ollama_client=ollama_client,
        )

        self.planning_system = PlanningSystem(
            agent_id=agent_id,
            agent_name=agent_name,
            memory_stream=memory_stream,
            ollama_client=ollama_client,
        )

        self.reaction_system = ReactionSystem(
            agent_id=agent_id,
            agent_name=agent_name,
            memory_stream=memory_stream,
            ollama_client=ollama_client,
        )

        # Custom action executor (can be overridden)
        self._action_executor: Optional[Callable] = None

        # Execution state
        self._status = GraphStatus.IDLE
        self._current_state: Optional[AgentGraphState] = None

    @property
    def status(self) -> GraphStatus:
        """Current graph status"""
        return self._status

    def set_action_executor(
        self,
        executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ) -> None:
        """Set custom action executor"""
        self._action_executor = executor

    async def run_cycle(
        self,
        current_time: Optional[datetime] = None,
        max_iterations: int = 10,
    ) -> ExecutionResult:
        """
        Run one complete agent cycle.

        Executes: Perceive → Retrieve → (Reflect) → Plan → Act

        Args:
            current_time: Simulation time
            max_iterations: Safety limit on node transitions

        Returns:
            ExecutionResult with final state
        """
        start_time = datetime.utcnow()
        self._status = GraphStatus.RUNNING

        # Create initial state
        state = AgentGraphState.create(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_summary=self.agent_summary,
            current_time=current_time,
        )

        self._current_state = state

        try:
            # Execute graph nodes
            iterations = 0
            current_node = "perceive"

            while current_node != "end" and iterations < max_iterations:
                state = await self._execute_node(current_node, state)

                if state.error:
                    self._status = GraphStatus.ERROR
                    break

                current_node = get_next_node(state)
                iterations += 1

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            state.execution_duration_ms = duration

            if state.error:
                self._status = GraphStatus.ERROR
            else:
                self._status = GraphStatus.COMPLETED

            return ExecutionResult(
                status=self._status,
                state=state,
                action_taken=state.current_action,
                duration_ms=duration,
                nodes_executed=state.nodes_visited,
                error=state.error,
            )

        except Exception as e:
            self._status = GraphStatus.ERROR
            state.error = str(e)

            return ExecutionResult(
                status=GraphStatus.ERROR,
                state=state,
                error=str(e),
                nodes_executed=state.nodes_visited,
            )

    async def _execute_node(
        self,
        node_name: str,
        state: AgentGraphState
    ) -> AgentGraphState:
        """Execute a single node"""

        if node_name == "perceive":
            return await perceive_node(
                state=state,
                world=self.world,
                memory_stream=self.memory_stream,
            )

        elif node_name == "retrieve":
            return await retrieve_node(
                state=state,
                memory_stream=self.memory_stream,
            )

        elif node_name == "reflect":
            return await reflect_node(
                state=state,
                reflection_system=self.reflection_system,
            )

        elif node_name == "plan":
            return await plan_node(
                state=state,
                planning_system=self.planning_system,
            )

        elif node_name == "react":
            return await react_node(
                state=state,
                reaction_system=self.reaction_system,
                planning_system=self.planning_system,
            )

        elif node_name == "act":
            return await act_node(
                state=state,
                world=self.world,
                memory_stream=self.memory_stream,
                action_executor=self._action_executor,
            )

        elif node_name == "error_handler":
            # Log error and try to recover
            state.retry_count += 1
            if state.retry_count < 3:
                state.error = None  # Clear error to retry
            return state

        else:
            state.error = f"Unknown node: {node_name}"
            return state

    async def inject_event(
        self,
        event_description: str,
        source_agent_id: Optional[UUID] = None,
        source_agent_name: Optional[str] = None,
    ) -> None:
        """
        Inject an event for the agent to react to.

        Args:
            event_description: What happened
            source_agent_id: Who caused it
            source_agent_name: Name of source agent
        """
        if self._current_state:
            self._current_state.interrupt_event = event_description

    def get_current_activity(self) -> str:
        """Get description of current activity"""
        return self.planning_system.get_current_activity()

    def get_plan_context(self) -> Dict[str, Any]:
        """Get current planning context"""
        return self.planning_system.get_plan_context()

    async def force_reflection(self) -> List[Any]:
        """Force immediate reflection"""
        return await self.reflection_system.reflect(force=True)


def create_agent_graph(
    agent_id: UUID,
    agent_name: str,
    memory_stream: MemoryStream,
    world: WorldInterface,
    ollama_client: Optional[OllamaClient] = None,
    agent_summary: str = "",
) -> AgentGraph:
    """
    Factory function to create an agent graph.

    Args:
        agent_id: Agent's UUID
        agent_name: Agent's name
        memory_stream: Memory stream instance
        world: World interface
        ollama_client: LLM client
        agent_summary: Agent description

    Returns:
        Configured AgentGraph
    """
    return AgentGraph(
        agent_id=agent_id,
        agent_name=agent_name,
        memory_stream=memory_stream,
        world=world,
        ollama_client=ollama_client,
        agent_summary=agent_summary,
    )


class AgentGraphRunner:
    """
    Runs multiple agent graphs in a simulation.

    Handles scheduling, time management, and coordination
    between multiple agents.
    """

    def __init__(
        self,
        world: WorldInterface,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize runner.

        Args:
            world: Shared world interface
            ollama_client: Shared LLM client
        """
        self.world = world
        self.client = ollama_client
        self._agents: Dict[UUID, AgentGraph] = {}
        self._simulation_time: datetime = datetime.utcnow()

    def register_agent(self, graph: AgentGraph) -> None:
        """Register an agent graph"""
        self._agents[graph.agent_id] = graph

    def unregister_agent(self, agent_id: UUID) -> None:
        """Unregister an agent"""
        self._agents.pop(agent_id, None)

    async def run_tick(
        self,
        delta_minutes: int = 15,
    ) -> Dict[UUID, ExecutionResult]:
        """
        Run one simulation tick for all agents.

        Args:
            delta_minutes: Time advancement per tick

        Returns:
            Results for each agent
        """
        from datetime import timedelta

        # Advance simulation time
        self._simulation_time += timedelta(minutes=delta_minutes)

        results = {}

        # Run each agent (could be parallelized with asyncio.gather)
        for agent_id, graph in self._agents.items():
            result = await graph.run_cycle(current_time=self._simulation_time)
            results[agent_id] = result

        return results

    async def run_simulation(
        self,
        duration_hours: int = 24,
        tick_minutes: int = 15,
    ) -> None:
        """
        Run simulation for specified duration.

        Args:
            duration_hours: Total simulation hours
            tick_minutes: Minutes per tick
        """
        ticks = (duration_hours * 60) // tick_minutes

        for tick in range(ticks):
            await self.run_tick(delta_minutes=tick_minutes)

            # Could add callbacks, logging, checkpointing here

    def get_agent_status(self, agent_id: UUID) -> Optional[GraphStatus]:
        """Get an agent's current status"""
        graph = self._agents.get(agent_id)
        return graph.status if graph else None

    def get_all_activities(self) -> Dict[UUID, str]:
        """Get current activities for all agents"""
        return {
            agent_id: graph.get_current_activity()
            for agent_id, graph in self._agents.items()
        }
