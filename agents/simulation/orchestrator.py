"""
Simulation Orchestrator
Main coordinator for the agent simulation.
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
from enum import Enum
import asyncio
import logging

from agents.simulation.time_manager import TimeManager, SimulationTime, TimeOfDay, Season
from agents.simulation.scheduler import AgentScheduler, ActionPriority, ScheduledAction
from agents.simulation.events import EventSystem, EventType, EventScope, WorldEvent


logger = logging.getLogger(__name__)


class SimulationState(str, Enum):
    """State of the simulation"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"  # Single-step mode


@dataclass
class AgentContext:
    """Runtime context for an agent in the simulation"""
    agent_id: UUID
    name: str
    location: Optional[str] = None
    is_active: bool = True
    last_action_time: Optional[datetime] = None
    current_activity: str = "idle"

    # References to agent systems (set during registration)
    memory_stream: Any = None
    graph_runner: Any = None

    # Tracking
    actions_this_tick: int = 0
    total_actions: int = 0


@dataclass
class TickResult:
    """Result of a simulation tick"""
    tick_number: int
    simulation_time: Dict[str, Any]
    time_events: List[str]
    world_events: List[WorldEvent]
    agent_actions: Dict[UUID, List[Dict[str, Any]]]
    duration_ms: float


class SimulationOrchestrator:
    """
    Main orchestrator for the agent simulation.

    Coordinates:
    - Time progression
    - Agent scheduling and execution
    - World events
    - Inter-agent interactions

    Features:
    - Configurable tick rate
    - Pause/resume/step controls
    - Agent lifecycle management
    - Event broadcasting
    - Statistics tracking
    """

    def __init__(
        self,
        time_manager: Optional[TimeManager] = None,
        scheduler: Optional[AgentScheduler] = None,
        event_system: Optional[EventSystem] = None,
        tick_rate: float = 1.0,  # Ticks per second (real time)
        max_agents: int = 100,
    ):
        """
        Initialize the orchestrator.

        Args:
            time_manager: Time management system
            scheduler: Action scheduler
            event_system: Event system
            tick_rate: How many ticks per real second
            max_agents: Maximum agents to support
        """
        self.time_manager = time_manager or TimeManager()
        self.scheduler = scheduler or AgentScheduler()
        self.event_system = event_system or EventSystem()

        self.tick_rate = tick_rate
        self.max_agents = max_agents

        # State
        self.state = SimulationState.STOPPED
        self._tick_count = 0

        # Agent management
        self._agents: Dict[UUID, AgentContext] = {}
        self._agent_locations: Dict[str, Set[UUID]] = {}

        # Graph runners (for agent execution)
        self._graph_runners: Dict[UUID, Any] = {}

        # Execution control
        self._run_task: Optional[asyncio.Task] = None
        self._step_event = asyncio.Event()

        # Callbacks
        self._tick_callbacks: List[callable] = []
        self._agent_action_callbacks: List[callable] = []

        # Statistics
        self._stats = {
            "total_ticks": 0,
            "total_agent_actions": 0,
            "total_events": 0,
            "start_time": None,
            "last_tick_duration_ms": 0,
        }

        # Register time-based event generation
        self._setup_time_callbacks()

    def _setup_time_callbacks(self) -> None:
        """Setup callbacks for time-based events"""

        def on_dawn(time: SimulationTime):
            self.event_system.create_event(
                event_type=EventType.SEASONAL,
                title="Dawn Breaks",
                description="The sun rises, bringing a new day.",
                scope=EventScope.WORLD,
                importance=0.3,
            )

        def on_dusk(time: SimulationTime):
            self.event_system.create_event(
                event_type=EventType.SEASONAL,
                title="Night Falls",
                description="Darkness descends as the sun sets.",
                scope=EventScope.WORLD,
                importance=0.3,
            )

        def on_season_change(time: SimulationTime, season: Season):
            descriptions = {
                Season.SPRING: "Flowers bloom and life returns to the land.",
                Season.SUMMER: "The days grow long and warm.",
                Season.AUTUMN: "Leaves turn golden as harvest time approaches.",
                Season.WINTER: "Cold winds blow and snow blankets the land.",
            }
            self.event_system.create_event(
                event_type=EventType.SEASONAL,
                title=f"{season.value.capitalize()} Arrives",
                description=descriptions.get(season, "The season changes."),
                scope=EventScope.WORLD,
                importance=0.6,
            )

        self.time_manager.on_dawn(on_dawn)
        self.time_manager.on_dusk(on_dusk)
        self.time_manager.on_season(on_season_change)

    # ==================== Agent Management ====================

    def register_agent(
        self,
        agent_id: UUID,
        name: str,
        location: Optional[str] = None,
        memory_stream: Any = None,
        graph_runner: Any = None,
    ) -> AgentContext:
        """
        Register an agent with the simulation.

        Args:
            agent_id: Agent's unique ID
            name: Agent's name
            location: Starting location
            memory_stream: Agent's memory stream
            graph_runner: Agent's graph runner for execution

        Returns:
            Agent context
        """
        if len(self._agents) >= self.max_agents:
            raise ValueError(f"Maximum agents ({self.max_agents}) reached")

        context = AgentContext(
            agent_id=agent_id,
            name=name,
            location=location,
            memory_stream=memory_stream,
            graph_runner=graph_runner,
        )

        self._agents[agent_id] = context

        if graph_runner:
            self._graph_runners[agent_id] = graph_runner

        # Track location
        if location:
            if location not in self._agent_locations:
                self._agent_locations[location] = set()
            self._agent_locations[location].add(agent_id)

        # Subscribe to events
        self.event_system.subscribe(
            agent_id=agent_id,
            callback=lambda event: self._on_agent_event(agent_id, event),
            scope=EventScope.LOCAL,
            location=location,
        )

        logger.info(f"Registered agent {name} ({agent_id})")
        return context

    def unregister_agent(self, agent_id: UUID) -> bool:
        """Remove an agent from the simulation"""
        context = self._agents.pop(agent_id, None)
        if not context:
            return False

        # Clean up location tracking
        if context.location and context.location in self._agent_locations:
            self._agent_locations[context.location].discard(agent_id)

        # Clean up subscriptions
        self.event_system.unsubscribe(agent_id)

        # Clean up graph runner
        self._graph_runners.pop(agent_id, None)

        logger.info(f"Unregistered agent {context.name} ({agent_id})")
        return True

    def get_agent(self, agent_id: UUID) -> Optional[AgentContext]:
        """Get agent context"""
        return self._agents.get(agent_id)

    def get_agents_at_location(self, location: str) -> List[AgentContext]:
        """Get all agents at a location"""
        agent_ids = self._agent_locations.get(location, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def move_agent(self, agent_id: UUID, new_location: str) -> bool:
        """Move an agent to a new location"""
        context = self._agents.get(agent_id)
        if not context:
            return False

        # Remove from old location
        if context.location and context.location in self._agent_locations:
            self._agent_locations[context.location].discard(agent_id)

        # Add to new location
        if new_location not in self._agent_locations:
            self._agent_locations[new_location] = set()
        self._agent_locations[new_location].add(agent_id)

        context.location = new_location
        return True

    def _on_agent_event(self, agent_id: UUID, event: WorldEvent) -> None:
        """Handle event notification for an agent"""
        context = self._agents.get(agent_id)
        if not context or not context.is_active:
            return

        # Could trigger reactions here if needed
        logger.debug(f"Agent {context.name} witnessed event: {event.title}")

    # ==================== Simulation Control ====================

    async def start(self) -> None:
        """Start the simulation loop"""
        if self.state == SimulationState.RUNNING:
            return

        self.state = SimulationState.RUNNING
        self._stats["start_time"] = datetime.utcnow()

        logger.info("Simulation started")
        self._run_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the simulation"""
        self.state = SimulationState.STOPPED

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None

        logger.info("Simulation stopped")

    def pause(self) -> None:
        """Pause the simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.time_manager.pause()
            logger.info("Simulation paused")

    def resume(self) -> None:
        """Resume the simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.time_manager.resume()
            logger.info("Simulation resumed")

    async def step(self) -> TickResult:
        """Execute a single simulation tick"""
        previous_state = self.state
        self.state = SimulationState.STEPPING

        result = await self._execute_tick()

        self.state = previous_state if previous_state != SimulationState.STEPPING else SimulationState.PAUSED
        return result

    async def _run_loop(self) -> None:
        """Main simulation loop"""
        tick_interval = 1.0 / self.tick_rate if self.tick_rate > 0 else 1.0

        while self.state in [SimulationState.RUNNING, SimulationState.PAUSED]:
            if self.state == SimulationState.PAUSED:
                await asyncio.sleep(0.1)
                continue

            start_time = datetime.utcnow()

            try:
                await self._execute_tick()
            except Exception as e:
                logger.error(f"Tick error: {e}")

            # Maintain tick rate
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_time = max(0, tick_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _execute_tick(self) -> TickResult:
        """Execute a single simulation tick"""
        start_time = datetime.utcnow()
        self._tick_count += 1

        # 1. Advance simulation time
        time_result = self.time_manager.tick()
        time_events = time_result.get("events", [])

        # 2. Generate random events (with some probability)
        world_events = []
        if self._tick_count % 10 == 0:  # Every 10 ticks, chance for random event
            import random
            if random.random() < 0.1:  # 10% chance
                locations = list(self._agent_locations.keys())
                if locations:
                    event = self.event_system.generate_random_event(
                        location=random.choice(locations)
                    )
                    if event:
                        world_events.append(event)

        # 3. Check event generation rules
        context = {
            "time": self.time_manager.current_time.to_dict(),
            "agents": len(self._agents),
            "tick": self._tick_count,
        }
        rule_events = self.event_system.check_generation_rules(context)
        world_events.extend(rule_events)

        # 4. Execute agent actions
        agent_actions: Dict[UUID, List[Dict[str, Any]]] = {}

        # Run graph runners for active agents
        active_agents = [
            ctx for ctx in self._agents.values()
            if ctx.is_active and ctx.graph_runner
        ]

        if active_agents:
            # Execute agent cycles in parallel (limited by scheduler)
            tasks = []
            for ctx in active_agents:
                if ctx.graph_runner:
                    tasks.append(self._run_agent_cycle(ctx))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for ctx, result in zip(active_agents, results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {ctx.name} cycle error: {result}")
                        agent_actions[ctx.agent_id] = [{"error": str(result)}]
                    elif result:
                        agent_actions[ctx.agent_id] = result

        # 5. Execute scheduled actions
        executed = await self.scheduler.execute_batch(max_actions=10)
        for action in executed:
            if action.agent_id not in agent_actions:
                agent_actions[action.agent_id] = []
            agent_actions[action.agent_id].append(action.to_dict())

        # 6. Update statistics
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._stats["total_ticks"] += 1
        self._stats["total_events"] += len(world_events)
        self._stats["last_tick_duration_ms"] = duration_ms

        # 7. Notify tick callbacks
        result = TickResult(
            tick_number=self._tick_count,
            simulation_time=time_result["current"],
            time_events=time_events,
            world_events=world_events,
            agent_actions=agent_actions,
            duration_ms=duration_ms,
        )

        for callback in self._tick_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

        return result

    async def _run_agent_cycle(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        """Run a single agent's execution cycle"""
        actions = []

        try:
            # Get appropriate activities based on time
            appropriate_activities = self.time_manager.get_appropriate_activities()
            env_modifiers = self.time_manager.get_environment_modifiers()

            # Run the agent's graph
            if hasattr(ctx.graph_runner, 'run_cycle'):
                result = await ctx.graph_runner.run_cycle(
                    current_time=self.time_manager.current_time
                )

                if result and hasattr(result, 'actions_taken'):
                    actions.extend(result.actions_taken)
                    ctx.total_actions += len(result.actions_taken)
                    self._stats["total_agent_actions"] += len(result.actions_taken)

                ctx.actions_this_tick = len(actions)
                ctx.last_action_time = datetime.utcnow()

                if actions:
                    ctx.current_activity = actions[-1].get("type", "unknown")

        except Exception as e:
            logger.error(f"Error running agent {ctx.name}: {e}")
            actions.append({"error": str(e)})

        return actions

    # ==================== Event Management ====================

    def create_event(
        self,
        event_type: EventType,
        title: str,
        description: str,
        scope: EventScope = EventScope.LOCAL,
        location: Optional[str] = None,
        importance: float = 0.5,
        involved_agents: Optional[List[UUID]] = None,
    ) -> WorldEvent:
        """Create a world event"""
        return self.event_system.create_event(
            event_type=event_type,
            title=title,
            description=description,
            scope=scope,
            location=location,
            importance=importance,
            involved_agents=involved_agents,
        )

    # ==================== Scheduling ====================

    def schedule_action(
        self,
        agent_id: UUID,
        action_type: str,
        description: str,
        priority: ActionPriority = ActionPriority.NORMAL,
        executor: Optional[callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ScheduledAction:
        """Schedule an action for an agent"""
        return self.scheduler.schedule(
            agent_id=agent_id,
            action_type=action_type,
            description=description,
            priority=priority,
            executor=executor,
            parameters=parameters,
        )

    # ==================== Callbacks ====================

    def on_tick(self, callback: callable) -> None:
        """Register a callback for each tick"""
        self._tick_callbacks.append(callback)

    def on_agent_action(self, callback: callable) -> None:
        """Register a callback for agent actions"""
        self._agent_action_callbacks.append(callback)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            **self._stats,
            "state": self.state.value,
            "tick_count": self._tick_count,
            "agent_count": len(self._agents),
            "active_agents": sum(1 for a in self._agents.values() if a.is_active),
            "scheduler_stats": self.scheduler.get_stats(),
            "event_stats": self.event_system.get_stats(),
            "current_time": self.time_manager.current_time.to_dict(),
        }

    def get_time(self) -> SimulationTime:
        """Get current simulation time"""
        return self.time_manager.current_time

    def get_time_of_day(self) -> TimeOfDay:
        """Get current time of day"""
        return self.time_manager.current_time.get_time_of_day()

    def get_season(self) -> Season:
        """Get current season"""
        return self.time_manager.current_time.get_season()

    # ==================== Persistence ====================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize simulation state"""
        return {
            "state": self.state.value,
            "tick_count": self._tick_count,
            "time": self.time_manager.current_time.to_dict(),
            "agents": {
                str(aid): {
                    "name": ctx.name,
                    "location": ctx.location,
                    "is_active": ctx.is_active,
                    "current_activity": ctx.current_activity,
                    "total_actions": ctx.total_actions,
                }
                for aid, ctx in self._agents.items()
            },
            "stats": self._stats,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        time_manager: Optional[TimeManager] = None,
        scheduler: Optional[AgentScheduler] = None,
        event_system: Optional[EventSystem] = None,
    ) -> "SimulationOrchestrator":
        """Create from serialized state"""
        # Create time manager from saved time
        if not time_manager:
            time_data = data.get("time", {})
            sim_time = SimulationTime.from_dict(time_data)
            time_manager = TimeManager(start_time=sim_time)

        orchestrator = cls(
            time_manager=time_manager,
            scheduler=scheduler,
            event_system=event_system,
        )

        orchestrator._tick_count = data.get("tick_count", 0)
        orchestrator._stats = data.get("stats", orchestrator._stats)

        return orchestrator
