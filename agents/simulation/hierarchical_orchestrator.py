"""
Hierarchical Simulation Orchestrator
Extends the base SimulationOrchestrator with hierarchical agent scheduling
and collective agent management.
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
import asyncio
import logging

from agents.simulation.orchestrator import (
    SimulationOrchestrator,
    AgentContext,
    TickResult,
    SimulationState,
)
from agents.simulation.time_manager import TimeManager, SimulationTime
from agents.simulation.scheduler import AgentScheduler
from agents.simulation.events import EventSystem
from agents.simulation.hierarchical_scheduler import (
    HierarchicalScheduler,
    AgentTier,
    PlayerContext,
    TierConfig,
)
from agents.collective.collective_manager import CollectiveManager
from agents.collective.settlement_agent import SettlementAgent
from agents.collective.kingdom_agent import KingdomAgent

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalTickResult(TickResult):
    """Extended tick result with hierarchical info."""
    agents_by_tier: Dict[str, int] = field(default_factory=dict)
    collective_events: List[Dict[str, Any]] = field(default_factory=list)
    tier_execution_counts: Dict[str, int] = field(default_factory=dict)


class HierarchicalOrchestrator(SimulationOrchestrator):
    """
    Simulation orchestrator with hierarchical agent scheduling.

    Extends base orchestrator with:
    - Tiered agent execution (party, nearby, settlement, region, background)
    - Collective agents (settlements, kingdoms)
    - Player-centric tier classification
    - Simplified execution for distant agents
    """

    def __init__(
        self,
        time_manager: Optional[TimeManager] = None,
        scheduler: Optional[AgentScheduler] = None,
        event_system: Optional[EventSystem] = None,
        hierarchical_scheduler: Optional[HierarchicalScheduler] = None,
        collective_manager: Optional[CollectiveManager] = None,
        tick_rate: float = 1.0,
        max_agents: int = 1000,  # Higher limit with hierarchical scheduling
    ):
        super().__init__(
            time_manager=time_manager,
            scheduler=scheduler,
            event_system=event_system,
            tick_rate=tick_rate,
            max_agents=max_agents,
        )

        self.hierarchical_scheduler = hierarchical_scheduler or HierarchicalScheduler()
        self.collective_manager = collective_manager or CollectiveManager()

        # Player tracking
        self._player_id: Optional[UUID] = None
        self._player_x: int = 0
        self._player_y: int = 0
        self._party_ids: Set[UUID] = set()
        self._active_interaction_ids: Set[UUID] = set()

        # Agent positions cache
        self._agent_positions: Dict[UUID, tuple] = {}

        # Extended statistics
        self._hierarchical_stats = {
            'tier_executions': {tier.name: 0 for tier in AgentTier},
            'collective_ticks': 0,
            'simplified_cycles': 0,
            'full_cycles': 0,
        }

    # ==================== Player Management ====================

    def set_player(
        self,
        player_id: UUID,
        x: int,
        y: int,
        settlement_id: Optional[UUID] = None,
        region_id: Optional[str] = None
    ):
        """Set the player position for tier classification."""
        self._player_id = player_id
        self._player_x = x
        self._player_y = y

        # Update hierarchical scheduler
        context = PlayerContext(
            player_id=player_id,
            x=x,
            y=y,
            settlement_id=settlement_id,
            region_id=region_id,
            party_agent_ids=self._party_ids.copy(),
            active_interaction_agent_ids=self._active_interaction_ids.copy(),
        )
        self.hierarchical_scheduler.update_player_context(context)

    def update_player_position(self, x: int, y: int, settlement_id: Optional[UUID] = None):
        """Update player position."""
        self._player_x = x
        self._player_y = y

        if self._player_id:
            context = PlayerContext(
                player_id=self._player_id,
                x=x,
                y=y,
                settlement_id=settlement_id,
                party_agent_ids=self._party_ids.copy(),
                active_interaction_agent_ids=self._active_interaction_ids.copy(),
            )
            self.hierarchical_scheduler.update_player_context(context)

    def add_to_party(self, agent_id: UUID):
        """Add an agent to the player's party."""
        self._party_ids.add(agent_id)
        self.hierarchical_scheduler.set_agent_in_party(agent_id, True)

    def remove_from_party(self, agent_id: UUID):
        """Remove an agent from the player's party."""
        self._party_ids.discard(agent_id)
        self.hierarchical_scheduler.set_agent_in_party(agent_id, False)

    def start_interaction(self, agent_id: UUID):
        """Mark an agent as in active interaction with player."""
        self._active_interaction_ids.add(agent_id)
        self.hierarchical_scheduler.set_agent_active_interaction(agent_id, True)

    def end_interaction(self, agent_id: UUID):
        """End active interaction with an agent."""
        self._active_interaction_ids.discard(agent_id)
        self.hierarchical_scheduler.set_agent_active_interaction(agent_id, False)

    # ==================== Agent Management (Extended) ====================

    def register_agent(
        self,
        agent_id: UUID,
        name: str,
        location: Optional[str] = None,
        memory_stream: Any = None,
        graph_runner: Any = None,
        x: int = 0,
        y: int = 0,
        settlement_id: Optional[UUID] = None,
        region_id: Optional[str] = None,
    ) -> AgentContext:
        """Register an agent with hierarchical scheduling."""
        # Register with base orchestrator
        context = super().register_agent(
            agent_id=agent_id,
            name=name,
            location=location,
            memory_stream=memory_stream,
            graph_runner=graph_runner,
        )

        # Register with hierarchical scheduler
        self.hierarchical_scheduler.register_agent(
            agent_id=agent_id,
            x=x,
            y=y,
            settlement_id=settlement_id,
            region_id=region_id,
        )

        # Track position
        self._agent_positions[agent_id] = (x, y)

        # Register with collective manager if in a settlement
        if settlement_id:
            self.collective_manager.register_agent_in_settlement(agent_id, settlement_id)

        return context

    def unregister_agent(self, agent_id: UUID) -> bool:
        """Remove an agent from the simulation."""
        # Unregister from hierarchical scheduler
        self.hierarchical_scheduler.unregister_agent(agent_id)

        # Unregister from collective manager
        self.collective_manager.unregister_agent(agent_id)

        # Remove from position tracking
        self._agent_positions.pop(agent_id, None)

        # Remove from party/interaction
        self._party_ids.discard(agent_id)
        self._active_interaction_ids.discard(agent_id)

        # Unregister from base orchestrator
        return super().unregister_agent(agent_id)

    def update_agent_position(
        self,
        agent_id: UUID,
        x: int,
        y: int,
        settlement_id: Optional[UUID] = None,
        region_id: Optional[str] = None
    ):
        """Update an agent's position."""
        self._agent_positions[agent_id] = (x, y)
        self.hierarchical_scheduler.update_agent_position(
            agent_id=agent_id,
            x=x,
            y=y,
            settlement_id=settlement_id,
            region_id=region_id,
        )

        # Check if settlement changed
        old_settlement = self.collective_manager.agent_to_settlement.get(agent_id)
        if settlement_id != old_settlement:
            self.collective_manager.move_agent_to_settlement(agent_id, settlement_id)

    # ==================== Settlement/Kingdom Management ====================

    def register_settlement(
        self,
        settlement_id: UUID,
        name: str,
        settlement_type: str,
        x: int,
        y: int,
        population: int = 100,
        faction_id: Optional[UUID] = None
    ) -> SettlementAgent:
        """Register a settlement as a collective agent."""
        return self.collective_manager.register_settlement(
            settlement_id=settlement_id,
            name=name,
            settlement_type=settlement_type,
            x=x,
            y=y,
            population=population,
            faction_id=faction_id,
        )

    def register_kingdom(
        self,
        faction_id: UUID,
        name: str,
        faction_type: str,
        capital_settlement_id: Optional[UUID] = None
    ) -> KingdomAgent:
        """Register a kingdom as a collective agent."""
        return self.collective_manager.register_kingdom(
            faction_id=faction_id,
            name=name,
            faction_type=faction_type,
            capital_settlement_id=capital_settlement_id,
        )

    # ==================== Tick Execution (Extended) ====================

    async def _execute_tick(self) -> HierarchicalTickResult:
        """Execute a tick with hierarchical scheduling."""
        start_time = datetime.utcnow()
        self._tick_count += 1

        # 1. Advance simulation time
        time_result = self.time_manager.tick()
        time_events = time_result.get("events", [])
        game_hour = self.time_manager.current_time.hour

        # 2. Execute collective agents
        collective_result = await self.collective_manager.tick(
            current_tick=self._tick_count,
            game_hour=game_hour,
        )
        self._hierarchical_stats['collective_ticks'] += 1

        # 3. Generate world events
        world_events = []
        if self._tick_count % 10 == 0:
            import random
            if random.random() < 0.1:
                locations = list(self._agent_locations.keys())
                if locations:
                    event = self.event_system.generate_random_event(
                        location=random.choice(locations)
                    )
                    if event:
                        world_events.append(event)

        # 4. Get agents to execute this tick (hierarchical)
        agents_by_tier = self.hierarchical_scheduler.get_agents_to_execute(
            current_tick=self._tick_count,
            agent_positions=self._agent_positions,
        )

        # 5. Execute agents by tier
        agent_actions: Dict[UUID, List[Dict[str, Any]]] = {}
        tier_execution_counts = {tier.name: 0 for tier in AgentTier}

        for tier in AgentTier:
            agent_ids = agents_by_tier.get(tier, [])
            if not agent_ids:
                continue

            tier_config = self.hierarchical_scheduler.tier_configs[tier]
            tier_execution_counts[tier.name] = len(agent_ids)
            self._hierarchical_stats['tier_executions'][tier.name] += len(agent_ids)

            # Execute in batches
            for i in range(0, len(agent_ids), tier_config.batch_size):
                batch = agent_ids[i:i + tier_config.batch_size]
                batch_results = await self._execute_agent_batch(
                    agent_ids=batch,
                    use_simplified=tier_config.use_simplified_cycle,
                )
                agent_actions.update(batch_results)

                # Mark as executed
                for agent_id in batch:
                    self.hierarchical_scheduler.mark_executed(agent_id, self._tick_count)

        # 6. Execute scheduled actions
        executed = await self.scheduler.execute_batch(max_actions=10)
        for action in executed:
            if action.agent_id not in agent_actions:
                agent_actions[action.agent_id] = []
            agent_actions[action.agent_id].append(action.to_dict())

        # 7. Update statistics
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._stats["total_ticks"] += 1
        self._stats["total_events"] += len(world_events)
        self._stats["last_tick_duration_ms"] = duration_ms

        # 8. Build result
        tier_counts = self.hierarchical_scheduler.get_tier_stats()
        result = HierarchicalTickResult(
            tick_number=self._tick_count,
            simulation_time=time_result["current"],
            time_events=time_events,
            world_events=world_events,
            agent_actions=agent_actions,
            duration_ms=duration_ms,
            agents_by_tier={tier.name: count for tier, count in tier_counts.items()},
            collective_events=collective_result.get('events_generated', []),
            tier_execution_counts=tier_execution_counts,
        )

        # 9. Notify callbacks
        for callback in self._tick_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

        return result

    async def _execute_agent_batch(
        self,
        agent_ids: List[UUID],
        use_simplified: bool = False
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """Execute a batch of agents."""
        results: Dict[UUID, List[Dict[str, Any]]] = {}

        async def run_one(agent_id: UUID):
            ctx = self._agents.get(agent_id)
            if not ctx or not ctx.is_active or not ctx.graph_runner:
                return agent_id, []

            try:
                # Get collective modifiers for this agent
                modifiers = self.collective_manager.get_agent_modifiers(agent_id)

                if use_simplified:
                    # Simplified execution for distant agents
                    self._hierarchical_stats['simplified_cycles'] += 1
                    actions = await self._run_simplified_cycle(ctx, modifiers)
                else:
                    # Full execution for nearby agents
                    self._hierarchical_stats['full_cycles'] += 1
                    actions = await self._run_full_cycle(ctx, modifiers)

                return agent_id, actions

            except Exception as e:
                logger.error(f"Agent {ctx.name} cycle error: {e}")
                return agent_id, [{"error": str(e)}]

        # Run in parallel
        tasks = [run_one(aid) for aid in agent_ids]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in task_results:
            if isinstance(item, Exception):
                logger.error(f"Batch execution error: {item}")
            else:
                agent_id, actions = item
                results[agent_id] = actions

        return results

    async def _run_full_cycle(
        self,
        ctx: AgentContext,
        modifiers: Any
    ) -> List[Dict[str, Any]]:
        """Run full agent cycle with LLM reasoning."""
        actions = []

        if hasattr(ctx.graph_runner, 'run_cycle'):
            result = await ctx.graph_runner.run_cycle(
                current_time=self.time_manager.current_time,
                settlement_context=modifiers.settlement_context if modifiers else None,
                kingdom_context=modifiers.kingdom_context if modifiers else None,
            )

            if result and hasattr(result, 'actions_taken'):
                actions.extend(result.actions_taken)
                ctx.total_actions += len(result.actions_taken)
                self._stats["total_agent_actions"] += len(result.actions_taken)

            ctx.actions_this_tick = len(actions)
            ctx.last_action_time = datetime.utcnow()

            if actions:
                ctx.current_activity = actions[-1].get("type", "unknown")

        return actions

    async def _run_simplified_cycle(
        self,
        ctx: AgentContext,
        modifiers: Any
    ) -> List[Dict[str, Any]]:
        """
        Run simplified agent cycle without LLM reasoning.
        Uses rule-based behavior based on time of day and modifiers.
        """
        actions = []

        # Get appropriate activities for time of day
        appropriate_activities = self.time_manager.get_appropriate_activities()

        # Modify based on collective modifiers
        if modifiers:
            # Add encouraged activities
            appropriate_activities.extend(modifiers.encouraged_activities)
            # Remove discouraged activities
            appropriate_activities = [
                a for a in appropriate_activities
                if a not in modifiers.discouraged_activities
            ]

        # Pick activity based on simple rules
        if appropriate_activities:
            import random
            activity = random.choice(appropriate_activities)
            actions.append({
                "type": activity,
                "description": f"{ctx.name} is {activity}",
                "simplified": True,
            })

            ctx.current_activity = activity
            ctx.actions_this_tick = 1
            ctx.last_action_time = datetime.utcnow()

        return actions

    # ==================== Statistics (Extended) ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get extended statistics."""
        base_stats = super().get_stats()

        return {
            **base_stats,
            'hierarchical': {
                **self._hierarchical_stats,
                'scheduler_stats': self.hierarchical_scheduler.get_stats(),
                'collective_stats': self.collective_manager.get_stats(),
            },
            'party_size': len(self._party_ids),
            'active_interactions': len(self._active_interaction_ids),
        }

    def get_agent_tier(self, agent_id: UUID) -> Optional[str]:
        """Get an agent's current tier."""
        tier = self.hierarchical_scheduler.get_agent_tier(agent_id)
        return tier.name if tier else None

    def get_tier_distribution(self) -> Dict[str, int]:
        """Get count of agents in each tier."""
        counts = self.hierarchical_scheduler.get_tier_stats()
        return {tier.name: count for tier, count in counts.items()}

    # ==================== Persistence (Extended) ====================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize simulation state."""
        base_dict = super().to_dict()

        return {
            **base_dict,
            'player_id': str(self._player_id) if self._player_id else None,
            'player_position': {'x': self._player_x, 'y': self._player_y},
            'party_ids': [str(pid) for pid in self._party_ids],
            'hierarchical_stats': self._hierarchical_stats,
            'collectives': self.collective_manager.to_dict(),
        }
