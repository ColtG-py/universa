"""
Full Stack Integration Test
Tests the complete integration of all Universa components.
"""

import asyncio
import sys
import logging
from uuid import uuid4

# Add project root to path
sys.path.insert(0, '/home/colt/Projects/universa')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class IntegrationTestResult:
    """Tracks test results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def add_pass(self, name: str, message: str = ""):
        self.passed.append((name, message))
        logger.info(f"  ✓ {name}" + (f" - {message}" if message else ""))

    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        logger.error(f"  ✗ {name} - {error}")

    def add_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        logger.warning(f"  ○ {name} - {reason}")

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        return f"\n{'='*60}\nResults: {len(self.passed)}/{total} passed, {len(self.failed)} failed, {len(self.skipped)} skipped\n{'='*60}"


async def test_api_imports(result: IntegrationTestResult):
    """Test that all API components can be imported."""
    logger.info("\n[1] Testing API Imports...")

    try:
        from api.services import (
            WorldService, GameService, AgentService,
            PlayerService, PartyService, DebugService,
            DialogueService, InteractionService
        )
        result.add_pass("All services import", f"{8} services")
    except Exception as e:
        result.add_fail("Service imports", str(e))
        return

    try:
        from api.routers import worlds, game, agents, player, party, debug, dialogue
        result.add_pass("All routers import", "7 routers")
    except Exception as e:
        result.add_fail("Router imports", str(e))

    try:
        from api.websocket_manager import WebSocketManager, ws_manager
        result.add_pass("WebSocket manager imports", "")
    except Exception as e:
        result.add_fail("WebSocket imports", str(e))

    try:
        from api.main import app
        result.add_pass("FastAPI app creation", f"{len(app.routes)} routes")
    except Exception as e:
        result.add_fail("App creation", str(e))


async def test_agent_framework(result: IntegrationTestResult):
    """Test agent framework components."""
    logger.info("\n[2] Testing Agent Framework...")

    try:
        from agents.debug import AgentInspector, LLMCallTracker
        result.add_pass("Debug tools import", "")
    except Exception as e:
        result.add_fail("Debug tools import", str(e))

    try:
        from agents.simulation import (
            HierarchicalScheduler, AgentTier, HierarchicalOrchestrator
        )
        result.add_pass("Hierarchical scheduler import", "7 tiers defined")
    except Exception as e:
        result.add_fail("Hierarchical scheduler import", str(e))

    try:
        from agents.collective import CollectiveManager, SettlementAgent
        result.add_pass("Collective agents import", "")
    except Exception as e:
        result.add_fail("Collective agents import", str(e))

    # Test scheduler functionality
    try:
        from agents.simulation import HierarchicalScheduler, AgentTier
        scheduler = HierarchicalScheduler()
        agent_id = uuid4()

        scheduler.register_agent(agent_id, "Test Agent", 100, 100)
        tier = scheduler.get_agent_tier(agent_id)
        assert tier is not None

        result.add_pass("Scheduler register/tier", f"default tier: {tier.name}")
    except Exception as e:
        result.add_fail("Scheduler functionality", str(e))


async def test_dialogue_system(result: IntegrationTestResult):
    """Test the dialogue system."""
    logger.info("\n[3] Testing Dialogue System...")

    try:
        from api.services.dialogue_service import DialogueService, Conversation

        service = DialogueService()

        # Start conversation
        conv = await service.start_conversation(
            session_id="test-session",
            player_id="player-1",
            player_name="Hero",
            agent_id=str(uuid4()),
            agent_name="Elena",
            player_x=0, player_y=0
        )

        assert isinstance(conv, Conversation)
        assert len(conv.turns) >= 1

        result.add_pass("Start conversation", f"{len(conv.turns)} initial turns")
    except Exception as e:
        result.add_fail("Start conversation", str(e))

    try:
        # Continue conversation
        turn = await service.continue_conversation(
            conversation_id=conv.conversation_id,
            speaker_id="player-1",
            speaker_name="Hero",
            message="Hello there!"
        )

        assert turn is not None
        result.add_pass("Continue conversation", f"turn added")
    except Exception as e:
        result.add_fail("Continue conversation", str(e))

    try:
        # End conversation
        ended = await service.end_conversation(conv.conversation_id)
        assert ended is not None
        assert ended.state.value == "ended"

        result.add_pass("End conversation", f"{len(ended.turns)} total turns")
    except Exception as e:
        result.add_fail("End conversation", str(e))


async def test_interaction_service(result: IntegrationTestResult):
    """Test the interaction service."""
    logger.info("\n[4] Testing Interaction Service...")

    try:
        from api.services.interaction_service import InteractionService, InteractionResult

        service = InteractionService()

        # Test tile interaction
        res = await service.interact(
            session_id="test-session",
            player_id="player-1",
            player_name="Hero",
            player_x=10, player_y=10,
            target_type="tile",
            target_id="10,10",
            action="examine"
        )

        assert isinstance(res, InteractionResult)
        assert res.success

        result.add_pass("Tile examine", res.message[:40])
    except Exception as e:
        result.add_fail("Tile interaction", str(e))

    try:
        # Test object interaction
        res = await service.interact(
            session_id="test-session",
            player_id="player-1",
            player_name="Hero",
            player_x=10, player_y=10,
            target_type="object",
            target_id="obj-123",
            action="examine"
        )

        assert res.success
        result.add_pass("Object examine", "")
    except Exception as e:
        result.add_fail("Object interaction", str(e))

    try:
        # Get available actions
        actions = await service.get_available_actions("agent", "agent-123")
        assert "talk" in actions
        assert "examine" in actions

        result.add_pass("Available actions", f"{len(actions)} actions for agent")
    except Exception as e:
        result.add_fail("Available actions", str(e))


async def test_debug_service(result: IntegrationTestResult):
    """Test debug service integration."""
    logger.info("\n[5] Testing Debug Service...")

    try:
        from api.services.debug_service import DebugService

        service = DebugService()

        # Test inspector lazy loading
        inspector = service._get_inspector()
        assert inspector is not None

        result.add_pass("Inspector lazy load", type(inspector).__name__)
    except Exception as e:
        result.add_fail("Inspector lazy load", str(e))

    try:
        # Test tracker lazy loading
        tracker = service._get_llm_tracker()
        assert tracker is not None

        result.add_pass("LLM tracker lazy load", type(tracker).__name__)
    except Exception as e:
        result.add_fail("LLM tracker lazy load", str(e))

    try:
        # Test simulation stats
        stats = await service.get_simulation_stats("test-session")
        assert stats is not None

        result.add_pass("Simulation stats", f"tick={stats.current_tick}")
    except Exception as e:
        result.add_fail("Simulation stats", str(e))

    try:
        # Test tier distribution
        tiers = await service.get_agents_by_tier("test-session")
        assert "PLAYER_PARTY" in tiers
        assert "DORMANT" in tiers

        result.add_pass("Tier distribution", f"{len(tiers)} tiers")
    except Exception as e:
        result.add_fail("Tier distribution", str(e))


async def test_game_service(result: IntegrationTestResult):
    """Test game service with session management."""
    logger.info("\n[6] Testing Game Service...")

    try:
        from api.services.game_service import GameService

        service = GameService()

        # Create session
        session = await service.create_session(
            world_id=str(uuid4()),
            player_config={
                "name": "Test Hero",
                "spawn_x": 512,
                "spawn_y": 512
            },
            party_config={"size": 2},
            settings={"debug_mode": True}
        )

        assert session.id is not None
        result.add_pass("Create session", f"id={session.id[:8]}...")
    except Exception as e:
        result.add_fail("Create session", str(e))
        return

    try:
        # Get session
        fetched = await service.get_session(session.id)
        assert fetched is not None
        assert fetched.id == session.id

        result.add_pass("Get session", f"status={fetched.status}")
    except Exception as e:
        result.add_fail("Get session", str(e))

    try:
        # Execute tick
        tick_result = await service.tick(session.id)
        assert tick_result is not None
        assert tick_result.tick_number > 0

        result.add_pass("Execute tick", f"tick={tick_result.tick_number}, time={tick_result.game_time}")
    except Exception as e:
        result.add_fail("Execute tick", str(e))

    try:
        # Get tier distribution
        tiers = await service.get_tier_distribution(session.id)
        assert tiers is not None

        result.add_pass("Tier distribution", f"party={tiers.get('PLAYER_PARTY', 0)}")
    except Exception as e:
        result.add_fail("Tier distribution", str(e))

    try:
        # End session
        ended = await service.end_session(session.id)
        assert ended

        result.add_pass("End session", "")
    except Exception as e:
        result.add_fail("End session", str(e))


async def test_websocket_manager(result: IntegrationTestResult):
    """Test WebSocket manager functionality."""
    logger.info("\n[7] Testing WebSocket Manager...")

    try:
        from api.websocket_manager import WebSocketManager

        manager = WebSocketManager()

        # Check methods exist
        methods = [
            'broadcast_tick', 'broadcast_agent_update', 'broadcast_chat',
            'broadcast_event', 'broadcast_dialogue', 'broadcast_dialogue_started',
            'broadcast_dialogue_ended', 'broadcast_agent_action', 'broadcast_tier_update'
        ]

        missing = [m for m in methods if not hasattr(manager, m)]
        if missing:
            result.add_fail("WebSocket methods", f"missing: {missing}")
        else:
            result.add_pass("WebSocket methods", f"{len(methods)} broadcast methods")
    except Exception as e:
        result.add_fail("WebSocket manager", str(e))


async def test_hierarchical_orchestrator(result: IntegrationTestResult):
    """Test hierarchical orchestrator creation."""
    logger.info("\n[8] Testing Hierarchical Orchestrator...")

    try:
        from agents.simulation import HierarchicalOrchestrator
        from agents.simulation.time_manager import TimeManager
        from agents.simulation.scheduler import AgentScheduler
        from agents.simulation.events import EventSystem
        from agents.simulation.hierarchical_scheduler import HierarchicalScheduler
        from agents.collective import CollectiveManager

        orchestrator = HierarchicalOrchestrator(
            time_manager=TimeManager(),
            scheduler=AgentScheduler(max_concurrent=10),
            event_system=EventSystem(),
            hierarchical_scheduler=HierarchicalScheduler(),
            collective_manager=CollectiveManager(),
            tick_rate=1.0,
            max_agents=1000
        )

        assert orchestrator is not None
        result.add_pass("Orchestrator creation", "")
    except Exception as e:
        result.add_fail("Orchestrator creation", str(e))

    try:
        # Register agent
        agent_id = uuid4()
        orchestrator.register_agent(agent_id, "Test Agent", 100, 100)

        # Check agent was registered
        tier = orchestrator.hierarchical_scheduler.get_agent_tier(agent_id)
        assert tier is not None

        result.add_pass("Agent registration", f"tier={tier.name}")
    except Exception as e:
        result.add_fail("Agent registration", str(e))

    try:
        # Set player
        player_id = uuid4()
        orchestrator.set_player(player_id, 100, 100)

        # Check player context was set via the hierarchical scheduler
        assert orchestrator.hierarchical_scheduler._player_context is not None
        result.add_pass("Player context set", "")
    except Exception as e:
        result.add_fail("Player context", str(e))


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("UNIVERSA FULL STACK INTEGRATION TESTS")
    print("=" * 60)

    result = IntegrationTestResult()

    # Run all test suites
    await test_api_imports(result)
    await test_agent_framework(result)
    await test_dialogue_system(result)
    await test_interaction_service(result)
    await test_debug_service(result)
    await test_game_service(result)
    await test_websocket_manager(result)
    await test_hierarchical_orchestrator(result)

    # Print summary
    print(result.summary())

    # Return exit code
    return 0 if len(result.failed) == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
