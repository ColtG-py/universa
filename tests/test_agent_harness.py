#!/usr/bin/env python3
"""
Agent Simulation Test Harness
Tests the full agent lifecycle: perceive → retrieve → reflect → plan → act
"""

import asyncio
import sys
import os
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm.ollama_client import OllamaClient, get_ollama_client
from agents.memory.memory_stream import MemoryStream
from agents.world.interface import WorldInterface, LocationData
from agents.graph.agent_graph import AgentGraph, create_agent_graph, ExecutionResult
from agents.simulation.orchestrator import SimulationOrchestrator
from agents.simulation.time_manager import TimeManager, SimulationTime
from agents.simulation.events import EventSystem, EventType, EventScope


class MockWorldInterface(WorldInterface):
    """
    Mock world interface for testing without full world data.
    Provides simple location data and agent awareness.
    """

    def __init__(self):
        super().__init__(world_state=None)
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._locations: Dict[str, LocationData] = {}

        # Create some test locations
        self._setup_test_world()

    def _setup_test_world(self):
        """Setup a simple test world"""
        # Village center
        self._locations["village_center"] = LocationData(
            x=100, y=100, chunk_x=0, chunk_y=0,
            biome_type="temperate_grassland",
            temperature_c=18.0,
            has_road=True,
            road_type="main_road",
            settlement_id=1,
            settlement_type="village",
            faction_id=1,
            faction_name="Riverside",
        )

        # Market square
        self._locations["market"] = LocationData(
            x=105, y=100, chunk_x=0, chunk_y=0,
            biome_type="temperate_grassland",
            temperature_c=18.0,
            has_road=True,
            settlement_id=1,
        )

        # Forest edge
        self._locations["forest"] = LocationData(
            x=150, y=100, chunk_x=0, chunk_y=0,
            biome_type="temperate_deciduous_forest",
            temperature_c=16.0,
            timber_quality=0.8,
            timber_type="hardwood",
            vegetation_density=0.9,
        )

        # River
        self._locations["river"] = LocationData(
            x=100, y=150, chunk_x=0, chunk_y=0,
            biome_type="temperate_grassland",
            has_water=True,
            river_flow=50.0,
            fishing_quality=0.7,
        )

    def register_test_agent(self, agent_id: str, name: str, location: str):
        """Register an agent in the test world"""
        loc = self._locations.get(location, self._locations["village_center"])
        self._agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "x": loc.x,
            "y": loc.y,
            "location_name": location,
        }

    def query_location(self, x: int, y: int, world_state=None) -> Optional[LocationData]:
        """Return test location data"""
        # Find closest location
        for name, loc in self._locations.items():
            if abs(loc.x - x) <= 10 and abs(loc.y - y) <= 10:
                return loc

        # Default location
        return LocationData(
            x=x, y=y, chunk_x=x // 256, chunk_y=y // 256,
            biome_type="temperate_grassland",
            temperature_c=18.0,
        )

    def get_location_description(self, location_name: str) -> str:
        """Get a natural language description of a location"""
        descriptions = {
            "village_center": "the village center of Riverside, with cobblestone streets and a central well",
            "market": "the bustling market square where merchants sell their wares",
            "forest": "the edge of a dense deciduous forest with tall oak trees",
            "river": "the banks of a gentle river teeming with fish",
        }
        return descriptions.get(location_name, "an unremarkable stretch of grassland")

    def get_nearby_npcs(self, location_name: str) -> list:
        """Get NPCs at a location (for testing)"""
        npcs = {
            "village_center": ["the village elder", "a wandering merchant"],
            "market": ["a baker selling bread", "a smith hammering at his forge"],
            "forest": ["a woodcutter gathering timber"],
            "river": ["a fisherman casting his line"],
        }
        return npcs.get(location_name, [])


class TestHarness:
    """
    Test harness for running agent simulations.
    """

    def __init__(self):
        self.ollama: Optional[OllamaClient] = None
        self.world: Optional[MockWorldInterface] = None
        self.orchestrator: Optional[SimulationOrchestrator] = None
        self.agents: Dict[str, AgentGraph] = {}

    async def setup(self) -> bool:
        """Initialize all systems"""
        print("=" * 60)
        print("AGENT SIMULATION TEST HARNESS")
        print("=" * 60)

        # 1. Check Ollama availability
        print("\n[1/4] Checking Ollama connection...")
        try:
            self.ollama = get_ollama_client()
            if await self.ollama.is_available():
                models = await self.ollama.list_models()
                model_names = [m.get("name", "") for m in models]
                print(f"  ✓ Ollama connected. Available models: {len(models)}")

                # Check for required models
                has_qwen = any("qwen3" in m.lower() for m in model_names)
                has_embed = any("nomic-embed" in m.lower() for m in model_names)

                if has_qwen:
                    print(f"  ✓ Qwen3 model available")
                else:
                    print(f"  ✗ Qwen3 model NOT found. Run: ollama pull qwen3:8b")
                    return False

                if has_embed:
                    print(f"  ✓ Embedding model available")
                else:
                    print(f"  ⚠ Embedding model not found. Run: ollama pull nomic-embed-text")
            else:
                print("  ✗ Ollama not available. Start with: ollama serve")
                return False
        except Exception as e:
            print(f"  ✗ Ollama error: {e}")
            return False

        # 2. Setup mock world
        print("\n[2/4] Setting up test world...")
        self.world = MockWorldInterface()
        print(f"  ✓ Test world created with {len(self.world._locations)} locations")

        # 3. Setup simulation orchestrator
        print("\n[3/4] Initializing simulation orchestrator...")
        start_time = SimulationTime(year=1, month=6, day=15, hour=8, minute=0)
        time_manager = TimeManager(start_time=start_time, tick_minutes=15)
        self.orchestrator = SimulationOrchestrator(
            time_manager=time_manager,
            tick_rate=0,  # Manual stepping for tests
        )
        print(f"  ✓ Simulation ready at {start_time.to_string()}")

        # 4. Create test agents
        print("\n[4/4] Creating test agents...")
        await self._create_test_agents()

        print("\n" + "=" * 60)
        print("SETUP COMPLETE")
        print("=" * 60)
        return True

    async def _create_test_agents(self):
        """Create test agents"""

        # Elena the Blacksmith
        elena_id = uuid4()
        elena_summary = """Name: Elena (age: 28)
Innate traits: hardworking, friendly, skilled with her hands
Elena is the village blacksmith who took over her father's forge three years ago.
She enjoys her craft and takes pride in making quality tools for the villagers.
She is known for her warm smile and willingness to help neighbors."""

        elena_memory = MemoryStream(agent_id=elena_id)

        # Set up LLM-based importance scoring
        async def score_importance(text: str) -> float:
            try:
                prompt = f"""On a scale of 1-10, rate the importance of this memory for an agent's life.
1 = mundane (brushing teeth), 10 = life-changing (death of loved one).
Memory: {text}
Respond with just the number."""
                response = await self.ollama.generate(prompt, max_tokens=10, temperature=0.3)
                # Parse number from response
                for char in response.text:
                    if char.isdigit():
                        return int(char) / 10.0
                return 0.5
            except Exception:
                return 0.5

        # Set up embedding generation
        async def generate_embedding(text: str) -> list:
            try:
                return await self.ollama.embed(text)
            except Exception:
                return []

        # Note: MemoryStream expects sync functions, so we wrap them
        def sync_importance(text: str) -> float:
            return asyncio.get_event_loop().run_until_complete(score_importance(text))

        def sync_embedding(text: str) -> list:
            return asyncio.get_event_loop().run_until_complete(generate_embedding(text))

        elena_memory.set_importance_scorer(sync_importance)
        elena_memory.set_embedding_generator(sync_embedding)

        # Register in world
        self.world.register_test_agent(str(elena_id), "Elena", "village_center")

        # Create agent graph
        elena_graph = create_agent_graph(
            agent_id=elena_id,
            agent_name="Elena",
            memory_stream=elena_memory,
            world=self.world,
            ollama_client=self.ollama,
            agent_summary=elena_summary,
        )

        self.agents["elena"] = elena_graph

        # Register with orchestrator
        self.orchestrator.register_agent(
            agent_id=elena_id,
            name="Elena",
            location="village_center",
            memory_stream=elena_memory,
            graph_runner=elena_graph,
        )

        print(f"  ✓ Created agent: Elena the Blacksmith (ID: {elena_id})")

        # Seed some initial memories
        await elena_memory.add_observation(
            "Elena woke up in her room above the forge as the morning sun streamed through the window.",
            location_x=100, location_y=100,
        )
        await elena_memory.add_observation(
            "Elena ate a simple breakfast of bread and cheese.",
            location_x=100, location_y=100,
        )
        await elena_memory.add_observation(
            "Elena lit the forge fire and prepared her tools for the day's work.",
            location_x=100, location_y=100,
        )
        print(f"  ✓ Seeded 3 initial memories for Elena")

    async def run_single_cycle(self, agent_name: str = "elena") -> ExecutionResult:
        """Run a single agent cycle and display results"""
        agent = self.agents.get(agent_name)
        if not agent:
            print(f"Agent '{agent_name}' not found")
            return None

        print(f"\n{'='*60}")
        print(f"RUNNING CYCLE FOR: {agent_name.upper()}")
        print(f"Time: {self.orchestrator.get_time().to_string()}")
        print(f"{'='*60}")

        start = datetime.utcnow()
        result = await agent.run_cycle(
            current_time=datetime.utcnow(),
            max_iterations=10,
        )
        duration = (datetime.utcnow() - start).total_seconds()

        print(f"\nExecution completed in {duration:.2f}s")
        print(f"Status: {result.status.value}")
        print(f"Nodes visited: {result.nodes_executed}")

        if result.action_taken:
            print(f"\nAction taken: {result.action_taken}")

        if result.error:
            print(f"\nError: {result.error}")

        # Show state details
        state = result.state
        if state.perception:
            print(f"\nPerception:")
            print(f"  Location: ({state.perception.location_x}, {state.perception.location_y})")
            if state.perception.observations:
                print(f"  Observations: {len(state.perception.observations)}")
                for obs in state.perception.observations[:3]:
                    print(f"    - {obs}")

        if state.retrieved_memories:
            print(f"\nRetrieved memories: {len(state.retrieved_memories)}")
            for mem in state.retrieved_memories[:3]:
                print(f"  - {mem.description[:60]}...")

        if state.current_plan:
            print(f"\nCurrent plan: {state.current_plan}")

        return result

    async def run_simulation_ticks(self, num_ticks: int = 4):
        """Run multiple simulation ticks"""
        print(f"\n{'='*60}")
        print(f"RUNNING {num_ticks} SIMULATION TICKS")
        print(f"{'='*60}")

        for i in range(num_ticks):
            print(f"\n--- TICK {i+1}/{num_ticks} ---")

            # Step the simulation
            result = await self.orchestrator.step()

            print(f"Time: {result.simulation_time['hour']:02d}:{result.simulation_time['minute']:02d}")
            print(f"Time of day: {result.simulation_time['time_of_day']}")

            if result.time_events:
                print(f"Time events: {result.time_events}")

            if result.world_events:
                for event in result.world_events:
                    print(f"World event: {event.title}")

            if result.agent_actions:
                for agent_id, actions in result.agent_actions.items():
                    print(f"Agent actions: {len(actions)}")
                    for action in actions[:2]:
                        if isinstance(action, dict):
                            print(f"  - {action.get('type', action)}")

            print(f"Duration: {result.duration_ms:.1f}ms")

    async def test_memory_retrieval(self, agent_name: str = "elena"):
        """Test memory retrieval"""
        agent = self.agents.get(agent_name)
        if not agent:
            return

        print(f"\n{'='*60}")
        print("TESTING MEMORY RETRIEVAL")
        print(f"{'='*60}")

        # Add some more memories
        await agent.memory_stream.add_observation(
            "A farmer named Thomas arrived at the forge asking about a new plow blade.",
        )
        await agent.memory_stream.add_observation(
            "Elena agreed to make Thomas a plow blade for 5 silver coins.",
        )

        # Retrieve with query
        print("\nQuerying: 'plow blade'")
        memories = await agent.memory_stream.retrieve(query="plow blade", limit=5)
        print(f"Found {len(memories)} relevant memories:")
        for mem in memories:
            print(f"  [{mem.memory_type.value}] {mem.description[:60]}...")
            print(f"    Importance: {mem.importance:.2f}")

        # Check reflection trigger
        print(f"\nReflection trigger status:")
        print(f"  Accumulated importance: {agent.memory_stream.get_importance_sum():.1f}")
        print(f"  Should reflect: {agent.memory_stream.should_reflect()}")

    async def test_llm_generation(self):
        """Test basic LLM generation"""
        print(f"\n{'='*60}")
        print("TESTING LLM GENERATION")
        print(f"{'='*60}")

        # Test basic generation
        prompt = "Elena the blacksmith is starting her day. In one sentence, describe what she does first."
        print(f"\nPrompt: {prompt}")

        response = await self.ollama.generate(
            prompt=prompt,
            system="You are narrating the life of a medieval blacksmith. Be concise.",
            temperature=0.7,
            max_tokens=100,
        )

        print(f"Response: {response.text}")
        print(f"Tokens generated: {response.eval_count}")

    async def test_embedding(self):
        """Test embedding generation"""
        print(f"\n{'='*60}")
        print("TESTING EMBEDDINGS")
        print(f"{'='*60}")

        texts = [
            "Elena is working at the forge",
            "The blacksmith hammers hot metal",
            "A farmer tends to his crops",
        ]

        print("\nGenerating embeddings...")
        embeddings = await self.ollama.embed_batch(texts)

        print(f"Generated {len(embeddings)} embeddings")
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            print(f"  [{i+1}] '{text[:40]}...' -> dim={len(emb)}")

        # Calculate similarities
        if len(embeddings) >= 3:
            from agents.memory.memory_stream import MemoryStream
            ms = MemoryStream(agent_id=uuid4())

            sim_01 = ms._cosine_similarity(embeddings[0], embeddings[1])
            sim_02 = ms._cosine_similarity(embeddings[0], embeddings[2])

            print(f"\nSimilarities:")
            print(f"  'forge work' vs 'hammering metal': {sim_01:.3f}")
            print(f"  'forge work' vs 'farming': {sim_02:.3f}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator:
            await self.orchestrator.stop()
        print("\nTest harness cleaned up.")


async def main():
    """Main test runner"""
    harness = TestHarness()

    try:
        # Setup
        if not await harness.setup():
            print("\nSetup failed. Please check the error messages above.")
            return

        # Run tests
        print("\n" + "=" * 60)
        print("RUNNING TESTS")
        print("=" * 60)

        # Test 1: LLM Generation
        await harness.test_llm_generation()

        # Test 2: Embeddings
        await harness.test_embedding()

        # Test 3: Memory Retrieval
        await harness.test_memory_retrieval()

        # Test 4: Single Agent Cycle
        await harness.run_single_cycle("elena")

        # Test 5: Simulation Ticks
        await harness.run_simulation_ticks(4)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await harness.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
