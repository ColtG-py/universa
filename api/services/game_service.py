"""
Game Service
Handles game session management, tick execution, and agent orchestration.
Integrates with the agent framework for simulation.
"""

import logging
import asyncio
import sys
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from uuid import uuid4, UUID
from datetime import datetime

# Add agents to path
sys.path.insert(0, '/home/colt/Projects/universa')

from api.models.responses import (
    SessionResponse,
    TickResponse,
    AutoTickResponse,
    SessionStateResponse,
    AgentUpdateEntry
)
from api.websocket_manager import ws_manager
from api.services.supabase_client import get_supabase_client, SupabaseClient

if TYPE_CHECKING:
    from api.services.world_service import WorldService

logger = logging.getLogger(__name__)


class GameService:
    """
    Service for game session management.

    Integrates with:
    - Agent framework for agent simulation
    - World service for world data
    - WebSocket for real-time updates
    """

    def __init__(self):
        # Active sessions: session_id -> session data
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # Auto-tick tasks
        self._auto_tick_tasks: Dict[str, asyncio.Task] = {}
        # Agent orchestrators per session
        self._orchestrators: Dict[str, Any] = {}
        # World service reference (injected)
        self._world_service: Optional["WorldService"] = None
        # LLM client (shared)
        self._ollama_client = None
        # Supabase client (lazy-loaded)
        self._db: Optional[SupabaseClient] = None
        # Auto-save counter (save every N ticks)
        self._save_interval = 10

    def _get_db(self) -> Optional[SupabaseClient]:
        """Get Supabase client, initializing if needed."""
        if self._db is None:
            self._db = get_supabase_client()
        return self._db

    def set_world_service(self, world_service: "WorldService"):
        """Set the world service reference."""
        self._world_service = world_service

    def set_supabase_client(self, client: SupabaseClient):
        """Set the Supabase client (legacy method)."""
        self._db = client

    def _get_ollama_client(self):
        """Lazy-load Ollama client."""
        if self._ollama_client is None:
            try:
                from agents.llm.ollama_client import OllamaClient
                self._ollama_client = OllamaClient()
                logger.info("Ollama client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
        return self._ollama_client

    async def create_session(
        self,
        world_id: str,
        player_config: Dict[str, Any],
        party_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> SessionResponse:
        """
        Create a new game session.

        This will:
        1. Validate the world exists and is ready
        2. Create the player character
        3. Create party members as AI agents
        4. Initialize the simulation orchestrator
        """
        session_id = str(uuid4())
        player_id = str(uuid4())
        party_id = str(uuid4()) if party_config else None

        # Default settings
        default_settings = {
            "tick_interval_ms": 1000,
            "auto_tick": False,
            "debug_mode": False,
            "difficulty": "normal",
            "simulation_speed": 1.0,  # Game minutes per real second
        }
        if settings:
            default_settings.update(settings)

        # Get spawn location
        # NOTE: Existing world data only covers tiles 0-63 (4 chunks of 32x32)
        # Default spawn to center of available area
        spawn_x = player_config.get("spawn_x", 32)
        spawn_y = player_config.get("spawn_y", 32)

        # If spawn_settlement_id provided, get settlement location
        if player_config.get("spawn_settlement_id") and self._world_service:
            settlements = await self._world_service.get_settlements(world_id)
            for s in settlements:
                if s.get("id") == player_config.get("spawn_settlement_id"):
                    spawn_x = s.get("x", spawn_x)
                    spawn_y = s.get("y", spawn_y)
                    break

        session_data = {
            "id": session_id,
            "world_id": world_id,
            "player_id": player_id,
            "party_id": party_id,
            "created_at": datetime.utcnow(),
            "status": "active",
            "current_tick": 0,
            "game_time": "Day 1, 08:00",
            "game_minutes": 8 * 60,  # Start at 8:00 AM
            "settings": default_settings,
            "player": {
                "id": player_id,
                "name": player_config.get("name", "Adventurer"),
                "x": spawn_x,
                "y": spawn_y,
                "health": 100.0,
                "max_health": 100.0,
                "stamina": 100.0,
                "max_stamina": 100.0,
                "stats": {
                    "strength": player_config.get("strength", 10),
                    "dexterity": player_config.get("dexterity", 10),
                    "constitution": player_config.get("constitution", 10),
                    "intelligence": player_config.get("intelligence", 10),
                    "wisdom": player_config.get("wisdom", 10),
                    "charisma": player_config.get("charisma", 10),
                }
            },
            "party_members": [],
            "agents": {},  # agent_id -> agent data
            "events": []
        }

        # Create party members if requested
        if party_config:
            await self._create_party_members(session_data, party_config, spawn_x, spawn_y)

        # Initialize simulation orchestrator
        await self._init_orchestrator(session_id, session_data)

        self._sessions[session_id] = session_data

        # Persist to database
        await self._persist_new_session(session_data)

        logger.info(f"Created session {session_id} for world {world_id}")

        return SessionResponse(
            id=session_id,
            world_id=world_id,
            player_id=player_id,
            party_id=party_id,
            created_at=session_data["created_at"],
            status="active",
            current_tick=0,
            game_time=session_data["game_time"],
            settings=default_settings
        )

    async def _create_party_members(
        self,
        session_data: Dict,
        party_config: Dict,
        spawn_x: int,
        spawn_y: int
    ):
        """Create party members as AI agents."""
        size = party_config.get("size", 3)
        roles = party_config.get("roles") or ["warrior", "mage", "healer", "rogue", "ranger"]
        names = party_config.get("names") or ["Aldric", "Elena", "Theron", "Lyra", "Kael", "Mira"]

        # Role to stats mapping
        role_stats = {
            "warrior": {"strength": 16, "constitution": 14, "dexterity": 12},
            "mage": {"intelligence": 16, "wisdom": 14, "charisma": 12},
            "healer": {"wisdom": 16, "charisma": 14, "constitution": 12},
            "rogue": {"dexterity": 16, "intelligence": 14, "charisma": 12},
            "ranger": {"dexterity": 16, "wisdom": 14, "strength": 12},
            "companion": {"charisma": 14, "wisdom": 12, "constitution": 12},
        }

        # Role to personality mapping
        role_personalities = {
            "warrior": "brave, protective, straightforward",
            "mage": "curious, analytical, sometimes absent-minded",
            "healer": "compassionate, calm, wise",
            "rogue": "witty, cunning, observant",
            "ranger": "quiet, nature-loving, vigilant",
            "companion": "loyal, supportive, adaptable",
        }

        for i in range(min(size, 6)):
            role = roles[i] if i < len(roles) else "companion"
            name = names[i] if i < len(names) else f"Companion {i+1}"

            # Base stats with role bonuses
            base_stats = {
                "strength": 10, "dexterity": 10, "constitution": 10,
                "intelligence": 10, "wisdom": 10, "charisma": 10
            }
            base_stats.update(role_stats.get(role, {}))

            # Position around player
            offset_x = (i % 3) - 1  # -1, 0, 1
            offset_y = (i // 3) + 1  # 1, 2

            member = {
                "id": str(uuid4()),
                "name": name,
                "role": role,
                "x": spawn_x + offset_x,
                "y": spawn_y + offset_y,
                "health": 1.0,
                "energy": 1.0,
                "morale": 1.0,
                "stats": base_stats,
                "personality": role_personalities.get(role, "friendly and helpful"),
                "current_action": None,
                "current_thought": None,
                "following": True,
                "loyalty": 0.6,
                "trust": 0.5,
                "affection": 0.5,
                "tier": "active",  # Party members are always active tier
            }
            session_data["party_members"].append(member)
            session_data["agents"][member["id"]] = member

        logger.info(f"Created {len(session_data['party_members'])} party members")

    async def _init_orchestrator(self, session_id: str, session_data: Dict):
        """Initialize the hierarchical simulation orchestrator for this session."""
        try:
            from agents.simulation.hierarchical_orchestrator import HierarchicalOrchestrator
            from agents.simulation.hierarchical_scheduler import HierarchicalScheduler
            from agents.simulation.time_manager import TimeManager
            from agents.simulation.scheduler import AgentScheduler
            from agents.simulation.events import EventSystem
            from agents.collective import CollectiveManager

            time_manager = TimeManager()
            scheduler = AgentScheduler(max_concurrent=10)
            event_system = EventSystem()
            hierarchical_scheduler = HierarchicalScheduler()
            collective_manager = CollectiveManager()

            orchestrator = HierarchicalOrchestrator(
                time_manager=time_manager,
                scheduler=scheduler,
                event_system=event_system,
                hierarchical_scheduler=hierarchical_scheduler,
                collective_manager=collective_manager,
                tick_rate=1.0,
                max_agents=5000,  # Support larger populations with hierarchical scheduling
            )

            # Set player context
            player = session_data.get("player", {})
            player_id = session_data.get("player_id")
            if player_id:
                from uuid import UUID
                orchestrator.set_player(
                    player_id=UUID(player_id),
                    x=player.get("x", 0),
                    y=player.get("y", 0),
                )

            # Register party members
            for member in session_data.get("party_members", []):
                member_id = UUID(member["id"])
                orchestrator.register_agent(
                    agent_id=member_id,
                    name=member["name"],
                    x=member.get("x", 0),
                    y=member.get("y", 0),
                )
                orchestrator.add_to_party(member_id)

            self._orchestrators[session_id] = orchestrator
            logger.info(f"Initialized hierarchical orchestrator for session {session_id}")

        except ImportError as e:
            logger.warning(f"Could not initialize hierarchical orchestrator: {e}")
            # Fallback to base orchestrator
            try:
                from agents.simulation.orchestrator import SimulationOrchestrator
                from agents.simulation.time_manager import TimeManager
                from agents.simulation.scheduler import AgentScheduler
                from agents.simulation.events import EventSystem

                orchestrator = SimulationOrchestrator(
                    time_manager=TimeManager(),
                    scheduler=AgentScheduler(max_concurrent=10),
                    event_system=EventSystem(),
                    tick_rate=1.0,
                )
                self._orchestrators[session_id] = orchestrator
                logger.info(f"Initialized fallback orchestrator for session {session_id}")
            except Exception as e2:
                logger.warning(f"Could not initialize fallback orchestrator: {e2}")
                self._orchestrators[session_id] = None

    async def _persist_new_session(self, session_data: Dict[str, Any]):
        """Persist a new session and its player to the database."""
        db = self._get_db()
        if not db:
            return

        try:
            # Save session
            await db.create_session(
                session_id=session_data["id"],
                world_id=session_data["world_id"],
                settings=session_data.get("settings", {})
            )

            # Save player
            player = session_data.get("player", {})
            await db.create_player(
                player_id=session_data["player_id"],
                session_id=session_data["id"],
                name=player.get("name", "Adventurer"),
                x=player.get("x", 512),
                y=player.get("y", 512),
                stats=player.get("stats", {})
            )

            # Save party members as agents
            for member in session_data.get("party_members", []):
                await db.save_agent(
                    agent_id=member["id"],
                    session_id=session_data["id"],
                    world_id=session_data["world_id"],
                    name=member["name"],
                    agent_type="party_member",
                    x=member.get("x", 0),
                    y=member.get("y", 0),
                    tier="active",
                    stats=member.get("stats", {})
                )

            logger.info(f"Persisted session {session_data['id']} to database")
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

    async def _save_session_state(self, session_id: str):
        """Save current session state to the database."""
        db = self._get_db()
        session = self._sessions.get(session_id)
        if not db or not session:
            return

        try:
            # Update session
            await db.update_session(
                session_id=session_id,
                current_tick=session.get("current_tick", 0),
                game_time=session.get("game_time"),
                status=session.get("status")
            )

            # Update player position and stats
            player = session.get("player", {})
            await db.update_player(
                player_id=session["player_id"],
                x=player.get("x"),
                y=player.get("y"),
                health=player.get("health"),
                stamina=player.get("stamina")
            )

            logger.debug(f"Saved session state for {session_id}")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    async def _load_session_from_db(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session from the database."""
        db = self._get_db()
        if not db:
            return None

        try:
            # Get session record
            db_session = await db.get_session(session_id)
            if not db_session:
                return None

            # Get player record
            db_player = await db.get_player(session_id)

            # Build session data structure
            session_data = {
                "id": session_id,
                "world_id": db_session.get("world_id"),
                "player_id": db_player.get("player_id") if db_player else str(uuid4()),
                "party_id": None,
                "created_at": db_session.get("created_at"),
                "status": db_session.get("status", "active"),
                "current_tick": db_session.get("current_tick", 0),
                "game_time": db_session.get("game_time", "Day 1, 08:00"),
                "game_minutes": 8 * 60,  # Would need to parse from game_time
                "settings": db_session.get("settings", {}),
                "player": {
                    "id": db_player.get("player_id") if db_player else str(uuid4()),
                    "name": db_player.get("name", "Adventurer") if db_player else "Adventurer",
                    "x": db_player.get("x", 512) if db_player else 512,
                    "y": db_player.get("y", 512) if db_player else 512,
                    "health": db_player.get("health", 100.0) if db_player else 100.0,
                    "max_health": 100.0,
                    "stamina": db_player.get("stamina", 100.0) if db_player else 100.0,
                    "max_stamina": 100.0,
                    "stats": db_player.get("stats", {}) if db_player else {}
                },
                "party_members": [],
                "agents": {},
                "events": []
            }

            # Load party members/agents
            agents = await db.get_agents_for_session(session_id)
            for agent in agents:
                agent_data = {
                    "id": agent.get("agent_id"),
                    "name": agent.get("name", "Unknown"),
                    "x": agent.get("x", 0),
                    "y": agent.get("y", 0),
                    "agent_type": agent.get("agent_type", "npc"),
                    "role": agent.get("agent_type"),
                    "health": 1.0,
                    "energy": 1.0,
                    "tier": agent.get("tier", "background"),
                    "stats": agent.get("stats", {}),
                    "current_action": None,
                    "current_thought": None,
                }
                session_data["agents"][agent_data["id"]] = agent_data
                if agent.get("agent_type") == "party_member":
                    agent_data["following"] = True
                    agent_data["loyalty"] = 0.6
                    agent_data["trust"] = 0.5
                    agent_data["affection"] = 0.5
                    session_data["party_members"].append(agent_data)

            return session_data
        except Exception as e:
            logger.error(f"Failed to load session from DB: {e}")
            return None

    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Get session details from memory or database."""
        session = self._sessions.get(session_id)

        # Try loading from database if not in memory
        if not session:
            session = await self._load_session_from_db(session_id)
            if session:
                self._sessions[session_id] = session
                # Initialize orchestrator for loaded session
                await self._init_orchestrator(session_id, session)

        if not session:
            return None

        return SessionResponse(
            id=session["id"],
            world_id=session["world_id"],
            player_id=session["player_id"],
            party_id=session.get("party_id"),
            created_at=session["created_at"],
            status=session["status"],
            current_tick=session["current_tick"],
            game_time=session["game_time"],
            settings=session["settings"]
        )

    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state with player info.

        Returns a dict that can be serialized for the frontend,
        including complete player data.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        player = session.get("player", {})

        return {
            "session_id": session_id,
            "world_id": session["world_id"],
            "status": session["status"],
            "current_tick": session["current_tick"],
            "game_time": session["game_time"],
            "player": {
                "id": player.get("id", ""),
                "name": player.get("name", "Adventurer"),
                "x": player.get("x", 0),
                "y": player.get("y", 0),
                "health": player.get("health", 100.0),
                "max_health": player.get("max_health", 100.0),
                "stamina": player.get("stamina", 100.0),
                "max_stamina": player.get("max_stamina", 100.0),
                "stats": player.get("stats", {}),
            },
            "player_position": {"x": player.get("x", 0), "y": player.get("y", 0)},
            "party_size": len(session.get("party_members", [])),
            "auto_tick_enabled": session["settings"].get("auto_tick", False)
        }

    async def end_session(self, session_id: str) -> bool:
        """End a game session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Stop auto-tick if running
        await self.stop_auto_tick(session_id)

        # Stop orchestrator
        orchestrator = self._orchestrators.pop(session_id, None)
        if orchestrator:
            try:
                if asyncio.iscoroutinefunction(orchestrator.stop):
                    await orchestrator.stop()
                else:
                    orchestrator.stop()
            except Exception as e:
                logger.warning(f"Error stopping orchestrator: {e}")

        session["status"] = "ended"

        # Save final state and mark as ended in database
        await self._save_session_state(session_id)
        db = self._get_db()
        if db:
            await db.end_session(session_id)

        logger.info(f"Ended session {session_id}")
        return True

    async def tick(self, session_id: str, num_ticks: int = 1) -> Optional[TickResponse]:
        """
        Execute one or more simulation ticks.

        Each tick:
        1. Updates game time
        2. Runs agent cognition cycles
        3. Processes world events
        4. Updates entity positions
        5. Broadcasts changes via WebSocket
        """
        session = self._sessions.get(session_id)
        if not session or session["status"] != "active":
            return None

        start_time = datetime.utcnow()
        events = []
        agent_updates: List[AgentUpdateEntry] = []
        agents_updated = 0

        for _ in range(num_ticks):
            session["current_tick"] += 1
            tick_num = session["current_tick"]

            # Advance game time (1 tick = 5 minutes in-game by default)
            session["game_minutes"] += 5
            session["game_time"] = self._format_game_time(session["game_minutes"])

            # Run agent simulation if orchestrator available
            orchestrator = self._orchestrators.get(session_id)
            if orchestrator:
                try:
                    tick_result = await orchestrator.step()
                    agents_updated += len(tick_result.agent_actions) if hasattr(tick_result, 'agent_actions') else 0

                    # Broadcast agent actions in real-time
                    if hasattr(tick_result, 'agent_actions'):
                        for action in tick_result.agent_actions:
                            await ws_manager.broadcast_agent_action(
                                session_id=session_id,
                                agent_id=str(action.agent_id) if hasattr(action, 'agent_id') else '',
                                agent_name=str(action.agent_name) if hasattr(action, 'agent_name') else 'Agent',
                                action=str(action.action_type) if hasattr(action, 'action_type') else str(action),
                                location=None,  # Would come from action if available
                                details=action.to_dict() if hasattr(action, 'to_dict') else {}
                            )

                    # Collect events
                    if hasattr(tick_result, 'world_events'):
                        for event in tick_result.world_events:
                            events.append({
                                "type": "world_event",
                                "tick": tick_num,
                                "description": str(event)
                            })
                            # Also broadcast events in real-time
                            await ws_manager.broadcast_event(
                                session_id=session_id,
                                event_type="world_event",
                                description=str(event),
                                location=None,
                                involved_agents=[]
                            )
                except Exception as e:
                    logger.error(f"Orchestrator tick failed: {e}")

            # Update party member states (simple simulation if no orchestrator)
            for member in session.get("party_members", []):
                # Decay needs slightly
                member["energy"] = max(0.0, member.get("energy", 1.0) - 0.001)

                # Update position if following player
                if member.get("following"):
                    player = session.get("player", {})
                    # Move towards player if too far
                    dx = player.get("x", 0) - member.get("x", 0)
                    dy = player.get("y", 0) - member.get("y", 0)
                    if abs(dx) > 2:
                        member["x"] += 1 if dx > 0 else -1
                    if abs(dy) > 2:
                        member["y"] += 1 if dy > 0 else -1

                # Create agent update entry
                agent_updates.append(AgentUpdateEntry(
                    agent_id=member.get("id", ""),
                    changes={
                        "name": member.get("name", "Unknown"),
                        "x": member.get("x", 0),
                        "y": member.get("y", 0),
                        "status": "active" if member.get("following") else "idle",
                        "current_action": member.get("current_action"),
                        "energy": member.get("energy", 1.0),
                        "health": member.get("health", 1.0),
                        "role": member.get("role"),
                    }
                ))
                agents_updated += 1

            # Generate periodic events
            if tick_num % 20 == 0:  # Every ~100 game minutes
                events.append({
                    "type": "time_event",
                    "tick": tick_num,
                    "description": f"Time passes. It is now {session['game_time']}."
                })

            # Store events
            session["events"].extend(events[-10:])  # Keep last 10
            if len(session["events"]) > 100:
                session["events"] = session["events"][-100:]

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Periodically save session state to database
        if session["current_tick"] % self._save_interval == 0:
            asyncio.create_task(self._save_session_state(session_id))

        # Broadcast tick update via WebSocket
        await ws_manager.broadcast_tick(
            session_id,
            session["current_tick"],
            session["game_time"],
            events
        )

        return TickResponse(
            session_id=session_id,
            tick_number=session["current_tick"],
            game_time=session["game_time"],
            events=events,
            agent_updates=agent_updates,
            agents_updated=agents_updated,
            duration_ms=duration_ms
        )

    def _format_game_time(self, total_minutes: int) -> str:
        """Format game time from total minutes."""
        day = (total_minutes // (24 * 60)) + 1
        hour = (total_minutes // 60) % 24
        minute = total_minutes % 60
        return f"Day {day}, {hour:02d}:{minute:02d}"

    async def start_auto_tick(
        self,
        session_id: str,
        interval_ms: int = 1000
    ) -> Optional[AutoTickResponse]:
        """Start automatic tick execution."""
        session = self._sessions.get(session_id)
        if not session or session["status"] != "active":
            return None

        # Stop existing auto-tick if any
        await self.stop_auto_tick(session_id)

        # Update settings
        session["settings"]["auto_tick"] = True
        session["settings"]["tick_interval_ms"] = interval_ms

        # Start auto-tick task
        async def auto_tick_loop():
            while session_id in self._sessions and self._sessions[session_id]["settings"]["auto_tick"]:
                await self.tick(session_id, 1)
                await asyncio.sleep(interval_ms / 1000)

        task = asyncio.create_task(auto_tick_loop())
        self._auto_tick_tasks[session_id] = task

        logger.info(f"Started auto-tick for session {session_id} at {interval_ms}ms interval")

        return AutoTickResponse(
            session_id=session_id,
            auto_tick_enabled=True,
            interval_ms=interval_ms
        )

    async def stop_auto_tick(self, session_id: str) -> Optional[AutoTickResponse]:
        """Stop automatic tick execution."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        session["settings"]["auto_tick"] = False

        # Cancel task if running
        task = self._auto_tick_tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped auto-tick for session {session_id}")

        return AutoTickResponse(
            session_id=session_id,
            auto_tick_enabled=False,
            interval_ms=session["settings"]["tick_interval_ms"]
        )

    async def get_auto_tick_status(self, session_id: str) -> Optional[AutoTickResponse]:
        """Get auto-tick status."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        return AutoTickResponse(
            session_id=session_id,
            auto_tick_enabled=session["settings"].get("auto_tick", False),
            interval_ms=session["settings"].get("tick_interval_ms", 1000)
        )

    async def get_events(
        self,
        session_id: str,
        since_tick: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events since a given tick."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        events = session.get("events", [])
        return [e for e in events if e.get("tick", 0) > since_tick][:limit]

    async def get_party_members(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all party members in a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.get("party_members", [])

    async def get_agent(self, session_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent in the session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.get("agents", {}).get(agent_id)

    async def spawn_agent(
        self,
        session_id: str,
        name: str,
        x: int,
        y: int,
        agent_type: str = "npc",
        role: Optional[str] = None,
        personality: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Spawn a new agent in the session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        agent_id = str(uuid4())
        agent = {
            "id": agent_id,
            "name": name,
            "x": x,
            "y": y,
            "agent_type": agent_type,
            "role": role,
            "personality": personality or "neutral and observant",
            "health": 1.0,
            "energy": 1.0,
            "tier": "background",
            "current_action": None,
            "current_thought": None,
            "created_at": datetime.utcnow().isoformat()
        }

        session["agents"][agent_id] = agent

        # Register with orchestrator if available
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'register_agent'):
            from uuid import UUID
            orchestrator.register_agent(
                agent_id=UUID(agent_id),
                name=name,
                x=x,
                y=y,
            )

        logger.info(f"Spawned agent {agent_id} ({name}) in session {session_id}")
        return agent

    # ==================== Tier Tracking ====================

    async def get_tier_distribution(self, session_id: str) -> Dict[str, int]:
        """Get count of agents in each tier."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'get_tier_distribution'):
            return orchestrator.get_tier_distribution()
        return {}

    async def get_agent_tier(self, session_id: str, agent_id: str) -> Optional[str]:
        """Get the current tier of an agent."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'get_agent_tier'):
            from uuid import UUID
            return orchestrator.get_agent_tier(UUID(agent_id))
        return None

    async def get_hierarchical_stats(self, session_id: str) -> Dict[str, Any]:
        """Get detailed hierarchical execution statistics."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'get_stats'):
            stats = orchestrator.get_stats()
            return stats.get('hierarchical', {})
        return {}

    async def update_player_position(
        self,
        session_id: str,
        x: int,
        y: int,
        settlement_id: Optional[str] = None
    ) -> bool:
        """Update player position for tier recalculation."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Update session data
        player = session.get("player", {})
        player["x"] = x
        player["y"] = y

        # Update orchestrator
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'update_player_position'):
            from uuid import UUID
            settlement_uuid = UUID(settlement_id) if settlement_id else None
            orchestrator.update_player_position(x, y, settlement_uuid)

        return True

    async def start_interaction(self, session_id: str, agent_id: str) -> bool:
        """Mark an agent as in active interaction with the player."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'start_interaction'):
            from uuid import UUID
            orchestrator.start_interaction(UUID(agent_id))
            return True
        return False

    async def end_interaction(self, session_id: str, agent_id: str) -> bool:
        """End active interaction with an agent."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'end_interaction'):
            from uuid import UUID
            orchestrator.end_interaction(UUID(agent_id))
            return True
        return False

    # ==================== Collective Agent Management ====================

    async def register_settlement(
        self,
        session_id: str,
        settlement_id: str,
        name: str,
        settlement_type: str,
        x: int,
        y: int,
        population: int = 100,
        faction_id: Optional[str] = None
    ) -> bool:
        """Register a settlement with the collective manager."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'register_settlement'):
            from uuid import UUID
            orchestrator.register_settlement(
                settlement_id=UUID(settlement_id),
                name=name,
                settlement_type=settlement_type,
                x=x,
                y=y,
                population=population,
                faction_id=UUID(faction_id) if faction_id else None,
            )
            return True
        return False

    async def get_collective_stats(self, session_id: str) -> Dict[str, Any]:
        """Get collective agent statistics."""
        orchestrator = self._orchestrators.get(session_id)
        if orchestrator and hasattr(orchestrator, 'collective_manager'):
            return orchestrator.collective_manager.get_stats()
        return {}

    # ==================== Session Listing and Resume ====================

    async def list_sessions(
        self,
        world_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List available game sessions from the database.

        Args:
            world_id: Optional filter by world
            status: Optional filter by status ('active', 'ended')
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        db = self._get_db()
        if not db:
            # Return only in-memory sessions if no DB
            return [
                {
                    "id": session_id,
                    "world_id": s.get("world_id"),
                    "status": s.get("status"),
                    "current_tick": s.get("current_tick", 0),
                    "game_time": s.get("game_time"),
                    "player_name": s.get("player", {}).get("name", "Adventurer"),
                    "created_at": s.get("created_at"),
                }
                for session_id, s in self._sessions.items()
                if (world_id is None or s.get("world_id") == world_id) and
                   (status is None or s.get("status") == status)
            ][:limit]

        try:
            db_sessions = await db.list_sessions(world_id=world_id, status=status, limit=limit)
            sessions = []
            for db_session in db_sessions:
                session_id = db_session.get("session_id")
                # Try to get player info
                db_player = await db.get_player(session_id)
                sessions.append({
                    "id": session_id,
                    "world_id": db_session.get("world_id"),
                    "status": db_session.get("status"),
                    "current_tick": db_session.get("current_tick", 0),
                    "game_time": db_session.get("game_time"),
                    "player_name": db_player.get("name", "Adventurer") if db_player else "Adventurer",
                    "created_at": db_session.get("created_at"),
                })
            return sessions
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def resume_session(self, session_id: str) -> Optional[SessionResponse]:
        """Resume a previously saved session.

        Loads the session from the database and initializes it for play.

        Args:
            session_id: The ID of the session to resume

        Returns:
            SessionResponse if successful, None if session not found
        """
        # Check if already loaded
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if session.get("status") == "ended":
                logger.warning(f"Cannot resume ended session {session_id}")
                return None
            return await self.get_session(session_id)

        # Load from database
        session = await self._load_session_from_db(session_id)
        if not session:
            return None

        if session.get("status") == "ended":
            logger.warning(f"Cannot resume ended session {session_id}")
            return None

        # Store in memory
        self._sessions[session_id] = session

        # Initialize orchestrator
        await self._init_orchestrator(session_id, session)

        logger.info(f"Resumed session {session_id}")
        return await self.get_session(session_id)
