"""
Debug Service
Handles agent inspection, memory viewing, and simulation statistics.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from api.models.responses import (
    MemoryEntry,
    AgentThoughtsResponse,
    AgentPlansResponse,
    PlanEntry,
    RelationshipEntry,
    LLMCallEntry,
    SimulationStatsResponse,
    CognitivePhase
)

logger = logging.getLogger(__name__)


class DebugService:
    """
    Service for debugging agent behavior.

    Provides access to:
    - Agent memories
    - Cognitive state
    - Planning hierarchy
    - Relationships
    - LLM call history
    - Simulation statistics
    """

    def __init__(self):
        # Caches for debug data (in production, these would come from the actual agent system)
        self._memories: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._thoughts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._plans: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._relationships: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._llm_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._simulation_stats: Dict[str, Dict[str, Any]] = {}
        # Reference to game service for orchestrator access
        self._game_service = None
        # Debug tools references
        self._inspector = None
        self._llm_tracker = None

    def set_game_service(self, game_service):
        """Set the game service reference for accessing orchestrators."""
        self._game_service = game_service

    def _get_inspector(self):
        """Lazy load the agent inspector."""
        if self._inspector is None:
            try:
                from agents.debug.inspector import get_inspector
                self._inspector = get_inspector()
            except ImportError:
                logger.warning("Could not import agent inspector")
        return self._inspector

    def _get_llm_tracker(self):
        """Lazy load the LLM call tracker."""
        if self._llm_tracker is None:
            try:
                from agents.debug.llm_tracker import get_tracker
                self._llm_tracker = get_tracker()
            except ImportError:
                logger.warning("Could not import LLM tracker")
        return self._llm_tracker

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def get_memories(
        self,
        session_id: str,
        agent_id: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 50,
        include_embeddings: bool = False
    ) -> List[MemoryEntry]:
        """Get agent's memories with optional filters."""
        # Try inspector first
        inspector = self._get_inspector()
        if inspector:
            try:
                memories = await inspector.get_memories(
                    agent_id=UUID(agent_id),
                    memory_type=memory_type,
                    min_importance=min_importance,
                    limit=limit
                )
                return [
                    MemoryEntry(
                        id=m.get("id", ""),
                        content=m.get("description", ""),
                        memory_type=m.get("type", "observation"),
                        importance=m.get("importance", 0.5),
                        created_at=datetime.fromisoformat(m["created_at"]) if m.get("created_at") else datetime.utcnow(),
                        last_accessed=None,
                        access_count=m.get("access_count", 0),
                        embedding=None
                    )
                    for m in memories
                ]
            except Exception as e:
                logger.error(f"Inspector get_memories failed: {e}")

        # Fallback to cached data
        agent_memories = self._memories.get(session_id, {}).get(agent_id, [])

        # Filter by type
        if memory_type:
            agent_memories = [m for m in agent_memories if m.get("type") == memory_type]

        # Filter by importance
        agent_memories = [m for m in agent_memories if m.get("importance", 0) >= min_importance]

        # Sort by recency
        agent_memories.sort(key=lambda m: m.get("created_at", datetime.min), reverse=True)

        # Limit
        agent_memories = agent_memories[:limit]

        return [
            MemoryEntry(
                id=m.get("id", ""),
                content=m.get("content", ""),
                memory_type=m.get("type", "observation"),
                importance=m.get("importance", 0.5),
                created_at=m.get("created_at", datetime.utcnow()),
                last_accessed=m.get("last_accessed"),
                access_count=m.get("access_count", 0),
                embedding=m.get("embedding") if include_embeddings else None
            )
            for m in agent_memories
        ]

    async def search_memories(
        self,
        session_id: str,
        agent_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search agent's memories using semantic similarity."""
        # TODO: Integrate with actual embedding search
        # For now, return simple text matching
        agent_memories = self._memories.get(session_id, {}).get(agent_id, [])

        query_lower = query.lower()
        matches = [
            m for m in agent_memories
            if query_lower in m.get("content", "").lower()
        ]

        return [
            {
                "id": m.get("id"),
                "content": m.get("content"),
                "type": m.get("type"),
                "importance": m.get("importance"),
                "relevance_score": 0.8  # Placeholder
            }
            for m in matches[:limit]
        ]

    # =========================================================================
    # Cognitive State
    # =========================================================================

    async def get_thoughts(
        self,
        session_id: str,
        agent_id: str
    ) -> AgentThoughtsResponse:
        """Get agent's current cognitive state."""
        thoughts = self._thoughts.get(session_id, {}).get(agent_id, {})

        return AgentThoughtsResponse(
            agent_id=agent_id,
            phase=CognitivePhase(thoughts.get("phase", "idle")),
            current_thought=thoughts.get("current_thought"),
            current_perception=thoughts.get("current_perception"),
            retrieved_memories=thoughts.get("retrieved_memories", []),
            pending_action=thoughts.get("pending_action"),
            last_think_time=thoughts.get("last_think_time"),
            think_duration_ms=thoughts.get("think_duration_ms")
        )

    async def get_plans(
        self,
        session_id: str,
        agent_id: str
    ) -> AgentPlansResponse:
        """Get agent's planning hierarchy."""
        plans = self._plans.get(session_id, {}).get(agent_id, {})

        def to_plan_entry(p: Dict) -> PlanEntry:
            return PlanEntry(
                description=p.get("description", ""),
                start_time=p.get("start_time", "00:00"),
                duration_minutes=p.get("duration_minutes", 30),
                location=p.get("location"),
                status=p.get("status", "pending")
            )

        day_plan = [to_plan_entry(p) for p in plans.get("day_plan", [])]
        hour_plan = [to_plan_entry(p) for p in plans.get("hour_plan", [])]
        current = plans.get("current_action")

        return AgentPlansResponse(
            agent_id=agent_id,
            day_plan=day_plan,
            hour_plan=hour_plan,
            current_action=to_plan_entry(current) if current else None,
            day_progress=plans.get("day_progress", 0.0),
            hour_progress=plans.get("hour_progress", 0.0)
        )

    async def get_needs(
        self,
        session_id: str,
        agent_id: str
    ) -> Dict[str, float]:
        """Get agent's current needs state."""
        # TODO: Get from actual agent system
        return {
            "hunger": 0.7,
            "thirst": 0.8,
            "rest": 0.6,
            "warmth": 0.9,
            "safety": 0.85,
            "social": 0.5
        }

    # =========================================================================
    # Relationships
    # =========================================================================

    async def get_relationships(
        self,
        session_id: str,
        agent_id: str,
        min_familiarity: float = 0.0
    ) -> List[RelationshipEntry]:
        """Get agent's relationships with other agents."""
        agent_rels = self._relationships.get(session_id, {}).get(agent_id, {})

        relationships = []
        for other_id, rel in agent_rels.items():
            if rel.get("familiarity", 0) >= min_familiarity:
                relationships.append(RelationshipEntry(
                    other_id=other_id,
                    other_name=rel.get("other_name", "Unknown"),
                    familiarity=rel.get("familiarity", 0),
                    trust=rel.get("trust", 0.5),
                    affection=rel.get("affection", 0.5),
                    respect=rel.get("respect", 0.5),
                    relationship_type=rel.get("relationship_type"),
                    last_interaction=rel.get("last_interaction"),
                    interaction_count=rel.get("interaction_count", 0)
                ))

        return relationships

    async def get_relationship_detail(
        self,
        session_id: str,
        agent_id: str,
        other_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed relationship between two agents."""
        agent_rels = self._relationships.get(session_id, {}).get(agent_id, {})
        rel = agent_rels.get(other_id)

        if not rel:
            return None

        return {
            "agent_id": agent_id,
            "other_id": other_id,
            "other_name": rel.get("other_name"),
            "familiarity": rel.get("familiarity", 0),
            "trust": rel.get("trust", 0.5),
            "affection": rel.get("affection", 0.5),
            "respect": rel.get("respect", 0.5),
            "relationship_type": rel.get("relationship_type"),
            "shared_memories": rel.get("shared_memories", []),
            "interaction_history": rel.get("interaction_history", []),
            "last_interaction": rel.get("last_interaction")
        }

    # =========================================================================
    # LLM History
    # =========================================================================

    async def get_llm_history(
        self,
        session_id: str,
        agent_id: str,
        limit: int = 10
    ) -> List[LLMCallEntry]:
        """Get recent LLM calls made by this agent."""
        # Try LLM tracker first
        tracker = self._get_llm_tracker()
        if tracker:
            try:
                calls = tracker.get_agent_history(
                    agent_id=UUID(agent_id),
                    limit=limit
                )
                return [
                    LLMCallEntry(
                        id=c.call_id,
                        timestamp=c.timestamp,
                        purpose=c.purpose,
                        prompt_summary=c.prompt[:200] + "..." if len(c.prompt) > 200 else c.prompt,
                        response_summary=c.response[:200] + "..." if len(c.response) > 200 else c.response,
                        tokens_in=c.tokens_in,
                        tokens_out=c.tokens_out,
                        duration_ms=c.duration_ms,
                        model=c.model
                    )
                    for c in calls
                ]
            except Exception as e:
                logger.error(f"LLM tracker get_agent_history failed: {e}")

        # Fallback to cached data
        history = self._llm_history.get(session_id, {}).get(agent_id, [])

        # Sort by recency
        history.sort(key=lambda h: h.get("timestamp", datetime.min), reverse=True)

        return [
            LLMCallEntry(
                id=h.get("id", ""),
                timestamp=h.get("timestamp", datetime.utcnow()),
                purpose=h.get("purpose", "unknown"),
                prompt_summary=h.get("prompt_summary", ""),
                response_summary=h.get("response_summary", ""),
                tokens_in=h.get("tokens_in", 0),
                tokens_out=h.get("tokens_out", 0),
                duration_ms=h.get("duration_ms", 0),
                model=h.get("model", "unknown")
            )
            for h in history[:limit]
        ]

    async def get_llm_stats(self, session_id: str) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        tracker = self._get_llm_tracker()
        if tracker:
            return tracker.get_stats()
        return {
            "total_calls": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "avg_duration_ms": 0,
            "calls_by_purpose": {},
        }

    # =========================================================================
    # Simulation Statistics
    # =========================================================================

    async def get_simulation_stats(self, session_id: str) -> SimulationStatsResponse:
        """Get overall simulation statistics."""
        stats = self._simulation_stats.get(session_id, {})

        # Try to get live stats from game service
        if self._game_service:
            try:
                # Get tier distribution
                tier_dist = await self._game_service.get_tier_distribution(session_id)
                hierarchical_stats = await self._game_service.get_hierarchical_stats(session_id)
                collective_stats = await self._game_service.get_collective_stats(session_id)

                # Get session info
                session = self._game_service._sessions.get(session_id, {})
                current_tick = session.get("current_tick", 0)
                game_time = session.get("game_time", "Day 1, 08:00")
                total_agents = len(session.get("agents", {}))
                party_size = len(session.get("party_members", []))

                # Calculate active agents (PLAYER_PARTY + ACTIVE + NEARBY tiers)
                active_agents = (
                    tier_dist.get("PLAYER_PARTY", 0) +
                    tier_dist.get("ACTIVE", 0) +
                    tier_dist.get("NEARBY", 0)
                )

                return SimulationStatsResponse(
                    session_id=session_id,
                    current_tick=current_tick,
                    game_time=game_time,
                    total_agents=total_agents + party_size,
                    active_agents=active_agents + party_size,
                    agents_by_tier=tier_dist,
                    avg_tick_duration_ms=hierarchical_stats.get("scheduler_stats", {}).get("avg_tick_duration_ms", 0),
                    llm_calls_per_tick=hierarchical_stats.get("full_cycles", 0),
                    memory_operations_per_tick=0,
                    total_settlements=collective_stats.get("settlements", {}).get("count", 0),
                    total_factions=collective_stats.get("kingdoms", {}).get("count", 0),
                    active_events=0
                )
            except Exception as e:
                logger.error(f"Failed to get live simulation stats: {e}")

        # Fallback to cached stats
        return SimulationStatsResponse(
            session_id=session_id,
            current_tick=stats.get("current_tick", 0),
            game_time=stats.get("game_time", "Day 1, 08:00"),
            total_agents=stats.get("total_agents", 0),
            active_agents=stats.get("active_agents", 0),
            agents_by_tier=stats.get("agents_by_tier", {}),
            avg_tick_duration_ms=stats.get("avg_tick_duration_ms", 0),
            llm_calls_per_tick=stats.get("llm_calls_per_tick", 0),
            memory_operations_per_tick=stats.get("memory_operations_per_tick", 0),
            total_settlements=stats.get("total_settlements", 0),
            total_factions=stats.get("total_factions", 0),
            active_events=stats.get("active_events", 0)
        )

    async def get_agents_by_tier(self, session_id: str) -> Dict[str, List[str]]:
        """Get agents grouped by execution tier."""
        if self._game_service:
            try:
                tier_dist = await self._game_service.get_tier_distribution(session_id)
                # Get orchestrator for detailed agent lists
                orchestrator = self._game_service._orchestrators.get(session_id)
                if orchestrator and hasattr(orchestrator, 'hierarchical_scheduler'):
                    scheduler = orchestrator.hierarchical_scheduler
                    result = {}
                    for tier_name in tier_dist.keys():
                        # Get agent IDs in this tier
                        agent_ids = []
                        for agent_id, info in scheduler._agents.items():
                            if info.tier.name == tier_name:
                                agent_ids.append(str(agent_id))
                        result[tier_name] = agent_ids
                    return result
            except Exception as e:
                logger.error(f"Failed to get agents by tier: {e}")

        return {
            "PLAYER_PARTY": [],
            "ACTIVE": [],
            "NEARBY": [],
            "SAME_SETTLEMENT": [],
            "SAME_REGION": [],
            "BACKGROUND": [],
            "DORMANT": []
        }

    async def get_active_agents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of agents that executed in the last tick."""
        if self._game_service:
            try:
                orchestrator = self._game_service._orchestrators.get(session_id)
                if orchestrator and hasattr(orchestrator, 'hierarchical_scheduler'):
                    scheduler = orchestrator.hierarchical_scheduler
                    session = self._game_service._sessions.get(session_id, {})
                    current_tick = session.get("current_tick", 0)

                    active = []
                    for agent_id, info in scheduler._agents.items():
                        if info.last_executed_tick == current_tick:
                            active.append({
                                "agent_id": str(agent_id),
                                "tier": info.tier.name,
                                "executed_simplified": info.tier.value >= 4  # SAME_SETTLEMENT and above
                            })
                    return active
            except Exception as e:
                logger.error(f"Failed to get active agents: {e}")

        return []

    async def get_events(
        self,
        session_id: str,
        limit: int = 50,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent simulation events."""
        # TODO: Get from actual event log
        return []

    # =========================================================================
    # Collective Agents
    # =========================================================================

    async def get_collective_state(
        self,
        session_id: str,
        collective_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get state of a collective agent (settlement or kingdom)."""
        # TODO: Get from actual collective agent system
        return {
            "id": collective_id,
            "type": "settlement",
            "name": "Unknown Settlement",
            "population": 0,
            "aggregate_mood": 0.5,
            "resources": {},
            "recent_decisions": []
        }

    async def get_collective_members(
        self,
        session_id: str,
        collective_id: str
    ) -> List[Dict[str, Any]]:
        """Get all agents belonging to a collective."""
        # TODO: Get from actual collective agent system
        return []
