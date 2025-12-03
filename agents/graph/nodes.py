"""
Agent Graph Nodes
Individual processing nodes for the agent execution graph.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable
from datetime import datetime
from uuid import UUID

from agents.graph.state import (
    AgentGraphState,
    AgentPhase,
    PerceptionData,
    ActionResult,
    ActionType,
)
from agents.memory.memory_stream import MemoryStream
from agents.reasoning.reflection import ReflectionSystem
from agents.reasoning.planning import PlanningSystem
from agents.reasoning.reaction import ReactionSystem, EnvironmentChange
from agents.world.interface import WorldInterface
from agents.llm.ollama_client import OllamaClient
from agents.config import REFLECTION_THRESHOLD


async def perceive_node(
    state: AgentGraphState,
    world: WorldInterface,
    memory_stream: MemoryStream,
) -> AgentGraphState:
    """
    Perception node: Observe the environment.

    Gathers information about:
    - Current location and environment
    - Nearby agents
    - Recent events
    - Available resources

    Creates observations in the memory stream.
    """
    state.mark_phase(AgentPhase.PERCEIVING)

    try:
        # Get agent's current position (default to 100,100 if not set)
        agent_x = getattr(state, 'position_x', 100)
        agent_y = getattr(state, 'position_y', 100)

        # Query location data from world
        location_data = world.query_location(agent_x, agent_y)

        # Build location info
        if location_data:
            location_name = location_data.settlement_type or location_data.biome_type or "unknown"
            if location_data.faction_name:
                location_name = f"{location_name} ({location_data.faction_name})"

            state.perception = PerceptionData(
                location_x=agent_x,
                location_y=agent_y,
                location=location_name,
                nearby_agents=[],  # Would be populated by query_nearby_agents
                nearby_objects=[],
                observations=[
                    f"The area is {location_data.biome_type or 'unremarkable'}.",
                ],
                events=[],
                environment={
                    "temperature": location_data.temperature_c,
                    "biome": location_data.biome_type,
                    "has_road": location_data.has_road,
                    "has_water": location_data.has_water,
                    "settlement": location_data.settlement_type,
                },
                timestamp=state.current_time,
            )

            # Add basic observation to memory
            obs_text = f"I am at {location_name}"
            if location_data.temperature_c:
                obs_text += f" where the temperature is {location_data.temperature_c:.1f}°C"
            obs_text += "."

            await memory_stream.add_observation(
                description=obs_text,
                location_x=agent_x,
                location_y=agent_y,
                importance=0.2,
            )
        else:
            # Default perception if no location data
            state.perception = PerceptionData(
                location_x=agent_x,
                location_y=agent_y,
                location="unknown",
                nearby_agents=[],
                nearby_objects=[],
                observations=["I observe my surroundings."],
                events=[],
                environment={},
                timestamp=state.current_time,
            )

    except Exception as e:
        state.error = f"Perception error: {str(e)}"

    return state


async def retrieve_node(
    state: AgentGraphState,
    memory_stream: MemoryStream,
) -> AgentGraphState:
    """
    Retrieval node: Fetch relevant memories.

    Uses the Stanford paper's retrieval function:
    score = α_recency × recency + α_importance × importance + α_relevance × relevance

    Retrieves memories relevant to current context.
    """
    state.mark_phase(AgentPhase.RETRIEVING)

    try:
        # Build query from current context
        query_parts = []

        if state.perception:
            query_parts.append(f"at {state.perception.location}")

            for agent in state.perception.nearby_agents[:2]:
                query_parts.append(agent.get("name", ""))

            for event in state.perception.events[:2]:
                query_parts.append(event)

        if state.current_action:
            query_parts.append(state.current_action)

        query = " ".join(query_parts) if query_parts else state.agent_name
        state.relevance_query = query

        # Retrieve relevant memories
        memories = await memory_stream.retrieve(
            query=query,
            limit=10,
            alpha_recency=1.0,
            alpha_importance=1.0,
            alpha_relevance=1.5,  # Weight relevance slightly higher
        )

        state.retrieved_memories = memories

        # Check if we should reflect
        state.should_reflect = memory_stream.should_reflect()
        state.importance_accumulator = memory_stream.get_importance_sum()

    except Exception as e:
        state.error = f"Retrieval error: {str(e)}"

    return state


async def reflect_node(
    state: AgentGraphState,
    reflection_system: ReflectionSystem,
) -> AgentGraphState:
    """
    Reflection node: Generate higher-level insights.

    Triggered when importance accumulator exceeds threshold.
    Creates reflections that synthesize recent observations.
    """
    state.mark_phase(AgentPhase.REFLECTING)

    if not state.should_reflect:
        return state

    try:
        # Generate reflections
        reflections = await reflection_system.reflect(force=False)

        state.new_reflections = reflections

        # Reset importance accumulator after reflection
        state.importance_accumulator = 0.0
        state.should_reflect = False

    except Exception as e:
        state.error = f"Reflection error: {str(e)}"

    return state


async def plan_node(
    state: AgentGraphState,
    planning_system: PlanningSystem,
) -> AgentGraphState:
    """
    Planning node: Decide what to do next.

    Uses hierarchical planning:
    - Day plan: Broad daily goals
    - Hour plan: Current hour's activity
    - Action plan: Specific 5-15 minute action
    """
    state.mark_phase(AgentPhase.PLANNING)

    try:
        # Check if we need a new day plan
        current_hour = state.current_time.hour

        if not planning_system.current_day_plan:
            await planning_system.create_day_plan(
                agent_summary=state.agent_summary,
                date=state.current_time,
            )

        # Check if we need a new hour plan
        if not planning_system.current_hour_plan or \
           (planning_system.current_hour_plan.end_time and
            state.current_time >= planning_system.current_hour_plan.end_time):
            await planning_system.create_hour_plan(
                agent_summary=state.agent_summary,
                current_time=state.current_time,
            )

        # Get or create action plan
        if not planning_system.current_action or \
           (planning_system.current_action.is_complete or
            planning_system.current_action.is_cancelled):
            action_plan = await planning_system.create_action_plan(
                agent_summary=state.agent_summary,
                current_time=state.current_time,
            )
            state.current_action = action_plan.action
            state.current_plan = action_plan
        else:
            state.current_action = planning_system.current_action.action

        # Build plan context for action node
        state.plan_context = planning_system.get_plan_context()

    except Exception as e:
        state.error = f"Planning error: {str(e)}"

    return state


async def react_node(
    state: AgentGraphState,
    reaction_system: ReactionSystem,
    planning_system: PlanningSystem,
) -> AgentGraphState:
    """
    Reaction node: Handle interrupts and unexpected events.

    Decides whether to continue current plan or react to event.
    """
    state.mark_phase(AgentPhase.REACTING)

    if not state.interrupt_event:
        return state

    try:
        # Create environment change from event
        event = EnvironmentChange(
            change_id=UUID(int=0),  # Placeholder
            description=state.interrupt_event,
            source_agent_name=None,  # Would extract from event
        )

        # Decide whether to react
        decision = await reaction_system.should_react(
            event=event,
            current_activity=state.current_action or "idle",
            agent_summary=state.agent_summary,
        )

        state.reaction_decision = {
            "should_react": decision.should_react,
            "reaction_type": decision.reaction_type.value,
            "priority": decision.priority,
            "reasoning": decision.reasoning,
            "suggested_action": decision.suggested_action,
        }

        # If we should react, update the plan
        if decision.should_react:
            new_action = await planning_system.replan_from_event(
                event_description=state.interrupt_event,
                agent_summary=state.agent_summary,
                current_time=state.current_time,
            )
            state.current_action = new_action.action
            state.current_plan = new_action

        # Clear interrupt after handling
        state.interrupt_event = None

    except Exception as e:
        state.error = f"Reaction error: {str(e)}"

    return state


async def act_node(
    state: AgentGraphState,
    world: WorldInterface,
    memory_stream: MemoryStream,
    action_executor: Optional[Callable] = None,
) -> AgentGraphState:
    """
    Action node: Execute the planned action.

    Translates high-level action descriptions into world interactions.
    Records results in memory.
    """
    state.mark_phase(AgentPhase.ACTING)

    if not state.current_action:
        state.error = "No action to execute"
        return state

    try:
        action_desc = state.current_action.lower()

        # Determine action type from description
        action_type = _classify_action(action_desc)

        # Build action request
        action_request = {
            "agent_id": state.agent_id,
            "action_type": action_type.value,
            "description": state.current_action,
            "context": state.plan_context,
            "timestamp": state.current_time,
        }

        state.pending_action = action_request

        # Execute action (either via executor or world interface)
        if action_executor:
            result = await action_executor(action_request)
        else:
            result = await _default_action_executor(action_request, world)

        state.last_action_result = ActionResult(
            action_type=action_type,
            success=result.get("success", True),
            description=result.get("description", state.current_action),
            effects=result.get("effects", {}),
        )

        # Record action in memory
        await memory_stream.add_observation(
            description=f"I {state.current_action}",
            importance=0.4 if state.last_action_result.success else 0.5,
            location_x=state.perception.location_x if state.perception else None,
            location_y=state.perception.location_y if state.perception else None,
        )

    except Exception as e:
        state.error = f"Action error: {str(e)}"
        state.last_action_result = ActionResult(
            action_type=ActionType.WAIT,
            success=False,
            description=f"Failed: {str(e)}",
        )

    return state


def _classify_action(action_desc: str) -> ActionType:
    """Classify action description into action type"""
    action_lower = action_desc.lower()

    if any(w in action_lower for w in ["walk", "go", "travel", "move", "head"]):
        return ActionType.MOVE
    elif any(w in action_lower for w in ["talk", "speak", "say", "ask", "tell", "greet"]):
        return ActionType.SPEAK
    elif any(w in action_lower for w in ["use", "craft", "make", "build", "forge"]):
        return ActionType.USE_SKILL
    elif any(w in action_lower for w in ["look", "observe", "watch", "examine"]):
        return ActionType.OBSERVE
    elif any(w in action_lower for w in ["rest", "sleep", "relax", "eat", "drink"]):
        return ActionType.REST
    elif any(w in action_lower for w in ["interact", "pick", "open", "touch"]):
        return ActionType.INTERACT
    else:
        return ActionType.WAIT


async def _default_action_executor(
    action_request: Dict[str, Any],
    world: WorldInterface,
) -> Dict[str, Any]:
    """Default action executor using world interface"""
    action_type = action_request.get("action_type")

    if action_type == ActionType.MOVE.value:
        # Would call world.move_agent()
        return {
            "success": True,
            "description": action_request.get("description"),
            "effects": {"moved": True},
        }

    elif action_type == ActionType.SPEAK.value:
        # Would initiate dialogue
        return {
            "success": True,
            "description": action_request.get("description"),
            "effects": {"spoke": True},
        }

    else:
        # Generic success for other actions
        return {
            "success": True,
            "description": action_request.get("description"),
            "effects": {},
        }


# Node routing functions

def should_reflect(state: AgentGraphState) -> bool:
    """Determine if agent should enter reflection node"""
    return state.should_reflect and state.importance_accumulator >= REFLECTION_THRESHOLD


def should_react(state: AgentGraphState) -> bool:
    """Determine if agent should enter reaction node"""
    return state.interrupt_event is not None


def get_next_node(state: AgentGraphState) -> str:
    """
    Determine the next node based on current state.

    Returns node name for graph routing.
    """
    if state.error:
        return "error_handler"

    if state.phase == AgentPhase.PERCEIVING:
        return "retrieve"

    elif state.phase == AgentPhase.RETRIEVING:
        if should_reflect(state):
            return "reflect"
        elif should_react(state):
            return "react"
        else:
            return "plan"

    elif state.phase == AgentPhase.REFLECTING:
        if should_react(state):
            return "react"
        return "plan"

    elif state.phase == AgentPhase.REACTING:
        return "plan"

    elif state.phase == AgentPhase.PLANNING:
        return "act"

    elif state.phase == AgentPhase.ACTING:
        return "end"

    return "end"
