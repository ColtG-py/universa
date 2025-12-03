"""
Core Agent Tools
Standard tools available to all agents.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

from agents.tools.base import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolCategory,
)


# =============================================================================
# PERCEPTION TOOLS
# =============================================================================

observe_tool = Tool(
    name="observe",
    description="Observe the surroundings to gather information about the environment, nearby agents, and objects.",
    category=ToolCategory.PERCEPTION,
    parameters=[
        ToolParameter(
            name="focus",
            type="string",
            description="What to focus on: 'area' for general surroundings, 'agent' for a specific person, 'object' for a specific thing",
            required=False,
            enum=["area", "agent", "object"],
            default="area",
        ),
        ToolParameter(
            name="target",
            type="string",
            description="Name of the agent or object to focus on (required if focus is 'agent' or 'object')",
            required=False,
        ),
    ],
)


# =============================================================================
# MOVEMENT TOOLS
# =============================================================================

move_tool = Tool(
    name="move",
    description="Move to a different location. Can move to a named location, towards another agent, or in a direction.",
    category=ToolCategory.MOVEMENT,
    parameters=[
        ToolParameter(
            name="destination",
            type="string",
            description="Where to move: a location name (e.g., 'town square'), an agent's name, or a direction (north, south, east, west)",
            required=True,
        ),
        ToolParameter(
            name="speed",
            type="string",
            description="How fast to move: 'walk', 'run', or 'sneak'",
            required=False,
            enum=["walk", "run", "sneak"],
            default="walk",
        ),
    ],
    requires_location=True,
)


# =============================================================================
# SOCIAL TOOLS
# =============================================================================

speak_tool = Tool(
    name="speak",
    description="Say something to a nearby agent or announce to the area. Use for conversations, greetings, and communication.",
    category=ToolCategory.SOCIAL,
    parameters=[
        ToolParameter(
            name="message",
            type="string",
            description="What to say",
            required=True,
        ),
        ToolParameter(
            name="target",
            type="string",
            description="Name of agent to speak to. If omitted, speaks to the area.",
            required=False,
        ),
        ToolParameter(
            name="tone",
            type="string",
            description="Tone of voice: 'normal', 'whisper', 'shout', 'friendly', 'formal'",
            required=False,
            enum=["normal", "whisper", "shout", "friendly", "formal"],
            default="normal",
        ),
    ],
    requires_target=False,
)


# =============================================================================
# SKILL TOOLS
# =============================================================================

use_skill_tool = Tool(
    name="use_skill",
    description="Use a skill to perform an action. Skills include crafting, combat, magic, social abilities, and more.",
    category=ToolCategory.SKILL,
    parameters=[
        ToolParameter(
            name="skill_name",
            type="string",
            description="Name of the skill to use (e.g., 'blacksmithing', 'persuasion', 'fireball')",
            required=True,
        ),
        ToolParameter(
            name="target",
            type="string",
            description="Target of the skill (agent name, object, or location)",
            required=False,
        ),
        ToolParameter(
            name="parameters",
            type="object",
            description="Additional parameters specific to the skill",
            required=False,
        ),
    ],
)


# =============================================================================
# MEMORY TOOLS
# =============================================================================

recall_memory_tool = Tool(
    name="recall_memory",
    description="Search through memories to recall relevant information about a topic, person, place, or event.",
    category=ToolCategory.MEMORY,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="What to remember - a topic, person's name, place, or event",
            required=True,
        ),
        ToolParameter(
            name="time_range",
            type="string",
            description="Time range to search: 'recent' (last day), 'week', 'month', 'all'",
            required=False,
            enum=["recent", "week", "month", "all"],
            default="all",
        ),
        ToolParameter(
            name="count",
            type="integer",
            description="Maximum number of memories to recall",
            required=False,
            default=5,
        ),
    ],
)


# =============================================================================
# WORLD INTERACTION TOOLS
# =============================================================================

interact_tool = Tool(
    name="interact",
    description="Interact with an object in the environment. Pick up, use, open, examine, or manipulate objects.",
    category=ToolCategory.WORLD,
    parameters=[
        ToolParameter(
            name="action",
            type="string",
            description="What to do: 'pickup', 'drop', 'use', 'open', 'close', 'examine', 'give'",
            required=True,
            enum=["pickup", "drop", "use", "open", "close", "examine", "give"],
        ),
        ToolParameter(
            name="target",
            type="string",
            description="The object to interact with",
            required=True,
        ),
        ToolParameter(
            name="recipient",
            type="string",
            description="For 'give' action: who to give the item to",
            required=False,
        ),
    ],
)


rest_tool = Tool(
    name="rest",
    description="Rest to recover energy and satisfy needs. Can rest, sleep, eat, or drink.",
    category=ToolCategory.WORLD,
    parameters=[
        ToolParameter(
            name="activity",
            type="string",
            description="Type of rest: 'rest' (brief), 'sleep' (long), 'eat', 'drink'",
            required=True,
            enum=["rest", "sleep", "eat", "drink"],
        ),
        ToolParameter(
            name="duration_minutes",
            type="integer",
            description="How long to rest (for 'rest' and 'sleep')",
            required=False,
            default=15,
        ),
        ToolParameter(
            name="item",
            type="string",
            description="Specific food or drink to consume",
            required=False,
        ),
    ],
)


# =============================================================================
# TOOL HANDLERS
# =============================================================================

async def handle_observe(
    focus: str = "area",
    target: Optional[str] = None,
    **context
) -> ToolResult:
    """Handler for observe tool"""
    world = context.get("world")
    agent_id = context.get("agent_id")

    if not world:
        return ToolResult(
            success=False,
            output=None,
            error="No world interface available"
        )

    try:
        if focus == "area":
            # Get general area information
            location = await world.query_location(agent_id)
            nearby = await world.query_nearby_agents(agent_id, radius=10)

            output = {
                "location": location,
                "nearby_agents": nearby,
                "description": f"You observe the area around you.",
            }

        elif focus == "agent" and target:
            # Focus on specific agent
            agent_info = await world.query_agent(target)
            output = {
                "agent": agent_info,
                "description": f"You observe {target} closely.",
            }

        elif focus == "object" and target:
            # Focus on specific object
            object_info = await world.query_object(target)
            output = {
                "object": object_info,
                "description": f"You examine {target}.",
            }

        else:
            output = {
                "description": "You look around but don't see anything notable.",
            }

        return ToolResult(success=True, output=output)

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_move(
    destination: str,
    speed: str = "walk",
    **context
) -> ToolResult:
    """Handler for move tool"""
    world = context.get("world")
    agent_id = context.get("agent_id")

    if not world:
        return ToolResult(
            success=False,
            output=None,
            error="No world interface available"
        )

    try:
        # Calculate path and move
        path_result = await world.query_path(agent_id, destination)

        if not path_result.get("valid"):
            return ToolResult(
                success=False,
                output=None,
                error=f"Cannot reach {destination}: {path_result.get('reason', 'path blocked')}"
            )

        # Execute movement
        move_result = await world.move_agent(agent_id, destination, speed)

        return ToolResult(
            success=True,
            output={
                "new_location": destination,
                "distance": path_result.get("distance"),
                "description": f"You {speed} to {destination}.",
            },
            side_effects={
                "location_changed": True,
                "stamina_used": path_result.get("stamina_cost", 0),
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_speak(
    message: str,
    target: Optional[str] = None,
    tone: str = "normal",
    **context
) -> ToolResult:
    """Handler for speak tool"""
    dialogue_system = context.get("dialogue_system")
    agent_name = context.get("agent_name", "Someone")

    try:
        if target:
            description = f'{agent_name} says to {target}: "{message}"'
        else:
            description = f'{agent_name} says: "{message}"'

        return ToolResult(
            success=True,
            output={
                "spoken": True,
                "message": message,
                "target": target,
                "tone": tone,
                "description": description,
            },
            side_effects={
                "dialogue_initiated": target is not None,
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_use_skill(
    skill_name: str,
    target: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    **context
) -> ToolResult:
    """Handler for use_skill tool"""
    skill_system = context.get("skill_system")
    agent_id = context.get("agent_id")

    if not skill_system:
        # Fallback without skill system
        return ToolResult(
            success=True,
            output={
                "skill": skill_name,
                "target": target,
                "description": f"You attempt to use {skill_name}" + (f" on {target}" if target else ""),
            }
        )

    try:
        # Check if agent has the skill
        has_skill = await skill_system.agent_has_skill(agent_id, skill_name)

        if not has_skill:
            return ToolResult(
                success=False,
                output=None,
                error=f"You don't know the skill: {skill_name}"
            )

        # Execute skill
        result = await skill_system.execute_skill(
            agent_id=agent_id,
            skill_name=skill_name,
            target=target,
            parameters=parameters or {},
        )

        return ToolResult(
            success=result.get("success", True),
            output=result,
            side_effects={
                "xp_gained": result.get("xp_gained", 0),
                "stamina_used": result.get("stamina_cost", 0),
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_recall_memory(
    query: str,
    time_range: str = "all",
    count: int = 5,
    **context
) -> ToolResult:
    """Handler for recall_memory tool"""
    memory_stream = context.get("memory_stream")

    if not memory_stream:
        return ToolResult(
            success=False,
            output=None,
            error="No memory system available"
        )

    try:
        # Convert time range to hours
        hours_map = {
            "recent": 24,
            "week": 168,
            "month": 720,
            "all": None,
        }
        hours = hours_map.get(time_range)

        # Retrieve memories
        if hours:
            memories = await memory_stream.retrieve_recent(
                hours=hours,
                limit=count,
            )
            # Filter by relevance
            memories = await memory_stream.retrieve(
                query=query,
                limit=count,
            )
        else:
            memories = await memory_stream.retrieve(
                query=query,
                limit=count,
            )

        # Format memories
        memory_texts = [
            {
                "description": m.description,
                "time": m.created_at.isoformat() if hasattr(m, 'created_at') else None,
                "importance": m.importance,
            }
            for m in memories
        ]

        return ToolResult(
            success=True,
            output={
                "query": query,
                "memories": memory_texts,
                "count": len(memory_texts),
                "description": f"You recall {len(memory_texts)} memories about '{query}'.",
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_interact(
    action: str,
    target: str,
    recipient: Optional[str] = None,
    **context
) -> ToolResult:
    """Handler for interact tool"""
    world = context.get("world")
    agent_id = context.get("agent_id")

    try:
        description_map = {
            "pickup": f"You pick up the {target}.",
            "drop": f"You drop the {target}.",
            "use": f"You use the {target}.",
            "open": f"You open the {target}.",
            "close": f"You close the {target}.",
            "examine": f"You examine the {target} closely.",
            "give": f"You give the {target} to {recipient}." if recipient else f"You hold out the {target}.",
        }

        return ToolResult(
            success=True,
            output={
                "action": action,
                "target": target,
                "recipient": recipient,
                "description": description_map.get(action, f"You {action} the {target}."),
            },
            side_effects={
                "inventory_changed": action in ["pickup", "drop", "give"],
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


async def handle_rest(
    activity: str,
    duration_minutes: int = 15,
    item: Optional[str] = None,
    **context
) -> ToolResult:
    """Handler for rest tool"""

    try:
        needs_effect = {
            "rest": {"energy": 0.1},
            "sleep": {"energy": 0.5, "tiredness": -0.5},
            "eat": {"hunger": -0.3},
            "drink": {"thirst": -0.3},
        }

        description_map = {
            "rest": f"You rest for {duration_minutes} minutes.",
            "sleep": f"You sleep for {duration_minutes} minutes.",
            "eat": f"You eat {item if item else 'some food'}.",
            "drink": f"You drink {item if item else 'some water'}.",
        }

        return ToolResult(
            success=True,
            output={
                "activity": activity,
                "duration": duration_minutes,
                "item": item,
                "description": description_map.get(activity, f"You {activity}."),
            },
            side_effects={
                "needs_changed": needs_effect.get(activity, {}),
                "time_passed": duration_minutes,
            }
        )

    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e))


# Attach handlers to tools
observe_tool.handler = handle_observe
move_tool.handler = handle_move
speak_tool.handler = handle_speak
use_skill_tool.handler = handle_use_skill
recall_memory_tool.handler = handle_recall_memory
interact_tool.handler = handle_interact
rest_tool.handler = handle_rest


def get_core_tools() -> List[Tool]:
    """Get all core tools"""
    return [
        observe_tool,
        move_tool,
        speak_tool,
        use_skill_tool,
        recall_memory_tool,
        interact_tool,
        rest_tool,
    ]


def create_core_registry() -> "ToolRegistry":
    """Create a registry with all core tools"""
    from agents.tools.base import ToolRegistry

    registry = ToolRegistry()
    for tool in get_core_tools():
        registry.register(tool)
    return registry
