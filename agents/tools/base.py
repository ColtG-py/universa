"""
Tool Base Classes
Foundation for MCP-style tool definitions.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
from enum import Enum
import json


class ToolCategory(str, Enum):
    """Categories of tools"""
    PERCEPTION = "perception"      # Observing the world
    MOVEMENT = "movement"          # Moving through the world
    SOCIAL = "social"              # Interacting with others
    SKILL = "skill"                # Using abilities
    MEMORY = "memory"              # Recalling information
    WORLD = "world"                # Interacting with objects
    SYSTEM = "system"              # Meta/system operations


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """Result of executing a tool"""
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    side_effects: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "side_effects": self.side_effects,
        }

    def to_message(self) -> str:
        """Convert to message for LLM"""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output)
        else:
            return f"Error: {self.error}"


@dataclass
class Tool:
    """
    Tool definition in MCP/OpenAI function calling format.

    Compatible with Qwen3's tool calling format.
    """
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., Awaitable[ToolResult]]] = None

    # Execution constraints
    cooldown_seconds: float = 0.0
    requires_location: bool = False
    requires_target: bool = False

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        This format is compatible with Ollama's tool calling API.
        """
        # Build properties and required list
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        if not self.handler:
            return ToolResult(
                success=False,
                output=None,
                error=f"No handler defined for tool: {self.name}"
            )

        start_time = datetime.utcnow()

        try:
            result = await self.handler(**kwargs)
            result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )


class ToolRegistry:
    """
    Registry of available tools.

    Manages tool registration, lookup, and formatting for LLM.
    """

    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, Tool] = {}
        self._by_category: Dict[ToolCategory, List[Tool]] = {
            cat: [] for cat in ToolCategory
        }

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        self._by_category[tool.category].append(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool"""
        tool = self._tools.pop(name, None)
        if tool:
            self._by_category[tool.category].remove(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category"""
        return self._by_category.get(category, [])

    def get_all(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def to_openai_format(
        self,
        categories: Optional[List[ToolCategory]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert all tools to OpenAI format for LLM.

        Args:
            categories: Optional filter by categories

        Returns:
            List of tool definitions in OpenAI format
        """
        tools = []

        for tool in self._tools.values():
            if categories is None or tool.category in categories:
                tools.append(tool.to_openai_format())

        return tools

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(p.name for p in tool.parameters)
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)


def create_tool(
    name: str,
    description: str,
    category: ToolCategory,
    parameters: Optional[List[Dict[str, Any]]] = None,
    handler: Optional[Callable] = None,
) -> Tool:
    """
    Factory function to create a tool.

    Args:
        name: Tool name
        description: What the tool does
        category: Tool category
        parameters: List of parameter definitions
        handler: Async function to execute

    Returns:
        Configured Tool
    """
    params = []
    if parameters:
        for p in parameters:
            params.append(ToolParameter(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                enum=p.get("enum"),
                default=p.get("default"),
            ))

    return Tool(
        name=name,
        description=description,
        category=category,
        parameters=params,
        handler=handler,
    )
