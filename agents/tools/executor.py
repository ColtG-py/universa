"""
Tool Executor
Executes tools based on LLM tool calls.
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass, field

from agents.tools.base import Tool, ToolRegistry, ToolResult, ToolCategory
from agents.llm.ollama_client import OllamaClient
from agents.memory.memory_stream import MemoryStream
from agents.world.interface import WorldInterface


@dataclass
class ToolCall:
    """A tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context passed to tool handlers"""
    agent_id: UUID
    agent_name: str
    world: Optional[WorldInterface] = None
    memory_stream: Optional[MemoryStream] = None
    skill_system: Optional[Any] = None
    dialogue_system: Optional[Any] = None
    current_time: datetime = field(default_factory=datetime.utcnow)
    additional: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool handlers"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "world": self.world,
            "memory_stream": self.memory_stream,
            "skill_system": self.skill_system,
            "dialogue_system": self.dialogue_system,
            "current_time": self.current_time,
            **self.additional,
        }


class ToolExecutor:
    """
    Executes tools from LLM responses.

    Handles:
    - Parsing tool calls from LLM output
    - Validating tool arguments
    - Executing tools with context
    - Recording tool usage in memory
    """

    def __init__(
        self,
        registry: ToolRegistry,
        ollama_client: Optional[OllamaClient] = None,
        context: Optional[ExecutionContext] = None,
    ):
        """
        Initialize executor.

        Args:
            registry: Tool registry
            ollama_client: LLM client for tool-calling inference
            context: Execution context for tools
        """
        self.registry = registry
        self.client = ollama_client
        self.context = context

        # Execution history
        self._history: List[Dict[str, Any]] = []

    def set_context(self, context: ExecutionContext) -> None:
        """Update execution context"""
        self.context = context

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        categories: Optional[List[ToolCategory]] = None,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate response with tool calling.

        Implements the ReAct pattern:
        1. LLM generates response (possibly with tool calls)
        2. Execute any tool calls
        3. Feed results back to LLM
        4. Repeat until no more tool calls or max iterations

        Args:
            prompt: User prompt
            system_prompt: System prompt
            categories: Filter tools by category
            max_iterations: Max tool-calling iterations

        Returns:
            Final response with tool results
        """
        if not self.client:
            return {
                "text": "",
                "error": "No LLM client available",
                "tool_calls": [],
                "tool_results": [],
            }

        # Get tools in OpenAI format
        tools = self.registry.to_openai_format(categories)

        if not tools:
            # No tools, just generate
            response = await self.client.generate(
                prompt=prompt,
                system=system_prompt,
            )
            return {
                "text": response.text,
                "tool_calls": [],
                "tool_results": [],
            }

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        all_tool_calls = []
        all_tool_results = []

        for iteration in range(max_iterations):
            # Generate with tools
            response = await self.client.chat_with_tools(
                messages=messages,
                tools=tools,
                temperature=0.7,
            )

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # No tool calls, we're done
                return {
                    "text": response.get("text", ""),
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                }

            # Execute tool calls
            for tc in tool_calls:
                parsed = self._parse_tool_call(tc)
                all_tool_calls.append(parsed)

                result = await self.execute_tool(parsed.name, parsed.arguments)
                all_tool_results.append(result)

                # Add tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "content": result.to_message(),
                })

        # Max iterations reached
        return {
            "text": response.get("text", ""),
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "max_iterations_reached": True,
        }

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        tool = self.registry.get(tool_name)

        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}"
            )

        # Validate required arguments
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Missing required argument: {param.name}"
                )

        # Add context to arguments
        context_dict = self.context.to_dict() if self.context else {}
        full_args = {**arguments, **context_dict}

        # Execute
        result = await tool.execute(**full_args)

        # Record in history
        self._history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Record in memory if available
        if self.context and self.context.memory_stream and result.success:
            description = result.output.get("description", f"Used {tool_name}")
            await self.context.memory_stream.add_observation(
                description=description,
                importance=0.3,
            )

        return result

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of results
        """
        results = []
        for tc in tool_calls:
            parsed = self._parse_tool_call(tc)
            result = await self.execute_tool(parsed.name, parsed.arguments)
            results.append(result)
        return results

    def _parse_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        """Parse a tool call from LLM output"""
        # Handle Ollama format
        function = tool_call.get("function", {})
        return ToolCall(
            name=function.get("name", tool_call.get("name", "")),
            arguments=function.get("arguments", tool_call.get("arguments", {})),
            call_id=tool_call.get("id"),
        )

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        if limit:
            return self._history[-limit:]
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear execution history"""
        self._history.clear()


class AgentToolExecutor(ToolExecutor):
    """
    Tool executor specialized for agent use.

    Adds:
    - Automatic context management
    - Skill validation
    - Need effects tracking
    """

    def __init__(
        self,
        agent_id: UUID,
        agent_name: str,
        registry: ToolRegistry,
        world: WorldInterface,
        memory_stream: MemoryStream,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize agent executor.

        Args:
            agent_id: Agent's UUID
            agent_name: Agent's name
            registry: Tool registry
            world: World interface
            memory_stream: Agent's memory stream
            ollama_client: LLM client
        """
        context = ExecutionContext(
            agent_id=agent_id,
            agent_name=agent_name,
            world=world,
            memory_stream=memory_stream,
        )

        super().__init__(
            registry=registry,
            ollama_client=ollama_client,
            context=context,
        )

        self.agent_id = agent_id
        self.agent_name = agent_name

    def set_skill_system(self, skill_system: Any) -> None:
        """Set skill system for skill validation"""
        if self.context:
            self.context.skill_system = skill_system

    def set_dialogue_system(self, dialogue_system: Any) -> None:
        """Set dialogue system for conversations"""
        if self.context:
            self.context.dialogue_system = dialogue_system

    async def decide_and_act(
        self,
        situation: str,
        available_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Decide what to do and execute it.

        Args:
            situation: Current situation description
            available_actions: Optional list of allowed actions

        Returns:
            Action result
        """
        # Build prompt
        prompt = f"""You are {self.agent_name}.

Current situation: {situation}

What do you do? Use the available tools to take action."""

        # Filter tools if actions specified
        categories = None
        if available_actions:
            # Map action names to categories (simplified)
            pass

        return await self.generate_with_tools(
            prompt=prompt,
            categories=categories,
        )
