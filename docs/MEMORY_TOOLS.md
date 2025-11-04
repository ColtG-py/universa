# Agent Simulation Layer - Part 4: Memory & Tools

## Overview

This document covers the memory systems and tool infrastructure that enable agents to learn, remember, and interact with the world. The three-layer memory architecture (episodic, semantic, procedural) allows agents to form experiences, accumulate knowledge, and develop skills, while the MCP-compliant tool system provides structured world interactions.

---

## Table of Contents

1. [Three-Layer Memory Architecture](#three-layer-memory-architecture)
2. [Episodic Memory](#episodic-memory)
3. [Semantic Memory](#semantic-memory)
4. [Procedural Memory](#procedural-memory)
5. [Tool Registry & MCP](#tool-registry--mcp)
6. [Core Tools](#core-tools)
7. [Memory Integration](#memory-integration)

---

## Three-Layer Memory Architecture

Based on cognitive science and the CoALA framework:

```python
from chromadb import Client as ChromaClient
from chromadb.config import Settings
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID

class AgentMemorySystem:
    """
    Unified memory system for agents
    
    Three layers:
    - Episodic: Personal experiences and events
    - Semantic: Facts and knowledge about the world
    - Procedural: How-to knowledge and learned behaviors
    """
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        
        # Initialize three memory types
        self.episodic = EpisodicMemory(agent_id)
        self.semantic = SemanticMemory(agent_id)
        self.procedural = ProceduralMemory(agent_id)
    
    async def consolidate_memories(self):
        """
        Periodic memory consolidation
        
        - Semantic facts from repeated episodic patterns
        - Procedural skills from successful episodic actions
        - Importance-based forgetting
        """
        # Identify patterns in episodic memory
        recent_episodes = await self.episodic.get_recent_episodes(days=7)
        
        # Extract repeated facts
        facts = await self._extract_facts_from_episodes(recent_episodes)
        for fact in facts:
            await self.semantic.store_fact(fact)
        
        # Identify successful procedures
        procedures = await self._extract_procedures_from_episodes(recent_episodes)
        for procedure in procedures:
            await self.procedural.store_procedure(
                procedure["name"],
                procedure["prompt"],
                procedure["success_rate"]
            )
    
    async def _extract_facts_from_episodes(
        self,
        episodes: List[Dict]
    ) -> List[str]:
        """Use LLM to extract facts from experiences"""
        # Implementation would use LLM to identify facts
        pass
    
    async def _extract_procedures_from_episodes(
        self,
        episodes: List[Dict]
    ) -> List[Dict]:
        """Identify successful procedures from experience"""
        pass
```

---

## Episodic Memory

Stores specific past experiences and events - the "what, when, where" of personal history.

```python
class EpisodicMemory:
    """
    Stores specific past experiences
    
    Like human autobiographical memory - remembers events
    with temporal and spatial context.
    """
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.chroma_client = ChromaClient(Settings(
            persist_directory=f"./memory/episodic/{agent_id}"
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"episodes_{agent_id}"
        )
    
    async def store_episode(
        self,
        context: List[str],
        actions: List[str],
        outcomes: List[Dict],
        skills_used: List[str],
        reflection: str,
        importance: float = 0.5
    ) -> UUID:
        """
        Store a new episodic memory
        
        Args:
            context: What was happening
            actions: What the agent did
            outcomes: What happened as a result
            skills_used: Which skills were used
            reflection: Agent's thoughts about the experience
            importance: How significant (0.0-1.0)
        """
        episode_id = uuid.uuid4()
        
        # Create narrative summary
        summary = f"""
Context: {' '.join(context)}
Actions taken: {' '.join(actions)}
Skills used: {', '.join(skills_used)}
Outcomes: {json.dumps(outcomes)}
Reflection: {reflection}
"""
        
        # Evaluate success
        success = self._evaluate_success(outcomes)
        
        # Store in vector database
        self.collection.add(
            documents=[summary],
            metadatas=[{
                "agent_id": str(self.agent_id),
                "timestamp": datetime.utcnow().isoformat(),
                "importance": importance,
                "actions": json.dumps(actions),
                "skills_used": json.dumps(skills_used),
                "success": success,
                "location": json.dumps(context)
            }],
            ids=[str(episode_id)]
        )
        
        # Also store in relational DB for querying
        await db.execute("""
            INSERT INTO episodic_memories 
            (episode_id, agent_id, summary, actions, outcomes, skills_used,
             reflection, importance, success, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, episode_id, self.agent_id, summary, 
           json.dumps(actions), json.dumps(outcomes), json.dumps(skills_used),
           reflection, importance, success, datetime.utcnow())
        
        return episode_id
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.3
    ) -> List[Dict]:
        """
        Search for relevant past experiences
        
        Uses semantic similarity to find related episodes
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"importance": {"$gte": min_importance}}
        )
        
        return self._format_results(results)
    
    async def get_recent_episodes(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent episodes chronologically"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return await db.fetch("""
            SELECT * FROM episodic_memories
            WHERE agent_id = $1 
            AND created_at > $2
            ORDER BY created_at DESC
            LIMIT $3
        """, self.agent_id, cutoff, limit)
    
    async def get_episodes_with_skill(
        self,
        skill_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Get episodes where a specific skill was used"""
        return await db.fetch("""
            SELECT * FROM episodic_memories
            WHERE agent_id = $1
            AND skills_used::jsonb ? $2
            ORDER BY created_at DESC
            LIMIT $3
        """, self.agent_id, skill_id, limit)
    
    def _evaluate_success(self, outcomes: List[Dict]) -> bool:
        """Heuristic to determine if episode was successful"""
        if not outcomes:
            return False
        
        error_count = sum(1 for o in outcomes if o.get("type") == "action_error")
        return error_count < len(outcomes) * 0.3
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results for agent consumption"""
        formatted = []
        
        if not results["documents"]:
            return formatted
        
        for i, doc in enumerate(results["documents"][0]):
            formatted.append({
                "episode_id": results["ids"][0][i],
                "summary": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        
        return formatted
```

---

## Semantic Memory

Stores facts and knowledge about the world - the "what" without the "when".

```python
class SemanticMemory:
    """
    Stores facts, concepts, and knowledge
    
    Like encyclopedic knowledge - knows things without
    remembering when/where it learned them.
    """
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.chroma_client = ChromaClient(Settings(
            persist_directory=f"./memory/semantic/{agent_id}"
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"facts_{agent_id}"
        )
    
    async def store_fact(
        self,
        fact: str,
        category: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "experience"
    ) -> UUID:
        """
        Store a semantic fact
        
        Args:
            fact: The factual statement
            category: Knowledge category (history, nature, etc.)
            confidence: How certain (0.0-1.0)
            source: Where fact came from (experience, told, etc.)
        """
        fact_id = uuid.uuid4()
        
        self.collection.add(
            documents=[fact],
            metadatas=[{
                "agent_id": str(self.agent_id),
                "category": category or "general",
                "confidence": confidence,
                "source": source,
                "created_at": datetime.utcnow().isoformat(),
                "access_count": 0
            }],
            ids=[str(fact_id)]
        )
        
        # Also store in PostgreSQL with pgvector
        await db.execute("""
            INSERT INTO semantic_facts
            (fact_id, agent_id, fact_text, category, 
             confidence, source, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, fact_id, self.agent_id, fact, category,
           confidence, source, datetime.utcnow())
        
        return fact_id
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None
    ) -> List[Dict]:
        """Search for relevant facts"""
        where_filter = {"agent_id": str(self.agent_id)}
        if category:
            where_filter["category"] = category
        
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter
        )
        
        # Update access count
        for result_id in results["ids"][0]:
            await self._increment_access_count(result_id)
        
        return self._format_results(results)
    
    async def update_fact(
        self,
        fact_id: UUID,
        new_confidence: Optional[float] = None,
        new_text: Optional[str] = None
    ):
        """Update an existing fact (learning/correction)"""
        if new_text:
            self.collection.update(
                ids=[str(fact_id)],
                documents=[new_text]
            )
        
        if new_confidence is not None:
            self.collection.update(
                ids=[str(fact_id)],
                metadatas=[{"confidence": new_confidence}]
            )
    
    async def _increment_access_count(self, fact_id: str):
        """Track fact usage for importance scoring"""
        await db.execute("""
            UPDATE semantic_facts
            SET access_count = access_count + 1
            WHERE fact_id = $1
        """, UUID(fact_id))
```

---

## Procedural Memory

Stores knowledge about how to perform tasks - learned skills and behaviors.

```python
class ProceduralMemory:
    """
    Stores how-to knowledge
    
    Like muscle memory - knows how to do things
    through practiced procedures.
    """
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.procedures: Dict[str, Dict] = {}
        asyncio.create_task(self.load_procedures())
    
    async def load_procedures(self):
        """Load stored procedures from database"""
        rows = await db.fetch("""
            SELECT procedure_name, procedure_prompt, 
                   success_rate, usage_count
            FROM procedural_memory
            WHERE agent_id = $1
        """, self.agent_id)
        
        for row in rows:
            self.procedures[row["procedure_name"]] = {
                "prompt": row["procedure_prompt"],
                "success_rate": row["success_rate"],
                "usage_count": row["usage_count"]
            }
    
    async def get_procedures(
        self,
        context: str,
        min_success_rate: float = 0.6
    ) -> List[Dict]:
        """
        Get relevant procedures for current context
        
        Returns procedures that:
        1. Are relevant to the context
        2. Have good success rates
        3. Have been used enough to be reliable
        """
        relevant = []
        
        # Simple keyword matching (could use embeddings)
        context_words = set(context.lower().split())
        
        for name, data in self.procedures.items():
            name_words = set(name.split("_"))
            
            if (data["success_rate"] >= min_success_rate and
                len(context_words & name_words) > 0):
                relevant.append({
                    "name": name,
                    "prompt": data["prompt"],
                    "success_rate": data["success_rate"],
                    "usage_count": data["usage_count"]
                })
        
        # Sort by success rate
        relevant.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return relevant
    
    async def store_procedure(
        self,
        procedure_name: str,
        procedure_prompt: str,
        initial_success_rate: float = 0.5
    ):
        """Store a new procedure"""
        self.procedures[procedure_name] = {
            "prompt": procedure_prompt,
            "success_rate": initial_success_rate,
            "usage_count": 0
        }
        
        await db.execute("""
            INSERT INTO procedural_memory
            (agent_id, procedure_name, procedure_prompt, 
             success_rate, usage_count, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (agent_id, procedure_name)
            DO UPDATE SET procedure_prompt = EXCLUDED.procedure_prompt
        """, self.agent_id, procedure_name, procedure_prompt,
           initial_success_rate, 0, datetime.utcnow())
    
    async def update_success_rate(
        self,
        procedure_name: str,
        success: bool
    ):
        """Update procedure success rate based on outcome"""
        if procedure_name not in self.procedures:
            return
        
        current = self.procedures[procedure_name]
        usage_count = current["usage_count"] + 1
        
        # Exponential moving average
        alpha = 0.1
        new_success_rate = (
            (1 - alpha) * current["success_rate"] +
            alpha * (1.0 if success else 0.0)
        )
        
        self.procedures[procedure_name]["success_rate"] = new_success_rate
        self.procedures[procedure_name]["usage_count"] = usage_count
        
        await db.execute("""
            UPDATE procedural_memory
            SET success_rate = $1, usage_count = $2
            WHERE agent_id = $3 AND procedure_name = $4
        """, new_success_rate, usage_count, self.agent_id, procedure_name)
```

---

## Tool Registry & MCP

MCP (Model Context Protocol) compliant tool system for structured world interactions.

```python
from pydantic import BaseModel, Field
from typing import Callable, Dict, Any, Optional, List
import json

class ToolParameter(BaseModel):
    """MCP-compliant tool parameter definition"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None

class ToolDefinition(BaseModel):
    """MCP-compliant tool definition"""
    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="What the tool does")
    parameters: List[ToolParameter] = Field(default_factory=list)
    
    def to_json_schema(self) -> Dict:
        """Convert to JSON Schema format for MCP"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class AgentToolRegistry:
    """MCP-compliant tool registry for agents"""
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}
        
        # Load core tools
        self._load_core_tools()
    
    def _load_core_tools(self):
        """Load essential tools all agents have"""
        # World observation
        self.register_tool(
            name="observe_location",
            function=self.observe_location,
            description="Observe the current location and nearby entities",
            parameters=[
                ToolParameter(
                    name="radius",
                    type="number",
                    description="Radius in cells to observe",
                    required=False,
                    default=5
                )
            ]
        )
        
        # Movement
        self.register_tool(
            name="move_to",
            function=self.move_to,
            description="Move to a specific location",
            parameters=[
                ToolParameter(name="x", type="number", description="X coordinate"),
                ToolParameter(name="y", type="number", description="Y coordinate")
            ]
        )
        
        # Skill usage
        self.register_tool(
            name="use_skill",
            function=self.use_skill,
            description="Use a skill to perform an action",
            parameters=[
                ToolParameter(
                    name="skill_id",
                    type="string",
                    description="Full skill ID (e.g., 'combat.melee.swords')"
                ),
                ToolParameter(
                    name="action_description",
                    type="string",
                    description="What you're trying to do"
                ),
                ToolParameter(
                    name="difficulty_modifier",
                    type="number",
                    description="Additional difficulty adjustment",
                    required=False,
                    default=0.0
                )
            ]
        )
        
        # Skill navigation
        self.register_tool(
            name="list_skill_categories",
            function=self.list_skill_categories,
            description="Get list of top-level skill categories",
            parameters=[]
        )
        
        self.register_tool(
            name="list_subcategories",
            function=self.list_subcategories,
            description="List subcategories within a category",
            parameters=[
                ToolParameter(
                    name="category_id",
                    type="string",
                    description="Category ID (e.g., 'combat', 'crafting')"
                )
            ]
        )
        
        self.register_tool(
            name="list_skills",
            function=self.list_skills,
            description="List specific skills in a subcategory",
            parameters=[
                ToolParameter(
                    name="subcategory_id",
                    type="string",
                    description="Subcategory ID (e.g., 'combat.melee')"
                )
            ]
        )
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: List[ToolParameter]
    ):
        """Register a tool with the agent"""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters
        )
        
        self.tools[name] = function
        self.tool_definitions[name] = tool_def
    
    async def get_available_tools(self) -> List[Dict]:
        """Get list of available tools in MCP format"""
        return [
            tool_def.to_json_schema()
            for tool_def in self.tool_definitions.values()
        ]
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        try:
            result = await self.tools[tool_name](**parameters)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Core tool implementations
    
    async def observe_location(self, radius: int = 5) -> Dict[str, Any]:
        """Observe environment around agent"""
        agent_state = await get_agent_state(self.agent_id)
        
        world_data = await query_world_region(
            world_id=agent_state.world_id,
            center_x=agent_state.position_x,
            center_y=agent_state.position_y,
            radius=radius
        )
        
        nearby_agents = await query_nearby_agents(
            world_id=agent_state.world_id,
            position_x=agent_state.position_x,
            position_y=agent_state.position_y,
            radius=radius,
            exclude_self=True
        )
        
        return {
            "terrain": world_data["terrain_summary"],
            "climate": world_data["climate_summary"],
            "resources": world_data["available_resources"],
            "nearby_agents": [
                {
                    "name": a.name,
                    "type": a.agent_type,
                    "distance": calculate_distance(
                        agent_state.position_x,
                        agent_state.position_y,
                        a.position_x,
                        a.position_y
                    )
                }
                for a in nearby_agents
            ]
        }
    
    async def move_to(self, x: int, y: int) -> Dict[str, Any]:
        """Move agent to new location"""
        agent_state = await get_agent_state(self.agent_id)
        
        distance = calculate_distance(
            agent_state.position_x,
            agent_state.position_y,
            x, y
        )
        
        # Check stamina
        stamina_cost = distance * 0.01
        if agent_state.stamina < stamina_cost:
            return {
                "success": False,
                "error": "Insufficient stamina"
            }
        
        # Update position
        await update_agent_position(self.agent_id, x, y)
        agent_state.stamina -= stamina_cost
        await save_agent_state(agent_state)
        
        return {
            "success": True,
            "new_position": {"x": x, "y": y},
            "stamina_used": stamina_cost
        }
    
    async def use_skill(
        self,
        skill_id: str,
        action_description: str,
        difficulty_modifier: float = 0.0
    ) -> Dict[str, Any]:
        """Use a skill to perform an action"""
        agent_state = await get_agent_state(self.agent_id)
        
        # Check if skill exists
        skill_def = skill_navigator.get_skill_definition(skill_id)
        if not skill_def:
            # Skill doesn't exist - submit request to Skill Architect
            queue = get_skill_queue(agent_state.world_id)
            request_id = await queue.submit_request(
                agent_id=self.agent_id,
                action_description=action_description,
                context={"skill_id": skill_id},
                priority=7  # High priority for active request
            )
            
            return {
                "success": False,
                "error": "Skill not found",
                "skill_request_submitted": True,
                "request_id": str(request_id)
            }
        
        # Use skill
        result = await stat_skill_progression.process_skill_use(
            agent_state,
            skill_id,
            success=True,  # Would actually roll for success
            difficulty_modifier=difficulty_modifier
        )
        
        return {
            "success": True,
            "skill_result": result
        }
    
    async def list_skill_categories(self) -> Dict[str, Any]:
        """Get top-level skill categories"""
        categories = skill_navigator.get_categories()
        return {"categories": categories}
    
    async def list_subcategories(self, category_id: str) -> Dict[str, Any]:
        """List subcategories"""
        subcats = skill_navigator.get_subcategories(category_id)
        return {"subcategories": subcats}
    
    async def list_skills(self, subcategory_id: str) -> Dict[str, Any]:
        """List skills in subcategory"""
        skills = skill_navigator.get_skills_in_subcategory(subcategory_id)
        return {"skills": skills}
```

---

## Core Tools

Essential tools available to all agents:

### World Interaction
- `observe_location` - See environment and nearby agents
- `move_to` - Navigate to location
- `gather_resource` - Collect materials
- `interact_with_agent` - Social interaction

### Skill System
- `list_skill_categories` - Browse top-level categories
- `list_subcategories` - Expand category
- `list_skills` - View specific skills
- `use_skill` - Perform skill-based action

### Memory
- `recall_similar_experience` - Search episodic memory
- `recall_fact` - Search semantic memory
- `recall_procedure` - Get procedural knowledge

---

## Memory Integration

How memory integrates with agent reasoning:

```python
async def recall_memory_node(self, state: AgentGraphState) -> AgentGraphState:
    """Retrieve relevant memories for current situation"""
    
    current_context = " ".join(state["observations"])
    
    # Retrieve semantic facts
    relevant_facts = await self.semantic_memory.search(
        query=current_context,
        limit=5
    )
    
    # Retrieve episodic memories
    relevant_experiences = await self.episodic_memory.search(
        query=current_context,
        limit=3
    )
    
    # Get procedural knowledge
    relevant_procedures = await self.procedural_memory.get_procedures(
        context=current_context
    )
    
    state["memory_context"] = {
        "facts": [f["fact_text"] for f in relevant_facts],
        "experiences": [e["summary"] for e in relevant_experiences],
        "procedures": [p["prompt"] for p in relevant_procedures]
    }
    
    return state
```

---

## Summary

Part 4 establishes memory and tool systems:

✅ **Three-Layer Memory** - Episodic, semantic, and procedural memory
✅ **MCP-Compliant Tools** - Structured world interactions
✅ **Core Tool Set** - Essential tools for all agents
✅ **Memory Integration** - Seamless memory recall in reasoning
✅ **Tool Registry** - Extensible tool system
✅ **Memory Consolidation** - Learning from experience

**Next:** Part 5 will provide complete examples, database schemas, and implementation guides.