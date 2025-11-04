# Agent Simulation Layer - Part 3: Skill Architect

## Overview

The Skill Architect is a specialized agent responsible for creating new skills dynamically based on agent requests and world context. This prevents skill proliferation, maintains taxonomy quality, and ensures skills match the technological and cultural level of the world.

**Design Philosophy:** "Skills evolve with civilization - stone age tools for stone age people, not smartphones."

---

## Table of Contents

1. [Architect Architecture](#architect-architecture)
2. [Skill Request Queue](#skill-request-queue)
3. [World Context Analysis](#world-context-analysis)
4. [Skill Creation Workflow](#skill-creation-workflow)
5. [Duplicate Detection](#duplicate-detection)
6. [Taxonomy Placement](#taxonomy-placement)
7. [Technology Level System](#technology-level-system)

---

## Architect Architecture

### SkillArchitectAgent

```python
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict, List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum

class SkillRequestStatus(str, Enum):
    """Status of a skill request"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    APPROVED = "approved"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    CREATED = "created"

class SkillArchitectAgent:
    """
    Specialized agent for creating new skills
    
    Responsibilities:
    - Evaluate skill requests from agents
    - Analyze world context and tech level
    - Determine taxonomy placement
    - Detect duplicates
    - Create balanced skill definitions
    """
    
    def __init__(
        self,
        world_id: UUID,
        model_name: str = "llama3.2:3b"
    ):
        self.agent_id = UUID("00000000-0000-0000-0000-000000000000")  # Special ID
        self.world_id = world_id
        
        # LLM for reasoning
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3  # Lower temperature for consistency
        )
        
        # World context cache
        self.world_context: Optional[WorldContext] = None
        self.last_context_update: datetime = datetime.utcnow()
        
        # Build agent graph
        self.graph = self._build_architect_graph()
    
    def _build_architect_graph(self) -> StateGraph:
        """Build the skill creation workflow"""
        workflow = StateGraph(ArchitectState)
        
        # Workflow nodes
        workflow.add_node("load_world_context", self.load_world_context_node)
        workflow.add_node("analyze_request", self.analyze_request_node)
        workflow.add_node("check_duplicates", self.check_duplicates_node)
        workflow.add_node("determine_placement", self.determine_placement_node)
        workflow.add_node("design_skill", self.design_skill_node)
        workflow.add_node("validate_skill", self.validate_skill_node)
        workflow.add_node("register_skill", self.register_skill_node)
        
        # Workflow edges
        workflow.set_entry_point("load_world_context")
        workflow.add_edge("load_world_context", "analyze_request")
        workflow.add_edge("analyze_request", "check_duplicates")
        
        # Conditional: if duplicate found, reject
        workflow.add_conditional_edges(
            "check_duplicates",
            self.duplicate_decision,
            {
                "create": "determine_placement",
                "reject_duplicate": "register_rejection"
            }
        )
        
        workflow.add_edge("determine_placement", "design_skill")
        workflow.add_edge("design_skill", "validate_skill")
        
        # Conditional: if valid, register; if invalid, reject
        workflow.add_conditional_edges(
            "validate_skill",
            self.validation_decision,
            {
                "register": "register_skill",
                "reject_invalid": "register_rejection"
            }
        )
        
        workflow.add_edge("register_skill", END)
        workflow.add_edge("register_rejection", END)
        
        return workflow.compile()
    
    async def load_world_context_node(self, state: "ArchitectState") -> "ArchitectState":
        """Load current world context for decision making"""
        
        # Refresh context if stale (older than 1 day)
        if (not self.world_context or 
            (datetime.utcnow() - self.last_context_update).days > 1):
            
            self.world_context = await load_world_context(self.world_id)
            self.last_context_update = datetime.utcnow()
        
        state["world_context"] = self.world_context
        return state
    
    async def analyze_request_node(self, state: "ArchitectState") -> "ArchitectState":
        """Analyze the skill request for validity and need"""
        
        request = state["request"]
        world_ctx = state["world_context"]
        
        analysis_prompt = f"""
You are a Skill Architect analyzing a skill creation request.

Request Details:
- Requesting Agent: {request.agent_id}
- Action Description: {request.action_description}
- Context: {request.context}

World Context:
- Technology Level: {world_ctx.tech_level}
- Available Materials: {world_ctx.available_materials}
- Known Skills: {len(world_ctx.existing_skills)} skills
- Cultural Era: {world_ctx.cultural_era}

Analyze this request:
1. Is the requested action appropriate for the current world tech level?
2. What materials or tools would be required?
3. Are these materials/tools available in the world?
4. What would be the natural name for this skill?
5. What category does this skill belong to?

Provide your analysis in JSON format:
{{
    "appropriate": true/false,
    "reason": "explanation",
    "suggested_skill_name": "skill name",
    "suggested_category": "category",
    "required_materials": ["list"],
    "required_tools": ["list"],
    "tech_appropriate": true/false
}}
"""
        
        response = await self.llm.ainvoke(analysis_prompt)
        
        # Parse response
        try:
            analysis = json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            analysis = {
                "appropriate": False,
                "reason": "Failed to analyze request",
                "tech_appropriate": False
            }
        
        state["analysis"] = analysis
        
        # If not appropriate, mark for rejection
        if not analysis.get("appropriate") or not analysis.get("tech_appropriate"):
            state["status"] = SkillRequestStatus.REJECTED
            state["rejection_reason"] = analysis.get("reason", "Inappropriate for world context")
        else:
            state["status"] = SkillRequestStatus.ANALYZING
        
        return state
    
    async def check_duplicates_node(self, state: "ArchitectState") -> "ArchitectState":
        """Check if a similar skill already exists"""
        
        if state["status"] == SkillRequestStatus.REJECTED:
            return state
        
        analysis = state["analysis"]
        suggested_name = analysis.get("suggested_skill_name", "")
        suggested_category = analysis.get("suggested_category", "")
        
        # Get existing skills in category
        existing_skills = await get_skills_in_category(suggested_category)
        
        # Use LLM to check for semantic similarity
        duplicate_check_prompt = f"""
Requested skill: {suggested_name}
Description: {state["request"].action_description}

Existing skills in {suggested_category}:
{chr(10).join([f"- {s.name}: {s.description}" for s in existing_skills])}

Is the requested skill substantially similar to any existing skill?
If yes, which one? If no, respond with "NONE".

Format:
DUPLICATE: [skill_id] | NONE
Explanation: [why similar or why unique]
"""
        
        response = await self.llm.ainvoke(duplicate_check_prompt)
        content = response.content.strip()
        
        if content.startswith("DUPLICATE:") and "NONE" not in content:
            # Extract duplicate skill ID
            duplicate_id = content.split("|")[0].replace("DUPLICATE:", "").strip()
            state["status"] = SkillRequestStatus.DUPLICATE
            state["duplicate_skill_id"] = duplicate_id
            state["rejection_reason"] = content.split("Explanation:")[-1].strip()
        else:
            state["status"] = SkillRequestStatus.APPROVED
        
        return state
    
    async def determine_placement_node(self, state: "ArchitectState") -> "ArchitectState":
        """Determine optimal taxonomy placement"""
        
        analysis = state["analysis"]
        suggested_category = analysis.get("suggested_category", "")
        skill_name = analysis.get("suggested_skill_name", "")
        
        # Get full taxonomy for the category
        category_tree = skill_navigator.get_subcategories(suggested_category)
        
        placement_prompt = f"""
You are placing a new skill in the skill taxonomy.

New Skill: {skill_name}
Description: {state["request"].action_description}
Suggested Category: {suggested_category}

Current taxonomy structure in {suggested_category}:
{json.dumps(category_tree, indent=2)}

Determine the best placement:
1. Should it be a top-level skill in {suggested_category}?
2. Should it be under an existing subcategory? Which one?
3. Should it create a new subcategory?

Provide your decision in JSON:
{{
    "placement": "top-level" | "under-subcategory" | "new-subcategory",
    "parent_skill_id": "skill.id" (if under-subcategory),
    "new_subcategory_name": "name" (if new-subcategory),
    "skill_id": "full.skill.id.path",
    "rationale": "explanation"
}}
"""
        
        response = await self.llm.ainvoke(placement_prompt)
        
        try:
            placement = json.loads(response.content)
        except:
            # Default placement
            placement = {
                "placement": "top-level",
                "skill_id": f"{suggested_category}.{skill_name.lower().replace(' ', '_')}",
                "rationale": "Default placement at top level"
            }
        
        state["placement"] = placement
        return state
    
    async def design_skill_node(self, state: "ArchitectState") -> "ArchitectState":
        """Design complete skill definition"""
        
        analysis = state["analysis"]
        placement = state["placement"]
        world_ctx = state["world_context"]
        request = state["request"]
        
        design_prompt = f"""
You are designing a new skill for the simulation.

Skill Name: {analysis['suggested_skill_name']}
Description: {request.action_description}
Category: {analysis['suggested_category']}
Skill ID: {placement['skill_id']}
Parent Skill: {placement.get('parent_skill_id', 'none')}

World Context:
- Tech Level: {world_ctx.tech_level}
- Available Materials: {world_ctx.available_materials}

Design the complete skill definition:

1. **Governing Stats**: Which two stats (primary and secondary) govern this skill?
   Consider what physical/mental attributes are needed.
   Choose from: strength, dexterity, constitution, intelligence, wisdom, charisma

2. **Base Difficulty**: How hard is this skill? (0.0 = trivial, 1.0 = extremely hard)
   Consider:
   - Complexity of the action
   - Physical/mental demands
   - Risk of failure
   
3. **Required Tools**: What tools are needed? (based on available tech)

4. **Required Materials**: What materials are consumed? (based on availability)

5. **XP Rewards**:
   - XP per use
   - XP to level up

6. **Prerequisites**: Required skills or minimum stat levels

Provide design in JSON:
{{
    "skill_id": "{placement['skill_id']}",
    "name": "skill name",
    "description": "detailed description",
    "primary_stat": "stat_name",
    "secondary_stat": "stat_name",
    "base_difficulty": 0.5,
    "xp_per_use": 1.0,
    "xp_to_level": 100.0,
    "required_tools": ["tool1", "tool2"],
    "required_materials": ["material1"],
    "required_skills": {{"skill.id": 5}},
    "required_stats": {{"stat_name": 10}},
    "rationale": "why these values"
}}
"""
        
        response = await self.llm.ainvoke(design_prompt)
        
        try:
            skill_design = json.loads(response.content)
        except:
            # Minimal fallback
            skill_design = {
                "skill_id": placement["skill_id"],
                "name": analysis["suggested_skill_name"],
                "description": request.action_description,
                "primary_stat": "intelligence",
                "secondary_stat": "dexterity",
                "base_difficulty": 0.5,
                "xp_per_use": 1.0,
                "xp_to_level": 100.0
            }
        
        state["skill_design"] = skill_design
        return state
    
    async def validate_skill_node(self, state: "ArchitectState") -> "ArchitectState":
        """Validate the designed skill for balance and correctness"""
        
        skill_design = state["skill_design"]
        
        validation_errors = []
        
        # Validate stat names
        valid_stats = ["strength", "dexterity", "constitution", 
                      "intelligence", "wisdom", "charisma"]
        
        if skill_design.get("primary_stat") not in valid_stats:
            validation_errors.append(f"Invalid primary stat: {skill_design.get('primary_stat')}")
        
        if skill_design.get("secondary_stat") not in valid_stats:
            validation_errors.append(f"Invalid secondary stat: {skill_design.get('secondary_stat')}")
        
        # Validate difficulty range
        difficulty = skill_design.get("base_difficulty", 0.5)
        if not 0.0 <= difficulty <= 1.0:
            validation_errors.append(f"Difficulty out of range: {difficulty}")
        
        # Validate XP values are positive
        if skill_design.get("xp_per_use", 1.0) <= 0:
            validation_errors.append("XP per use must be positive")
        
        if skill_design.get("xp_to_level", 100.0) <= 0:
            validation_errors.append("XP to level must be positive")
        
        # Validate required skills exist
        for req_skill_id in skill_design.get("required_skills", {}).keys():
            if not skill_navigator.get_skill_definition(req_skill_id):
                validation_errors.append(f"Required skill not found: {req_skill_id}")
        
        if validation_errors:
            state["status"] = SkillRequestStatus.REJECTED
            state["rejection_reason"] = "; ".join(validation_errors)
        else:
            state["status"] = SkillRequestStatus.CREATED
        
        return state
    
    async def register_skill_node(self, state: "ArchitectState") -> "ArchitectState":
        """Register the new skill in the system"""
        
        skill_design = state["skill_design"]
        
        # Create SkillDefinition
        new_skill = SkillDefinition(
            skill_id=skill_design["skill_id"],
            name=skill_design["name"],
            description=skill_design["description"],
            category=SkillCategory(skill_design["skill_id"].split(".")[0]),
            parent_skill=state["placement"].get("parent_skill_id"),
            primary_stat=skill_design["primary_stat"],
            secondary_stat=skill_design["secondary_stat"],
            base_difficulty=skill_design["base_difficulty"],
            xp_per_use=skill_design.get("xp_per_use", 1.0),
            xp_to_level=skill_design.get("xp_to_level", 100.0),
            required_skills=skill_design.get("required_skills", {}),
            required_tools=skill_design.get("required_tools", []),
            required_materials=skill_design.get("required_materials", [])
        )
        
        # Save to database
        await save_skill_definition(new_skill, self.world_id)
        
        # Add to navigator cache
        skill_navigator.skill_cache[new_skill.skill_id] = new_skill
        
        # Update request status
        request_id = state["request"].request_id
        await update_skill_request_status(
            request_id,
            SkillRequestStatus.CREATED,
            created_skill_id=new_skill.skill_id
        )
        
        # Notify requesting agent
        await notify_agent(
            state["request"].agent_id,
            "skill_created",
            {
                "skill_id": new_skill.skill_id,
                "skill_name": new_skill.name,
                "description": new_skill.description
            }
        )
        
        state["created_skill"] = new_skill
        
        logger.info(f"Skill Architect created new skill: {new_skill.skill_id}")
        
        return state
    
    async def register_rejection_node(self, state: "ArchitectState") -> "ArchitectState":
        """Register skill request rejection"""
        
        request_id = state["request"].request_id
        status = state["status"]
        reason = state.get("rejection_reason", "Rejected")
        
        await update_skill_request_status(
            request_id,
            status,
            rejection_reason=reason
        )
        
        # Notify requesting agent
        await notify_agent(
            state["request"].agent_id,
            "skill_request_rejected",
            {
                "reason": reason,
                "status": status,
                "duplicate_skill": state.get("duplicate_skill_id")
            }
        )
        
        logger.info(f"Skill Architect rejected request: {reason}")
        
        return state
    
    def duplicate_decision(self, state: "ArchitectState") -> str:
        """Decide whether to create or reject based on duplicate check"""
        if state["status"] == SkillRequestStatus.DUPLICATE:
            return "reject_duplicate"
        return "create"
    
    def validation_decision(self, state: "ArchitectState") -> str:
        """Decide whether to register or reject based on validation"""
        if state["status"] == SkillRequestStatus.CREATED:
            return "register"
        return "reject_invalid"
    
    async def process_request(
        self,
        skill_request: "SkillCreationRequest"
    ) -> Dict[str, Any]:
        """
        Process a skill creation request through the full workflow
        """
        initial_state = {
            "request": skill_request,
            "world_context": None,
            "analysis": {},
            "placement": {},
            "skill_design": {},
            "status": SkillRequestStatus.PENDING,
            "rejection_reason": None,
            "duplicate_skill_id": None,
            "created_skill": None
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        return result
```

---

## Skill Request Queue

### Request Data Models

```python
class SkillCreationRequest(BaseModel):
    """Request for a new skill to be created"""
    
    request_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: UUID
    world_id: UUID
    
    # What the agent is trying to do
    action_description: str = Field(
        description="What the agent wants to do (e.g., 'shoot a deer with a bow')"
    )
    
    # Context about why this skill is needed
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context about the situation (location, available resources, urgency)"
    )
    
    # Request metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=5, ge=1, le=10)  # 10 = highest
    status: SkillRequestStatus = SkillRequestStatus.PENDING
    
    # Results (filled after processing)
    created_skill_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    processed_at: Optional[datetime] = None

class SkillRequestQueue:
    """
    Priority queue for skill creation requests
    
    Manages requests from all agents, processes them in order
    """
    
    def __init__(self, world_id: UUID):
        self.world_id = world_id
        self.queue: List[SkillCreationRequest] = []
        self.processing_lock = asyncio.Lock()
    
    async def submit_request(
        self,
        agent_id: UUID,
        action_description: str,
        context: Dict[str, Any],
        priority: int = 5
    ) -> UUID:
        """
        Submit a skill creation request
        
        Returns: request_id
        """
        request = SkillCreationRequest(
            agent_id=agent_id,
            world_id=self.world_id,
            action_description=action_description,
            context=context,
            priority=priority
        )
        
        # Save to database
        await save_skill_request(request)
        
        # Add to in-memory queue
        self.queue.append(request)
        
        # Sort by priority (highest first)
        self.queue.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Skill request submitted: {request.request_id} by {agent_id}")
        
        return request.request_id
    
    async def get_next_request(self) -> Optional[SkillCreationRequest]:
        """Get next pending request from queue"""
        async with self.processing_lock:
            pending_requests = [r for r in self.queue if r.status == SkillRequestStatus.PENDING]
            
            if not pending_requests:
                return None
            
            # Return highest priority pending request
            return pending_requests[0]
    
    async def process_queue(self, architect: SkillArchitectAgent):
        """Process all pending requests"""
        
        while True:
            request = await self.get_next_request()
            
            if not request:
                break
            
            # Mark as analyzing
            request.status = SkillRequestStatus.ANALYZING
            await update_skill_request_status(request.request_id, SkillRequestStatus.ANALYZING)
            
            # Process through architect
            try:
                result = await architect.process_request(request)
                
                # Update request with result
                request.status = result["status"]
                request.processed_at = datetime.utcnow()
                
                if result.get("created_skill"):
                    request.created_skill_id = result["created_skill"].skill_id
                
                if result.get("rejection_reason"):
                    request.rejection_reason = result["rejection_reason"]
                
                await save_skill_request(request)
                
            except Exception as e:
                logger.error(f"Failed to process skill request {request.request_id}: {e}")
                request.status = SkillRequestStatus.REJECTED
                request.rejection_reason = f"Processing error: {str(e)}"
                await save_skill_request(request)
            
            # Remove from queue
            self.queue.remove(request)

# Global queue per world
skill_queues: Dict[UUID, SkillRequestQueue] = {}

def get_skill_queue(world_id: UUID) -> SkillRequestQueue:
    """Get or create skill queue for a world"""
    if world_id not in skill_queues:
        skill_queues[world_id] = SkillRequestQueue(world_id)
    return skill_queues[world_id]
```

---

## World Context Analysis

### WorldContext Model

```python
class TechLevel(str, Enum):
    """Technology levels for world context"""
    STONE_AGE = "stone_age"              # Stone tools, fire
    BRONZE_AGE = "bronze_age"            # Bronze working, simple metalwork
    IRON_AGE = "iron_age"                # Iron tools, advanced metallurgy
    MEDIEVAL = "medieval"                # Steel, castles, guilds
    RENAISSANCE = "renaissance"          # Printing, gunpowder
    INDUSTRIAL = "industrial"            # Steam, factories
    MODERN = "modern"                    # Electronics, automobiles
    FUTURISTIC = "futuristic"            # Advanced tech

class CulturalEra(str, Enum):
    """Cultural development level"""
    TRIBAL = "tribal"                    # Small bands, oral tradition
    SETTLEMENT = "settlement"            # Villages, agriculture
    CIVILIZATION = "civilization"        # Cities, writing, government
    EMPIRE = "empire"                    # Large-scale organization

class WorldContext(BaseModel):
    """
    Current state of the world for skill creation decisions
    """
    world_id: UUID
    
    # Technology and materials
    tech_level: TechLevel
    available_materials: List[str]  # e.g., ["stone", "wood", "bronze"]
    available_tools: List[str]      # e.g., ["hammer", "chisel", "forge"]
    
    # Cultural context
    cultural_era: CulturalEra
    population_size: int
    num_settlements: int
    
    # Existing skills
    existing_skills: List[str]  # All skill IDs currently in use
    skill_categories_developed: List[SkillCategory]
    
    # Environmental
    climate_types: List[str]
    dominant_biomes: List[str]
    
    # Recent events that might affect skills
    recent_discoveries: List[str]  # e.g., ["iron_smelting", "pottery_wheel"]
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)

async def load_world_context(world_id: UUID) -> WorldContext:
    """
    Load comprehensive world context for skill creation
    """
    # Query world state from database
    world_data = await db.fetchrow("""
        SELECT * FROM worlds WHERE world_id = $1
    """, world_id)
    
    # Get technology level from world events
    tech_level = await infer_tech_level(world_id)
    
    # Get available materials from world resources
    available_materials = await get_discovered_materials(world_id)
    available_tools = await get_created_tools(world_id)
    
    # Get cultural development
    cultural_era = await infer_cultural_era(world_id)
    population = await get_total_population(world_id)
    num_settlements = await count_settlements(world_id)
    
    # Get existing skills
    existing_skills = await get_all_skill_ids_in_use(world_id)
    developed_categories = list(set([
        SkillCategory(sid.split(".")[0]) for sid in existing_skills
    ]))
    
    # Environmental data
    climate_types = await get_world_climate_types(world_id)
    dominant_biomes = await get_dominant_biomes(world_id)
    
    # Recent discoveries
    recent_discoveries = await get_recent_technology_events(world_id, days=30)
    
    context = WorldContext(
        world_id=world_id,
        tech_level=tech_level,
        available_materials=available_materials,
        available_tools=available_tools,
        cultural_era=cultural_era,
        population_size=population,
        num_settlements=num_settlements,
        existing_skills=existing_skills,
        skill_categories_developed=developed_categories,
        climate_types=climate_types,
        dominant_biomes=dominant_biomes,
        recent_discoveries=recent_discoveries
    )
    
    return context

async def infer_tech_level(world_id: UUID) -> TechLevel:
    """
    Infer technology level from world state
    
    Looks at:
    - Materials being used
    - Tools that have been created
    - Buildings constructed
    - Skills in use
    """
    materials = await get_discovered_materials(world_id)
    
    # Technology inference rules
    if "steel" in materials or "gunpowder" in materials:
        return TechLevel.RENAISSANCE
    elif "iron" in materials:
        return TechLevel.IRON_AGE
    elif "bronze" in materials or "copper" in materials:
        return TechLevel.BRONZE_AGE
    else:
        return TechLevel.STONE_AGE
```

---

## Skill Creation Workflow

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Agent Needs New Skill                  │
│                                                         │
│  "I want to shoot a deer but don't have a skill"      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Submit Skill Creation Request                │
│                                                         │
│  skill_queue.submit_request(                           │
│      agent_id=self.agent_id,                           │
│      action="shoot deer with bow",                     │
│      context={"location": (x,y), "target": "deer"}    │
│  )                                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Skill Request Queue                    │
│                                                         │
│  [Pending Requests sorted by priority]                 │
│  1. shoot deer with bow (priority 8)                   │
│  2. make pottery (priority 5)                          │
│  3. build wall (priority 7)                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Skill Architect Processes                  │
│                                                         │
│  Step 1: Load World Context                            │
│    - Tech Level: Stone Age                             │
│    - Materials: [stone, wood, leather, bone]          │
│    - Tools: [stone knife, fire]                        │
│    - No metal available                                │
│                                                         │
│  Step 2: Analyze Request                               │
│    Q: Is bow hunting appropriate for Stone Age?        │
│    A: Yes! Perfect for this tech level                 │
│    Q: What's needed?                                   │
│    A: Wood, string (sinew), feathers                   │
│    Q: Available?                                       │
│    A: Yes, all available                               │
│                                                         │
│  Step 3: Check Duplicates                              │
│    Search existing skills...                           │
│    - survival.hunting exists                           │
│    - survival.hunting.archery exists                   │
│    Similar but agent doesn't have it                   │
│    → Not a duplicate (agent-specific request)          │
│                                                         │
│  Step 4: Determine Placement                           │
│    Category: survival                                  │
│    Subcategory: survival.hunting                       │
│    Skill: survival.hunting.archery                     │
│    (Already exists! Use existing skill)                │
│    OR if new:                                          │
│    Skill: survival.hunting.bow_hunting                 │
│                                                         │
│  Step 5: Design Skill (if new)                         │
│    Primary Stat: dexterity (aiming)                    │
│    Secondary Stat: strength (draw bow)                 │
│    Difficulty: 0.6 (moderately hard)                   │
│    Required Tools: [bow, arrows]                       │
│    Required Skills: {survival.hunting: 2}              │
│    XP: 1.5 per use, 100 to level                      │
│                                                         │
│  Step 6: Validate                                      │
│    ✓ Stats valid                                       │
│    ✓ Difficulty in range                              │
│    ✓ Prerequisites exist                               │
│    ✓ Tech-appropriate                                  │
│                                                         │
│  Step 7: Register Skill                                │
│    Save to database                                    │
│    Add to taxonomy                                     │
│    Notify requesting agent                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Agent Receives Notification                  │
│                                                         │
│  "New skill available: survival.hunting.archery"       │
│  Agent can now use this skill                          │
└─────────────────────────────────────────────────────────┘
```

---

## Duplicate Detection

### Semantic Similarity Check

```python
class DuplicateDetector:
    """
    Detect if a requested skill is substantially similar
    to an existing skill
    """
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
    
    async def is_duplicate(
        self,
        requested_skill_name: str,
        requested_description: str,
        existing_skills: List[SkillDefinition]
    ) -> tuple[bool, Optional[str], str]:
        """
        Check if requested skill duplicates an existing skill
        
        Returns:
            (is_duplicate, duplicate_skill_id, explanation)
        """
        
        if not existing_skills:
            return False, None, "No existing skills to compare"
        
        # Create comparison prompt
        existing_list = "\n".join([
            f"- {skill.skill_id}: {skill.name} - {skill.description}"
            for skill in existing_skills
        ])
        
        prompt = f"""
You are checking if a requested skill is a duplicate of existing skills.

Requested Skill:
Name: {requested_skill_name}
Description: {requested_description}

Existing Skills:
{existing_list}

Is the requested skill substantially similar to any existing skill?

Two skills are considered duplicates if:
1. They accomplish the same goal through the same method
2. They use the same tools and techniques
3. One is a direct subset or superset of the other

Two skills are NOT duplicates if:
1. They use different methods (e.g., "hunt with bow" vs "hunt with spear")
2. They have different prerequisites or complexity
3. They serve different contexts (e.g., "survival hunting" vs "sport hunting")

Respond in format:
DECISION: DUPLICATE | UNIQUE
SIMILAR_TO: [skill_id] (if duplicate)
EXPLANATION: [why similar or why unique]
"""
        
        response = await self.llm.ainvoke(prompt)
        content = response.content.strip()
        
        # Parse response
        is_duplicate = "DUPLICATE" in content.split("\n")[0]
        
        similar_skill_id = None
        if "SIMILAR_TO:" in content:
            similar_skill_id = content.split("SIMILAR_TO:")[1].split("\n")[0].strip()
        
        explanation = content.split("EXPLANATION:")[1].strip() if "EXPLANATION:" in content else ""
        
        return is_duplicate, similar_skill_id, explanation
```

---

## Taxonomy Placement

### Intelligent Placement Logic

```python
class TaxonomyPlacer:
    """
    Determines optimal placement in skill taxonomy
    """
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
    
    async def determine_placement(
        self,
        skill_name: str,
        skill_description: str,
        suggested_category: str
    ) -> Dict[str, Any]:
        """
        Determine where in taxonomy to place the skill
        
        Returns: {
            "placement_type": "top-level" | "under-subcategory" | "new-subcategory",
            "parent_skill_id": "skill.id",
            "skill_id": "full.path",
            "rationale": "explanation"
        }
        """
        
        # Get existing structure
        category_tree = skill_navigator.get_subcategories(suggested_category)
        
        placement_prompt = f"""
You are an expert at organizing skills into a hierarchical taxonomy.

New Skill:
Name: {skill_name}
Description: {skill_description}
Category: {suggested_category}

Existing Structure in {suggested_category}:
{json.dumps(category_tree, indent=2)}

Determine the best placement:

Option 1: TOP-LEVEL in {suggested_category}
  Use when: Skill is broad and fundamental
  Example: "survival" → "foraging" (top-level)

Option 2: UNDER EXISTING SUBCATEGORY
  Use when: Skill is a specialization of existing subcategory
  Example: "combat.melee" → "swords" (under melee)

Option 3: CREATE NEW SUBCATEGORY
  Use when: Skill represents a new discipline not yet covered
  Example: Creating "combat.mounted" for horseback fighting

Provide decision in JSON:
{{
    "placement_type": "top-level" | "under-subcategory" | "new-subcategory",
    "parent_skill_id": "parent.skill.id" (if under-subcategory),
    "new_subcategory_name": "name" (if new-subcategory),
    "skill_id": "full.skill.id.path",
    "rationale": "detailed explanation"
}}
"""
        
        response = await self.llm.ainvoke(placement_prompt)
        
        try:
            placement = json.loads(response.content)
        except:
            # Fallback to safe default
            skill_id = f"{suggested_category}.{skill_name.lower().replace(' ', '_')}"
            placement = {
                "placement_type": "top-level",
                "skill_id": skill_id,
                "rationale": "Default top-level placement"
            }
        
        return placement
```

---

## Technology Level System

### Tech-Appropriate Skill Filtering

```python
class TechAppropriateness:
    """
    Ensures skills match world technology level
    """
    
    # Material availability by tech level
    MATERIAL_REQUIREMENTS = {
        TechLevel.STONE_AGE: {
            "allowed": ["stone", "wood", "bone", "leather", "sinew", "plant_fiber"],
            "tools": ["stone_knife", "wooden_club", "fire", "stone_axe"]
        },
        TechLevel.BRONZE_AGE: {
            "allowed": ["bronze", "copper", "tin", "clay", "wool"] + 
                      MATERIAL_REQUIREMENTS[TechLevel.STONE_AGE]["allowed"],
            "tools": ["bronze_tools", "pottery_wheel", "loom"]
        },
        TechLevel.IRON_AGE: {
            "allowed": ["iron", "steel"] + 
                      MATERIAL_REQUIREMENTS[TechLevel.BRONZE_AGE]["allowed"],
            "tools": ["iron_tools", "forge", "anvil", "bellows"]
        },
        # ... more levels
    }
    
    @staticmethod
    def is_tech_appropriate(
        requested_materials: List[str],
        requested_tools: List[str],
        world_tech_level: TechLevel
    ) -> tuple[bool, str]:
        """
        Check if requested materials/tools match tech level
        
        Returns: (is_appropriate, reason)
        """
        allowed = TechAppropriateness.MATERIAL_REQUIREMENTS[world_tech_level]
        
        # Check materials
        for material in requested_materials:
            if material not in allowed["allowed"]:
                return False, f"Material '{material}' not available at {world_tech_level}"
        
        # Check tools
        for tool in requested_tools:
            # Simplified check - could be more sophisticated
            if any(advanced in tool.lower() for advanced in ["steel", "iron", "bronze"]):
                if world_tech_level == TechLevel.STONE_AGE:
                    return False, f"Tool '{tool}' requires higher tech level"
        
        return True, "Technology appropriate"
    
    @staticmethod
    async def suggest_tech_appropriate_alternative(
        requested_skill: str,
        world_tech_level: TechLevel,
        llm: ChatOllama
    ) -> str:
        """
        Suggest a tech-appropriate alternative to an anachronistic skill
        
        Example: "shoot gun" → "shoot bow" for Stone Age
        """
        allowed_materials = TechAppropriateness.MATERIAL_REQUIREMENTS[world_tech_level]["allowed"]
        
        prompt = f"""
The agent requested: {requested_skill}

However, the world is currently at {world_tech_level} technology level.
Available materials: {', '.join(allowed_materials)}

What would be an appropriate alternative skill that accomplishes
a similar goal using available technology?

Respond with ONLY the alternative skill name, no explanation.
"""
        
        response = await llm.ainvoke(prompt)
        return response.content.strip()
```

---

## Summary

Part 3 establishes the Skill Architect system:

✅ **Centralized Skill Creation** - Single architect agent prevents chaos
✅ **Request Queue** - Priority-based processing of skill requests
✅ **World Context Awareness** - Skills match technology and culture level
✅ **Duplicate Detection** - Semantic similarity prevents redundant skills
✅ **Taxonomy Placement** - Intelligent placement in skill tree
✅ **Tech-Level Enforcement** - Stone Age gets spears, not guns
✅ **Material Validation** - Required materials must exist in world

**Next:** Part 4 will cover the Memory & Tools systems with MCP integration.