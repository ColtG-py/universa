"""
Skill Architect
LLM-powered system for creating and validating new skills.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

from agents.skills.taxonomy import Skill, SkillTree, SkillCategory
from agents.llm.ollama_client import OllamaClient


class SkillRequestStatus(str, Enum):
    """Status of a skill creation request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class SkillRequest:
    """Request to create a new skill"""
    request_id: UUID = field(default_factory=uuid4)
    requester_agent_id: Optional[UUID] = None
    proposed_name: str = ""
    proposed_description: str = ""
    proposed_category: Optional[SkillCategory] = None
    proposed_parent: Optional[str] = None  # Parent skill name
    reasoning: str = ""  # Why this skill should exist
    status: SkillRequestStatus = SkillRequestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewer_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": str(self.request_id),
            "requester_agent_id": str(self.requester_agent_id) if self.requester_agent_id else None,
            "proposed_name": self.proposed_name,
            "proposed_description": self.proposed_description,
            "proposed_category": self.proposed_category.value if self.proposed_category else None,
            "proposed_parent": self.proposed_parent,
            "reasoning": self.reasoning,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "reviewer_notes": self.reviewer_notes,
        }


class SkillArchitect:
    """
    LLM-powered skill creation and validation system.

    Agents can propose new skills when they:
    - Try to do something not covered by existing skills
    - Develop unique techniques through experience
    - Combine existing skills in novel ways

    The architect validates proposals using LLM:
    - Checks for duplicates
    - Validates hierarchy placement
    - Ensures balanced stat requirements
    - Generates proper skill definition
    """

    VALIDATION_PROMPT = """You are a skill system architect for a fantasy RPG.

Existing skill categories: {categories}

Proposed new skill:
- Name: {name}
- Description: {description}
- Category: {category}
- Parent skill: {parent}
- Reasoning: {reasoning}

Similar existing skills:
{similar_skills}

Evaluate this skill proposal:
1. Is this skill distinct from existing ones? (not a duplicate)
2. Does it fit logically under the proposed parent?
3. Is the category appropriate?
4. Is it balanced (not too powerful or useless)?

Respond in this format:
APPROVED: [yes/no]
REASON: [brief explanation]
SUGGESTED_CHANGES: [any improvements, or "none"]
STAT_PRIMARY: [strength/dexterity/constitution/intelligence/wisdom/charisma]
STAT_SECONDARY: [stat or "none"]
"""

    SKILL_GENERATION_PROMPT = """You are designing a skill for a fantasy RPG.

Create a skill with:
- Name: {name}
- Category: {category}
- Parent: {parent}

Requirements:
- Must be balanced and fit the fantasy setting
- Should have clear use cases
- Must specify stat requirements

Provide:
DESCRIPTION: [1-2 sentence description]
PRIMARY_STAT: [main stat]
SECONDARY_STAT: [secondary stat or "none"]
STAT_REQUIREMENTS: [stat:value pairs, e.g., "strength:12,dexterity:10" or "none"]
IS_ACTIVE: [true/false - can be used actively]
IS_PASSIVE: [true/false - provides passive bonuses]
COOLDOWN: [seconds, 0 for no cooldown]
"""

    def __init__(
        self,
        skill_tree: SkillTree,
        ollama_client: Optional[OllamaClient] = None,
        auto_approve: bool = False,
    ):
        """
        Initialize skill architect.

        Args:
            skill_tree: The skill tree to extend
            ollama_client: LLM client for validation
            auto_approve: Auto-approve valid skills (for testing)
        """
        self.skill_tree = skill_tree
        self.client = ollama_client
        self.auto_approve = auto_approve

        # Request queue
        self._pending_requests: Dict[UUID, SkillRequest] = {}
        self._completed_requests: List[SkillRequest] = []

        # Track skill lineage
        self._skill_creators: Dict[UUID, UUID] = {}  # skill_id -> agent_id

    async def propose_skill(
        self,
        name: str,
        description: str,
        category: SkillCategory,
        parent_name: Optional[str] = None,
        reasoning: str = "",
        requester_id: Optional[UUID] = None,
    ) -> SkillRequest:
        """
        Propose a new skill.

        Args:
            name: Proposed skill name
            description: What the skill does
            category: Skill category
            parent_name: Parent skill name
            reasoning: Why this skill should exist
            requester_id: Agent proposing the skill

        Returns:
            SkillRequest with status
        """
        request = SkillRequest(
            requester_agent_id=requester_id,
            proposed_name=name,
            proposed_description=description,
            proposed_category=category,
            proposed_parent=parent_name,
            reasoning=reasoning,
        )

        # Check for obvious duplicates
        existing = self.skill_tree.get_by_name(name)
        if existing:
            request.status = SkillRequestStatus.REJECTED
            request.reviewer_notes = f"Skill '{name}' already exists."
            self._completed_requests.append(request)
            return request

        # Validate with LLM if available
        if self.client:
            is_valid, notes, suggestions = await self._validate_with_llm(request)

            if is_valid:
                if self.auto_approve:
                    request.status = SkillRequestStatus.APPROVED
                    request.reviewer_notes = notes
                    # Create the skill
                    await self._create_skill_from_request(request, suggestions)
                else:
                    request.status = SkillRequestStatus.PENDING
                    request.reviewer_notes = notes
                    self._pending_requests[request.request_id] = request
            else:
                request.status = SkillRequestStatus.REJECTED
                request.reviewer_notes = notes

        else:
            # No LLM - use heuristic validation
            is_valid, notes = self._validate_heuristic(request)
            if is_valid:
                request.status = SkillRequestStatus.PENDING if not self.auto_approve else SkillRequestStatus.APPROVED
            else:
                request.status = SkillRequestStatus.REJECTED
            request.reviewer_notes = notes

            if request.status == SkillRequestStatus.APPROVED:
                await self._create_skill_from_request(request, {})

        request.reviewed_at = datetime.utcnow()
        self._completed_requests.append(request)
        return request

    async def approve_request(
        self,
        request_id: UUID,
        notes: str = "",
    ) -> Optional[Skill]:
        """
        Approve a pending skill request.

        Args:
            request_id: Request to approve
            notes: Reviewer notes

        Returns:
            Created skill if successful
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return None

        request.status = SkillRequestStatus.APPROVED
        request.reviewer_notes = notes
        request.reviewed_at = datetime.utcnow()

        # Create the skill
        skill = await self._create_skill_from_request(request, {})

        # Move to completed
        del self._pending_requests[request_id]
        self._completed_requests.append(request)

        return skill

    async def reject_request(
        self,
        request_id: UUID,
        reason: str,
    ) -> bool:
        """
        Reject a pending skill request.

        Args:
            request_id: Request to reject
            reason: Rejection reason

        Returns:
            True if rejected
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return False

        request.status = SkillRequestStatus.REJECTED
        request.reviewer_notes = reason
        request.reviewed_at = datetime.utcnow()

        del self._pending_requests[request_id]
        self._completed_requests.append(request)

        return True

    def get_pending_requests(self) -> List[SkillRequest]:
        """Get all pending requests"""
        return list(self._pending_requests.values())

    def get_skill_creator(self, skill_id: UUID) -> Optional[UUID]:
        """Get the agent who created a skill"""
        return self._skill_creators.get(skill_id)

    async def generate_skill_from_action(
        self,
        action_description: str,
        agent_id: UUID,
    ) -> Optional[SkillRequest]:
        """
        Generate a skill proposal from an agent's action.

        Called when an agent does something novel that could become a skill.

        Args:
            action_description: What the agent did
            agent_id: Agent who performed the action

        Returns:
            Skill request if a new skill was proposed
        """
        if not self.client:
            return None

        prompt = f"""An agent performed this action: "{action_description}"

Could this become a learnable skill? If so, what would it be called and categorized as?

Respond with:
IS_SKILL: [yes/no]
NAME: [skill name if yes]
CATEGORY: [combat/magic/crafting/gathering/social/knowledge/physical/trade]
PARENT: [parent skill name or "none"]
DESCRIPTION: [brief description]
"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200,
            )

            # Parse response
            lines = response.text.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip().upper()] = value.strip()

            if data.get("IS_SKILL", "").lower() != "yes":
                return None

            # Map category string to enum
            category_map = {
                "combat": SkillCategory.COMBAT,
                "magic": SkillCategory.MAGIC,
                "crafting": SkillCategory.CRAFTING,
                "gathering": SkillCategory.GATHERING,
                "social": SkillCategory.SOCIAL,
                "knowledge": SkillCategory.KNOWLEDGE,
                "physical": SkillCategory.PHYSICAL,
                "trade": SkillCategory.TRADE,
            }
            category = category_map.get(
                data.get("CATEGORY", "").lower(),
                SkillCategory.PHYSICAL
            )

            return await self.propose_skill(
                name=data.get("NAME", "Unknown Skill"),
                description=data.get("DESCRIPTION", action_description),
                category=category,
                parent_name=data.get("PARENT") if data.get("PARENT", "").lower() != "none" else None,
                reasoning=f"Developed through action: {action_description}",
                requester_id=agent_id,
            )

        except Exception:
            return None

    async def _validate_with_llm(
        self,
        request: SkillRequest
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate request using LLM"""
        # Find similar skills
        similar = self._find_similar_skills(request.proposed_name)
        similar_text = "\n".join(
            f"- {s.name}: {s.description}"
            for s in similar[:5]
        )

        prompt = self.VALIDATION_PROMPT.format(
            categories=", ".join(c.value for c in SkillCategory),
            name=request.proposed_name,
            description=request.proposed_description,
            category=request.proposed_category.value if request.proposed_category else "unknown",
            parent=request.proposed_parent or "none",
            reasoning=request.reasoning,
            similar_skills=similar_text or "None found",
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent evaluation
                max_tokens=300,
            )

            # Parse response
            lines = response.text.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip().upper()] = value.strip()

            approved = data.get("APPROVED", "").lower() == "yes"
            reason = data.get("REASON", "No reason provided")
            suggestions = {
                "primary_stat": data.get("STAT_PRIMARY", "intelligence"),
                "secondary_stat": data.get("STAT_SECONDARY"),
                "changes": data.get("SUGGESTED_CHANGES", ""),
            }

            return (approved, reason, suggestions)

        except Exception as e:
            return (False, f"Validation error: {str(e)}", {})

    def _validate_heuristic(self, request: SkillRequest) -> Tuple[bool, str]:
        """Validate using heuristics (no LLM)"""
        # Check name length
        if len(request.proposed_name) < 3:
            return (False, "Name too short")
        if len(request.proposed_name) > 50:
            return (False, "Name too long")

        # Check for similar names
        similar = self._find_similar_skills(request.proposed_name)
        if similar:
            for s in similar:
                if s.name.lower() == request.proposed_name.lower():
                    return (False, f"Duplicate of existing skill: {s.name}")

        # Check parent exists
        if request.proposed_parent:
            parent = self.skill_tree.get_by_name(request.proposed_parent)
            if not parent:
                return (False, f"Parent skill not found: {request.proposed_parent}")

        return (True, "Passed heuristic validation")

    def _find_similar_skills(self, name: str) -> List[Skill]:
        """Find skills with similar names"""
        name_lower = name.lower()
        similar = []

        for skill in self.skill_tree.get_all_skills():
            skill_lower = skill.name.lower()
            # Check for substring match or common words
            if name_lower in skill_lower or skill_lower in name_lower:
                similar.append(skill)
            else:
                # Check word overlap
                name_words = set(name_lower.split())
                skill_words = set(skill_lower.split())
                if len(name_words & skill_words) > 0:
                    similar.append(skill)

        return similar

    async def _create_skill_from_request(
        self,
        request: SkillRequest,
        suggestions: Dict[str, Any],
    ) -> Skill:
        """Create a skill from an approved request"""
        # Find parent
        parent_id = None
        depth = 0
        if request.proposed_parent:
            parent = self.skill_tree.get_by_name(request.proposed_parent)
            if parent:
                parent_id = parent.skill_id
                depth = parent.depth + 1

        # Determine stats
        primary_stat = suggestions.get("primary_stat", "intelligence")
        secondary_stat = suggestions.get("secondary_stat")
        if secondary_stat and secondary_stat.lower() == "none":
            secondary_stat = None

        # Create skill
        skill = Skill(
            name=request.proposed_name,
            description=request.proposed_description,
            category=request.proposed_category or SkillCategory.PHYSICAL,
            parent_id=parent_id,
            depth=depth,
            primary_stat=primary_stat,
            secondary_stat=secondary_stat,
        )

        # Add to tree
        self.skill_tree.add_skill(skill)

        # Track creator
        if request.requester_agent_id:
            self._skill_creators[skill.skill_id] = request.requester_agent_id

        return skill
