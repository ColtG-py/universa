"""
Procedural Memory
Stores how-to knowledge and learned procedures.
"How to do things" - learned skills and routines.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from agents.db.supabase_client import SupabaseClient


@dataclass
class Procedure:
    """
    A procedural memory - how to do something.
    """
    procedure_id: UUID
    agent_id: UUID
    procedure_name: str
    procedure_prompt: str  # Natural language description of how to do it

    # Performance tracking
    success_rate: float = 0.5
    usage_count: int = 0

    # Related skill
    related_skill_id: Optional[str] = None

    # Steps (optional structured breakdown)
    steps: List[str] = field(default_factory=list)

    # Conditions for when to use
    prerequisites: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to database format"""
        return {
            "procedure_id": str(self.procedure_id),
            "agent_id": str(self.agent_id),
            "procedure_name": self.procedure_name,
            "procedure_prompt": self.procedure_prompt,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_database_row(cls, row: Dict[str, Any]) -> "Procedure":
        """Create from database row"""
        return cls(
            procedure_id=UUID(row["procedure_id"]),
            agent_id=UUID(row["agent_id"]),
            procedure_name=row["procedure_name"],
            procedure_prompt=row["procedure_prompt"],
            success_rate=row.get("success_rate", 0.5),
            usage_count=row.get("usage_count", 0),
            created_at=row.get("created_at", datetime.utcnow()),
        )


class ProceduralMemory:
    """
    Procedural memory system for storing how-to knowledge.

    Procedural memories are:
    - Action-oriented (how to do things)
    - Refined through practice
    - Context-dependent
    - Can be automatized over time
    """

    def __init__(
        self,
        agent_id: UUID,
        supabase_client: Optional[SupabaseClient] = None,
    ):
        """
        Initialize procedural memory.

        Args:
            agent_id: Agent this memory belongs to
            supabase_client: Database client
        """
        self.agent_id = agent_id
        self.client = supabase_client
        self.table_name = "procedural_memory"

        # In-memory cache
        self._procedures: Dict[str, Procedure] = {}  # name -> procedure

    async def learn_procedure(
        self,
        procedure_name: str,
        procedure_prompt: str,
        steps: List[str] = None,
        related_skill_id: Optional[str] = None,
        prerequisites: List[str] = None,
        applicable_contexts: List[str] = None,
    ) -> Procedure:
        """
        Learn a new procedure or update an existing one.

        Args:
            procedure_name: Name of the procedure
            procedure_prompt: Natural language description
            steps: Optional structured steps
            related_skill_id: Associated skill
            prerequisites: What's needed first
            applicable_contexts: When to use this

        Returns:
            Created/updated Procedure
        """
        # Check if procedure exists
        existing = await self.get_procedure(procedure_name)
        if existing:
            # Update existing procedure
            return await self._refine_procedure(
                existing,
                procedure_prompt,
                steps
            )

        procedure = Procedure(
            procedure_id=uuid4(),
            agent_id=self.agent_id,
            procedure_name=procedure_name,
            procedure_prompt=procedure_prompt,
            steps=steps or [],
            related_skill_id=related_skill_id,
            prerequisites=prerequisites or [],
            applicable_contexts=applicable_contexts or [],
        )

        # Persist
        if self.client:
            self.client.table(self.table_name).insert(
                procedure.to_database_dict()
            ).execute()

        # Cache
        self._procedures[procedure_name] = procedure

        return procedure

    async def get_procedure(
        self,
        procedure_name: str
    ) -> Optional[Procedure]:
        """
        Get a procedure by name.

        Args:
            procedure_name: Name of the procedure

        Returns:
            Procedure if found
        """
        # Check cache
        if procedure_name in self._procedures:
            return self._procedures[procedure_name]

        if not self.client:
            return None

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .eq("procedure_name", procedure_name)
            .execute()
        )

        if result.data:
            procedure = Procedure.from_database_row(result.data[0])
            self._procedures[procedure_name] = procedure
            return procedure

        return None

    async def use_procedure(
        self,
        procedure_name: str,
        success: bool
    ) -> Optional[Procedure]:
        """
        Record usage of a procedure and update success rate.

        Args:
            procedure_name: Name of the procedure
            success: Whether the usage was successful

        Returns:
            Updated procedure
        """
        procedure = await self.get_procedure(procedure_name)
        if not procedure:
            return None

        # Update success rate (moving average)
        old_rate = procedure.success_rate
        old_count = procedure.usage_count
        new_count = old_count + 1

        # Weighted update: give more weight to recent attempts
        if success:
            new_rate = (old_rate * old_count + 1.0) / new_count
        else:
            new_rate = (old_rate * old_count + 0.0) / new_count

        procedure.success_rate = new_rate
        procedure.usage_count = new_count
        procedure.last_used = datetime.utcnow()

        # Update in database
        if self.client:
            self.client.table(self.table_name).update({
                "success_rate": new_rate,
                "usage_count": new_count,
            }).eq("procedure_id", str(procedure.procedure_id)).execute()

        return procedure

    async def get_all_procedures(self) -> List[Procedure]:
        """Get all learned procedures"""
        if not self.client:
            return list(self._procedures.values())

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .order("usage_count", desc=True)
            .execute()
        )

        return [Procedure.from_database_row(row) for row in result.data or []]

    async def get_procedures_for_skill(
        self,
        skill_id: str
    ) -> List[Procedure]:
        """
        Get procedures related to a skill.

        Args:
            skill_id: Skill identifier

        Returns:
            Related procedures
        """
        if not self.client:
            return [p for p in self._procedures.values()
                    if p.related_skill_id == skill_id]

        # Since related_skill_id isn't in the schema,
        # search by name containing the skill
        skill_name = skill_id.split(".")[-1]  # Get last part
        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .ilike("procedure_name", f"%{skill_name}%")
            .execute()
        )

        return [Procedure.from_database_row(row) for row in result.data or []]

    async def get_best_procedures(
        self,
        min_success_rate: float = 0.7,
        min_uses: int = 3
    ) -> List[Procedure]:
        """
        Get well-practiced, successful procedures.

        Args:
            min_success_rate: Minimum success rate
            min_uses: Minimum usage count

        Returns:
            Best procedures
        """
        if not self.client:
            return [
                p for p in self._procedures.values()
                if p.success_rate >= min_success_rate and p.usage_count >= min_uses
            ]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .gte("success_rate", min_success_rate)
            .gte("usage_count", min_uses)
            .order("success_rate", desc=True)
            .execute()
        )

        return [Procedure.from_database_row(row) for row in result.data or []]

    async def search_procedures(
        self,
        query: str,
        limit: int = 5
    ) -> List[Procedure]:
        """
        Search procedures by name or description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching procedures
        """
        if not self.client:
            query_lower = query.lower()
            return [
                p for p in self._procedures.values()
                if query_lower in p.procedure_name.lower()
                or query_lower in p.procedure_prompt.lower()
            ][:limit]

        result = (
            self.client.table(self.table_name)
            .select("*")
            .eq("agent_id", str(self.agent_id))
            .or_(f"procedure_name.ilike.%{query}%,procedure_prompt.ilike.%{query}%")
            .limit(limit)
            .execute()
        )

        return [Procedure.from_database_row(row) for row in result.data or []]

    async def forget_procedure(
        self,
        procedure_name: str
    ) -> bool:
        """
        Remove a procedure from memory.

        Args:
            procedure_name: Procedure to forget

        Returns:
            True if deleted
        """
        if procedure_name in self._procedures:
            del self._procedures[procedure_name]

        if not self.client:
            return True

        result = (
            self.client.table(self.table_name)
            .delete()
            .eq("agent_id", str(self.agent_id))
            .eq("procedure_name", procedure_name)
            .execute()
        )

        return len(result.data or []) > 0

    async def _refine_procedure(
        self,
        procedure: Procedure,
        new_prompt: str,
        new_steps: Optional[List[str]]
    ) -> Procedure:
        """
        Refine an existing procedure with new information.

        Args:
            procedure: Existing procedure
            new_prompt: New description
            new_steps: New steps

        Returns:
            Refined procedure
        """
        # Combine prompts if significantly different
        if len(new_prompt) > len(procedure.procedure_prompt) * 0.5:
            # New prompt has substantial content, consider merging
            procedure.procedure_prompt = new_prompt

        if new_steps:
            procedure.steps = new_steps

        # Update in database
        if self.client:
            self.client.table(self.table_name).update({
                "procedure_prompt": procedure.procedure_prompt,
            }).eq("procedure_id", str(procedure.procedure_id)).execute()

        return procedure

    async def generate_procedure_from_episode(
        self,
        episode_summary: str,
        skill_used: str,
        success: bool
    ) -> Optional[Procedure]:
        """
        Learn a procedure from a successful episode.

        Args:
            episode_summary: What happened
            skill_used: Skill that was used
            success: Whether it was successful

        Returns:
            Learned procedure if successful episode
        """
        if not success:
            return None  # Only learn from successes

        procedure_name = f"how_to_{skill_used.replace('.', '_')}"

        return await self.learn_procedure(
            procedure_name=procedure_name,
            procedure_prompt=f"Based on experience: {episode_summary}",
            related_skill_id=skill_used
        )
