"""
Skill Manager
Central management of agent skills.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID

from agents.skills.taxonomy import Skill, SkillTree, SkillCategory, build_skill_tree
from agents.skills.progression import SkillProgress, SkillProgressionSystem


class SkillManager:
    """
    Central manager for the skill system.

    Handles:
    - Skill tree management
    - Agent skill tracking
    - Skill execution
    - Learning and progression
    """

    def __init__(
        self,
        skill_tree: Optional[SkillTree] = None,
        progression_system: Optional[SkillProgressionSystem] = None,
    ):
        """
        Initialize skill manager.

        Args:
            skill_tree: Pre-built skill tree (builds default if None)
            progression_system: Progression system (creates if None)
        """
        self.skill_tree = skill_tree or build_skill_tree()
        self.progression = progression_system or SkillProgressionSystem(
            skill_tree=self.skill_tree
        )

        # Track which skills each agent has unlocked
        self._agent_skills: Dict[UUID, set[UUID]] = {}

    def get_skill(self, name_or_id: str | UUID) -> Optional[Skill]:
        """Get a skill by name or ID"""
        if isinstance(name_or_id, UUID):
            return self.skill_tree.get_skill(name_or_id)
        return self.skill_tree.get_by_name(name_or_id)

    def get_skills_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category"""
        return self.skill_tree.get_by_category(category)

    def get_agent_skills(self, agent_id: UUID) -> List[Tuple[Skill, SkillProgress]]:
        """
        Get all skills an agent has learned.

        Returns:
            List of (Skill, SkillProgress) tuples
        """
        progress_dict = self.progression.get_all_progress(agent_id)
        result = []

        for skill_id, progress in progress_dict.items():
            skill = self.skill_tree.get_skill(skill_id)
            if skill:
                result.append((skill, progress))

        return result

    def agent_has_skill(self, agent_id: UUID, skill_name: str) -> bool:
        """Check if agent has learned a skill"""
        skill = self.skill_tree.get_by_name(skill_name)
        if not skill:
            return False

        level = self.progression.get_level(agent_id, skill.skill_id)
        return level > 0

    def get_agent_skill_level(self, agent_id: UUID, skill_name: str) -> int:
        """Get agent's level in a skill"""
        skill = self.skill_tree.get_by_name(skill_name)
        if not skill:
            return 0
        return self.progression.get_level(agent_id, skill.skill_id)

    def learn_skill(
        self,
        agent_id: UUID,
        skill_name: str,
        initial_xp: int = 0,
    ) -> Tuple[bool, str]:
        """
        Have an agent learn a new skill.

        Args:
            agent_id: Agent learning
            skill_name: Skill to learn
            initial_xp: Starting XP

        Returns:
            Tuple of (success, message)
        """
        skill = self.skill_tree.get_by_name(skill_name)
        if not skill:
            return (False, f"Unknown skill: {skill_name}")

        # Check if already learned
        current_level = self.progression.get_level(agent_id, skill.skill_id)
        if current_level > 0:
            return (True, f"Already know {skill_name} at level {current_level}")

        # Initialize progress
        progress = self.progression.get_progress(agent_id, skill.skill_id)
        progress.first_learned = datetime.utcnow()

        if initial_xp > 0:
            self.progression.add_xp(agent_id, skill.skill_id, initial_xp)

        # Track in agent skills
        if agent_id not in self._agent_skills:
            self._agent_skills[agent_id] = set()
        self._agent_skills[agent_id].add(skill.skill_id)

        return (True, f"Learned {skill_name}")

    def execute_skill(
        self,
        agent_id: UUID,
        skill_name: str,
        target: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        agent_stats: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a skill.

        Args:
            agent_id: Agent using skill
            skill_name: Skill to use
            target: Target of skill
            parameters: Additional parameters
            agent_stats: Agent's stats for modifiers

        Returns:
            Execution result
        """
        skill = self.skill_tree.get_by_name(skill_name)
        if not skill:
            return {
                "success": False,
                "error": f"Unknown skill: {skill_name}",
            }

        # Check if agent has the skill
        level = self.progression.get_level(agent_id, skill.skill_id)
        if level == 0:
            return {
                "success": False,
                "error": f"You haven't learned {skill_name}",
            }

        # Calculate effectiveness
        skill_bonus = self.progression.get_skill_bonus(agent_id, skill.skill_id)
        stat_modifier = skill.get_stat_modifier(agent_stats or {})
        total_modifier = skill_bonus * stat_modifier

        # Calculate success chance (difficulty 10 = average)
        difficulty = parameters.get("difficulty", 10) if parameters else 10
        success_chance = self.progression.calculate_success_chance(
            agent_id, skill.skill_id, difficulty, agent_stats
        )

        # Roll for success
        import random
        roll = random.random()
        success = roll < success_chance

        # Award XP
        xp_gained, level_ups = self.progression.use_skill(
            agent_id=agent_id,
            skill_id=skill.skill_id,
            difficulty=difficulty / 10.0,
            success=success,
        )

        return {
            "success": success,
            "skill": skill_name,
            "level": level,
            "target": target,
            "modifier": total_modifier,
            "success_chance": success_chance,
            "roll": roll,
            "xp_gained": xp_gained,
            "level_ups": level_ups,
            "description": self._describe_skill_use(skill, target, success),
        }

    def _describe_skill_use(
        self,
        skill: Skill,
        target: Optional[str],
        success: bool
    ) -> str:
        """Generate description of skill use"""
        if success:
            if target:
                return f"Successfully used {skill.name} on {target}."
            return f"Successfully used {skill.name}."
        else:
            if target:
                return f"Failed to use {skill.name} on {target}."
            return f"Failed to use {skill.name}."

    def can_learn_skill(
        self,
        agent_id: UUID,
        skill_name: str,
        agent_stats: Dict[str, int],
    ) -> Tuple[bool, List[str]]:
        """
        Check if an agent can learn a skill.

        Args:
            agent_id: Agent to check
            skill_name: Skill to learn
            agent_stats: Agent's current stats

        Returns:
            Tuple of (can_learn, list of unmet requirements)
        """
        skill = self.skill_tree.get_by_name(skill_name)
        if not skill:
            return (False, [f"Unknown skill: {skill_name}"])

        # Get current skill levels
        all_progress = self.progression.get_all_progress(agent_id)
        skill_levels = {
            sid: prog.current_level
            for sid, prog in all_progress.items()
        }

        return self.skill_tree.check_requirements(skill, agent_stats, skill_levels)

    def get_available_skills(
        self,
        agent_id: UUID,
        agent_stats: Dict[str, int],
        category: Optional[SkillCategory] = None,
    ) -> List[Skill]:
        """
        Get skills available for an agent to learn.

        Args:
            agent_id: Agent to check
            agent_stats: Agent's stats
            category: Optional category filter

        Returns:
            List of learnable skills
        """
        available = []

        skills = self.skill_tree.get_all_skills()
        if category:
            skills = [s for s in skills if s.category == category]

        for skill in skills:
            # Skip if already learned
            level = self.progression.get_level(agent_id, skill.skill_id)
            if level > 0:
                continue

            # Check requirements
            can_learn, _ = self.can_learn_skill(agent_id, skill.name, agent_stats)
            if can_learn:
                available.append(skill)

        return available

    def get_skill_tree_view(
        self,
        root_category: Optional[SkillCategory] = None
    ) -> Dict[str, Any]:
        """
        Get a tree view of skills.

        Args:
            root_category: Optional category to start from

        Returns:
            Nested dictionary of skills
        """
        def build_node(skill: Skill) -> Dict[str, Any]:
            children = self.skill_tree.get_children(skill.skill_id)
            return {
                "id": str(skill.skill_id),
                "name": skill.name,
                "description": skill.description,
                "category": skill.category.value,
                "depth": skill.depth,
                "children": [build_node(c) for c in children],
            }

        roots = self.skill_tree.get_root_skills()
        if root_category:
            roots = [r for r in roots if r.category == root_category]

        return {
            "skills": [build_node(r) for r in roots]
        }

    def get_agent_summary(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Get a summary of an agent's skills.

        Returns:
            Summary including specializations, total skills, etc.
        """
        all_skills = self.get_agent_skills(agent_id)
        specializations = self.progression.get_agent_specializations(agent_id)
        stat_bonuses = self.progression.get_stat_bonuses_from_skills(agent_id)

        return {
            "total_skills": len(all_skills),
            "total_levels": sum(p.current_level for _, p in all_skills),
            "specializations": [
                {
                    "skill": self.skill_tree.get_skill(sid).name if self.skill_tree.get_skill(sid) else "Unknown",
                    "level": level,
                }
                for sid, level in specializations[:5]  # Top 5
            ],
            "stat_bonuses": stat_bonuses,
            "highest_skill": all_skills[0][0].name if all_skills else None,
        }
