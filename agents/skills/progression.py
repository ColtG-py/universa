"""
Skill Progression System
XP gain, leveling, and skill development.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
import math


@dataclass
class SkillProgress:
    """Progress in a specific skill"""
    skill_id: UUID
    agent_id: UUID
    current_level: int = 0
    current_xp: int = 0
    total_xp: int = 0
    times_used: int = 0
    last_used: Optional[datetime] = None
    first_learned: Optional[datetime] = None

    # Level milestones reached
    milestones: List[int] = field(default_factory=list)

    def xp_to_next_level(self, base_cost: int = 100, multiplier: float = 1.5) -> int:
        """Calculate XP needed for next level"""
        return calculate_xp_for_level(self.current_level + 1, base_cost, multiplier)

    def xp_progress_percent(self, base_cost: int = 100, multiplier: float = 1.5) -> float:
        """Get percentage progress to next level"""
        needed = self.xp_to_next_level(base_cost, multiplier)
        return min(100.0, (self.current_xp / needed) * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "skill_id": str(self.skill_id),
            "agent_id": str(self.agent_id),
            "current_level": self.current_level,
            "current_xp": self.current_xp,
            "total_xp": self.total_xp,
            "times_used": self.times_used,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


def calculate_xp_for_level(
    level: int,
    base_cost: int = 100,
    multiplier: float = 1.5
) -> int:
    """
    Calculate XP required to reach a level.

    Uses exponential scaling:
    XP = base_cost * (multiplier ^ (level - 1))

    Args:
        level: Target level (1+)
        base_cost: XP for level 1
        multiplier: Growth rate per level

    Returns:
        XP required for level
    """
    if level <= 0:
        return 0
    return int(base_cost * (multiplier ** (level - 1)))


def calculate_total_xp_for_level(
    level: int,
    base_cost: int = 100,
    multiplier: float = 1.5
) -> int:
    """
    Calculate total XP required to reach a level from 0.

    Args:
        level: Target level
        base_cost: XP for level 1
        multiplier: Growth rate

    Returns:
        Total XP from level 0 to target
    """
    total = 0
    for lvl in range(1, level + 1):
        total += calculate_xp_for_level(lvl, base_cost, multiplier)
    return total


class SkillProgressionSystem:
    """
    Manages skill progression for agents.

    Features:
    - XP gain on skill use
    - Level-up mechanics
    - Parent skill XP propagation (50% to parent)
    - Stat bonuses from skill levels
    """

    # XP propagation rate to parent skills
    PARENT_XP_RATE = 0.5

    # Level milestones (unlock features, titles, etc.)
    MILESTONES = [10, 25, 50, 75, 100]

    def __init__(
        self,
        base_xp_cost: int = 100,
        xp_multiplier: float = 1.5,
        skill_tree: Optional[Any] = None,  # SkillTree
    ):
        """
        Initialize progression system.

        Args:
            base_xp_cost: XP for level 1
            xp_multiplier: XP growth rate
            skill_tree: Optional skill tree for parent lookup
        """
        self.base_cost = base_xp_cost
        self.multiplier = xp_multiplier
        self.skill_tree = skill_tree

        # Progress tracking: agent_id -> skill_id -> SkillProgress
        self._progress: Dict[UUID, Dict[UUID, SkillProgress]] = {}

    def get_progress(
        self,
        agent_id: UUID,
        skill_id: UUID
    ) -> SkillProgress:
        """Get or create progress for a skill"""
        if agent_id not in self._progress:
            self._progress[agent_id] = {}

        if skill_id not in self._progress[agent_id]:
            self._progress[agent_id][skill_id] = SkillProgress(
                skill_id=skill_id,
                agent_id=agent_id,
                first_learned=datetime.utcnow(),
            )

        return self._progress[agent_id][skill_id]

    def get_level(self, agent_id: UUID, skill_id: UUID) -> int:
        """Get current level in a skill"""
        progress = self.get_progress(agent_id, skill_id)
        return progress.current_level

    def get_all_progress(self, agent_id: UUID) -> Dict[UUID, SkillProgress]:
        """Get all skill progress for an agent"""
        return self._progress.get(agent_id, {})

    def add_xp(
        self,
        agent_id: UUID,
        skill_id: UUID,
        xp_amount: int,
        propagate_to_parent: bool = True,
    ) -> Tuple[int, List[Tuple[UUID, int]]]:
        """
        Add XP to a skill.

        Args:
            agent_id: Agent gaining XP
            skill_id: Skill gaining XP
            xp_amount: Amount of XP
            propagate_to_parent: Whether to give parent skill XP

        Returns:
            Tuple of (levels gained, list of (skill_id, new_level) for all affected)
        """
        progress = self.get_progress(agent_id, skill_id)

        # Add XP
        progress.current_xp += xp_amount
        progress.total_xp += xp_amount
        progress.times_used += 1
        progress.last_used = datetime.utcnow()

        # Check for level ups
        levels_gained = 0
        level_ups = []

        while True:
            xp_needed = calculate_xp_for_level(
                progress.current_level + 1,
                self.base_cost,
                self.multiplier
            )

            if progress.current_xp >= xp_needed:
                progress.current_xp -= xp_needed
                progress.current_level += 1
                levels_gained += 1

                # Check milestones
                if progress.current_level in self.MILESTONES:
                    progress.milestones.append(progress.current_level)

                level_ups.append((skill_id, progress.current_level))
            else:
                break

        # Propagate to parent
        if propagate_to_parent and self.skill_tree:
            skill = self.skill_tree.get_skill(skill_id)
            if skill and skill.parent_id:
                parent_xp = int(xp_amount * self.PARENT_XP_RATE)
                if parent_xp > 0:
                    _, parent_levels = self.add_xp(
                        agent_id,
                        skill.parent_id,
                        parent_xp,
                        propagate_to_parent=True,  # Recursive
                    )
                    level_ups.extend(parent_levels)

        return (levels_gained, level_ups)

    def use_skill(
        self,
        agent_id: UUID,
        skill_id: UUID,
        difficulty: float = 1.0,
        success: bool = True,
    ) -> Tuple[int, List[Tuple[UUID, int]]]:
        """
        Record skill use and award XP.

        Args:
            agent_id: Agent using skill
            skill_id: Skill being used
            difficulty: Difficulty modifier (0.5-2.0)
            success: Whether skill use was successful

        Returns:
            XP gained and any level ups
        """
        # Base XP for using a skill
        base_xp = 10

        # Modify by difficulty
        xp = int(base_xp * difficulty)

        # Bonus for success
        if success:
            xp = int(xp * 1.5)

        # Reduced XP if skill is much higher than difficulty
        current_level = self.get_level(agent_id, skill_id)
        if current_level > 50 and difficulty < 1.0:
            xp = int(xp * 0.5)  # Reduced XP for easy tasks at high levels

        return self.add_xp(agent_id, skill_id, xp)

    def get_skill_bonus(
        self,
        agent_id: UUID,
        skill_id: UUID,
    ) -> float:
        """
        Get skill level bonus as a multiplier.

        Returns:
            Multiplier (1.0 = no bonus, 2.0 = double effectiveness)
        """
        level = self.get_level(agent_id, skill_id)

        # Each level gives 1% bonus, with diminishing returns after 50
        if level <= 50:
            bonus = level * 0.01
        else:
            bonus = 0.50 + (level - 50) * 0.005

        return 1.0 + bonus

    def calculate_success_chance(
        self,
        agent_id: UUID,
        skill_id: UUID,
        difficulty: int,
        agent_stats: Optional[Dict[str, int]] = None,
    ) -> float:
        """
        Calculate chance of success for a skill check.

        Args:
            agent_id: Agent attempting skill
            skill_id: Skill being used
            difficulty: Difficulty class (1-20 scale)
            agent_stats: Agent's stats for modifier

        Returns:
            Success probability (0.0-1.0)
        """
        level = self.get_level(agent_id, skill_id)

        # Base chance from level
        base_chance = 0.5 + (level / 100) * 0.4  # 50% at level 0, 90% at level 100

        # Difficulty modifier
        difficulty_mod = (10 - difficulty) * 0.05  # ±50% based on difficulty

        # Stat modifier (if skill tree and stats available)
        stat_mod = 0.0
        if self.skill_tree and agent_stats:
            skill = self.skill_tree.get_skill(skill_id)
            if skill:
                stat_mod = (skill.get_stat_modifier(agent_stats) - 1.0)

        # Combine
        chance = base_chance + difficulty_mod + stat_mod

        # Clamp to reasonable bounds
        return max(0.05, min(0.95, chance))

    def get_agent_specializations(
        self,
        agent_id: UUID,
        min_level: int = 25
    ) -> List[Tuple[UUID, int]]:
        """
        Get agent's specializations (high-level skills).

        Args:
            agent_id: Agent to check
            min_level: Minimum level to count as specialization

        Returns:
            List of (skill_id, level) tuples
        """
        all_progress = self.get_all_progress(agent_id)
        specializations = []

        for skill_id, progress in all_progress.items():
            if progress.current_level >= min_level:
                specializations.append((skill_id, progress.current_level))

        # Sort by level descending
        specializations.sort(key=lambda x: x[1], reverse=True)
        return specializations

    def get_stat_bonuses_from_skills(
        self,
        agent_id: UUID,
    ) -> Dict[str, float]:
        """
        Calculate stat bonuses from skill levels.

        High-level skills provide passive stat bonuses.

        Returns:
            Dict of stat -> bonus amount
        """
        bonuses = {}

        if not self.skill_tree:
            return bonuses

        all_progress = self.get_all_progress(agent_id)

        for skill_id, progress in all_progress.items():
            skill = self.skill_tree.get_skill(skill_id)
            if skill and skill.is_passive and progress.current_level >= 10:
                # Every 10 levels gives +0.1 to primary stat
                if skill.primary_stat:
                    stat_bonus = (progress.current_level // 10) * 0.1
                    bonuses[skill.primary_stat] = bonuses.get(skill.primary_stat, 0) + stat_bonus

        return bonuses

    def to_database_format(
        self,
        agent_id: UUID
    ) -> List[Dict[str, Any]]:
        """Convert agent's progress to database format"""
        all_progress = self.get_all_progress(agent_id)
        return [p.to_dict() for p in all_progress.values()]

    def load_from_database(
        self,
        agent_id: UUID,
        data: List[Dict[str, Any]]
    ) -> None:
        """Load progress from database format"""
        if agent_id not in self._progress:
            self._progress[agent_id] = {}

        for item in data:
            skill_id = UUID(item["skill_id"])
            self._progress[agent_id][skill_id] = SkillProgress(
                skill_id=skill_id,
                agent_id=agent_id,
                current_level=item.get("current_level", 0),
                current_xp=item.get("current_xp", 0),
                total_xp=item.get("total_xp", 0),
                times_used=item.get("times_used", 0),
                last_used=datetime.fromisoformat(item["last_used"]) if item.get("last_used") else None,
            )
