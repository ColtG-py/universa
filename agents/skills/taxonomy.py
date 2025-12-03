"""
Skill Taxonomy
Hierarchical skill tree with categories and relationships.
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from enum import Enum


class SkillCategory(str, Enum):
    """Top-level skill categories"""
    # Combat
    COMBAT = "combat"
    MELEE = "melee"
    RANGED = "ranged"
    DEFENSE = "defense"

    # Magic
    MAGIC = "magic"
    ELEMENTAL = "elemental"
    DIVINE = "divine"
    ARCANE = "arcane"

    # Crafting
    CRAFTING = "crafting"
    SMITHING = "smithing"
    ALCHEMY = "alchemy"
    ENCHANTING = "enchanting"

    # Gathering
    GATHERING = "gathering"
    MINING = "mining"
    HERBALISM = "herbalism"
    HUNTING = "hunting"

    # Social
    SOCIAL = "social"
    PERSUASION = "persuasion"
    INTIMIDATION = "intimidation"
    DECEPTION = "deception"

    # Knowledge
    KNOWLEDGE = "knowledge"
    LORE = "lore"
    LANGUAGES = "languages"
    MEDICINE = "medicine"

    # Physical
    PHYSICAL = "physical"
    ATHLETICS = "athletics"
    STEALTH = "stealth"
    SURVIVAL = "survival"

    # Trade
    TRADE = "trade"
    MERCANTILE = "mercantile"
    FARMING = "farming"
    ANIMAL_HUSBANDRY = "animal_husbandry"


class StatRequirement(str, Enum):
    """Stats that skills can require"""
    STRENGTH = "strength"
    DEXTERITY = "dexterity"
    CONSTITUTION = "constitution"
    INTELLIGENCE = "intelligence"
    WISDOM = "wisdom"
    CHARISMA = "charisma"


@dataclass
class Skill:
    """
    A skill in the skill tree.

    Skills are hierarchical:
    - Parent skills provide general bonuses
    - Child skills are specializations
    - XP flows up to parent skills (50% rate)
    """
    skill_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    category: SkillCategory = SkillCategory.PHYSICAL

    # Hierarchy
    parent_id: Optional[UUID] = None
    depth: int = 0  # 0 = root, 1 = category, 2 = specialization, etc.

    # Requirements
    stat_requirements: Dict[str, int] = field(default_factory=dict)  # stat -> min value
    skill_requirements: List[UUID] = field(default_factory=list)  # prerequisite skills
    level_requirement: int = 0  # minimum level in parent skill

    # Stat bonuses when using this skill
    primary_stat: Optional[str] = None  # Main stat that affects skill
    secondary_stat: Optional[str] = None  # Secondary stat influence

    # Progression
    base_xp_cost: int = 100  # XP to reach level 1
    xp_multiplier: float = 1.5  # XP cost multiplier per level
    max_level: int = 100

    # Effects
    effects: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    is_active: bool = True  # Can be used actively
    is_passive: bool = False  # Provides passive bonuses
    cooldown_seconds: float = 0.0  # Cooldown between uses

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "skill_id": str(self.skill_id),
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "depth": self.depth,
            "stat_requirements": self.stat_requirements,
            "primary_stat": self.primary_stat,
            "secondary_stat": self.secondary_stat,
            "max_level": self.max_level,
            "is_active": self.is_active,
            "is_passive": self.is_passive,
        }

    def get_stat_modifier(self, stats: Dict[str, int]) -> float:
        """
        Calculate modifier based on relevant stats.

        Returns multiplier (1.0 = no bonus, 1.2 = 20% bonus, etc.)
        """
        modifier = 1.0

        if self.primary_stat and self.primary_stat in stats:
            # Primary stat: (stat - 10) * 2% per point
            stat_value = stats[self.primary_stat]
            modifier += (stat_value - 10) * 0.02

        if self.secondary_stat and self.secondary_stat in stats:
            # Secondary stat: (stat - 10) * 1% per point
            stat_value = stats[self.secondary_stat]
            modifier += (stat_value - 10) * 0.01

        return max(0.5, modifier)  # Minimum 50% effectiveness


class SkillTree:
    """
    Manages the complete skill hierarchy.

    Skills are organized in a tree:
    - Root categories (Combat, Magic, Crafting, etc.)
    - Subcategories (Melee, Ranged, Elemental, etc.)
    - Specific skills (Sword, Bow, Fireball, etc.)
    - Specializations (Longsword Mastery, Flame Arrow, etc.)
    """

    def __init__(self):
        """Initialize skill tree"""
        self._skills: Dict[UUID, Skill] = {}
        self._by_name: Dict[str, UUID] = {}
        self._by_category: Dict[SkillCategory, List[UUID]] = {
            cat: [] for cat in SkillCategory
        }
        self._children: Dict[UUID, List[UUID]] = {}  # parent_id -> child_ids

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the tree"""
        self._skills[skill.skill_id] = skill
        self._by_name[skill.name.lower()] = skill.skill_id
        self._by_category[skill.category].append(skill.skill_id)

        # Track parent-child relationships
        if skill.parent_id:
            if skill.parent_id not in self._children:
                self._children[skill.parent_id] = []
            self._children[skill.parent_id].append(skill.skill_id)

    def get_skill(self, skill_id: UUID) -> Optional[Skill]:
        """Get skill by ID"""
        return self._skills.get(skill_id)

    def get_by_name(self, name: str) -> Optional[Skill]:
        """Get skill by name"""
        skill_id = self._by_name.get(name.lower())
        return self._skills.get(skill_id) if skill_id else None

    def get_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category"""
        return [
            self._skills[sid]
            for sid in self._by_category.get(category, [])
        ]

    def get_children(self, parent_id: UUID) -> List[Skill]:
        """Get child skills of a parent"""
        child_ids = self._children.get(parent_id, [])
        return [self._skills[cid] for cid in child_ids]

    def get_ancestors(self, skill_id: UUID) -> List[Skill]:
        """Get all ancestor skills (parent, grandparent, etc.)"""
        ancestors = []
        current = self._skills.get(skill_id)

        while current and current.parent_id:
            parent = self._skills.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_path_to_skill(self, skill_id: UUID) -> List[Skill]:
        """Get path from root to skill"""
        path = self.get_ancestors(skill_id)
        path.reverse()
        skill = self._skills.get(skill_id)
        if skill:
            path.append(skill)
        return path

    def check_requirements(
        self,
        skill: Skill,
        agent_stats: Dict[str, int],
        agent_skill_levels: Dict[UUID, int],
    ) -> tuple[bool, List[str]]:
        """
        Check if an agent meets skill requirements.

        Args:
            skill: Skill to check
            agent_stats: Agent's stats
            agent_skill_levels: Agent's skill levels

        Returns:
            Tuple of (meets_requirements, list of unmet requirements)
        """
        unmet = []

        # Check stat requirements
        for stat, min_value in skill.stat_requirements.items():
            if agent_stats.get(stat, 0) < min_value:
                unmet.append(f"{stat} must be at least {min_value}")

        # Check skill requirements
        for prereq_id in skill.skill_requirements:
            prereq = self._skills.get(prereq_id)
            if prereq:
                current_level = agent_skill_levels.get(prereq_id, 0)
                if current_level < skill.level_requirement:
                    unmet.append(
                        f"{prereq.name} must be at least level {skill.level_requirement}"
                    )

        # Check parent skill level
        if skill.parent_id and skill.level_requirement > 0:
            parent_level = agent_skill_levels.get(skill.parent_id, 0)
            if parent_level < skill.level_requirement:
                parent = self._skills.get(skill.parent_id)
                parent_name = parent.name if parent else "parent skill"
                unmet.append(
                    f"{parent_name} must be at least level {skill.level_requirement}"
                )

        return (len(unmet) == 0, unmet)

    def get_all_skills(self) -> List[Skill]:
        """Get all skills"""
        return list(self._skills.values())

    def get_root_skills(self) -> List[Skill]:
        """Get top-level skills (no parent)"""
        return [s for s in self._skills.values() if s.parent_id is None]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tree to dictionary"""
        return {
            str(sid): skill.to_dict()
            for sid, skill in self._skills.items()
        }


def get_base_skills() -> List[Skill]:
    """
    Get the base skill taxonomy.

    Returns ~100 base skills organized hierarchically.
    """
    skills = []

    # =========================================================================
    # COMBAT SKILLS
    # =========================================================================
    combat = Skill(
        name="Combat",
        description="General combat abilities",
        category=SkillCategory.COMBAT,
        depth=0,
        primary_stat="strength",
        secondary_stat="dexterity",
        is_passive=True,
    )
    skills.append(combat)

    # Melee
    melee = Skill(
        name="Melee Combat",
        description="Close-quarters combat with weapons",
        category=SkillCategory.MELEE,
        parent_id=combat.skill_id,
        depth=1,
        primary_stat="strength",
        secondary_stat="dexterity",
    )
    skills.append(melee)

    for weapon, desc in [
        ("Swords", "Fighting with bladed weapons"),
        ("Axes", "Fighting with axes and hatchets"),
        ("Maces", "Fighting with blunt weapons"),
        ("Polearms", "Fighting with spears and halberds"),
        ("Daggers", "Fighting with short blades"),
        ("Unarmed", "Fighting without weapons"),
    ]:
        skill = Skill(
            name=weapon,
            description=desc,
            category=SkillCategory.MELEE,
            parent_id=melee.skill_id,
            depth=2,
            primary_stat="strength" if weapon != "Daggers" else "dexterity",
            secondary_stat="dexterity",
            level_requirement=5,
        )
        skills.append(skill)

    # Ranged
    ranged = Skill(
        name="Ranged Combat",
        description="Combat at a distance",
        category=SkillCategory.RANGED,
        parent_id=combat.skill_id,
        depth=1,
        primary_stat="dexterity",
        secondary_stat="strength",
    )
    skills.append(ranged)

    for weapon, desc in [
        ("Bows", "Using bows and arrows"),
        ("Crossbows", "Using crossbows"),
        ("Throwing", "Throwing weapons"),
    ]:
        skill = Skill(
            name=weapon,
            description=desc,
            category=SkillCategory.RANGED,
            parent_id=ranged.skill_id,
            depth=2,
            primary_stat="dexterity",
            level_requirement=5,
        )
        skills.append(skill)

    # Defense
    defense = Skill(
        name="Defense",
        description="Defensive combat techniques",
        category=SkillCategory.DEFENSE,
        parent_id=combat.skill_id,
        depth=1,
        primary_stat="constitution",
        secondary_stat="dexterity",
        is_passive=True,
    )
    skills.append(defense)

    for tech, desc in [
        ("Blocking", "Blocking attacks with weapons or shields"),
        ("Dodging", "Evading attacks"),
        ("Parrying", "Deflecting attacks"),
        ("Armor Use", "Effective use of armor"),
    ]:
        skill = Skill(
            name=tech,
            description=desc,
            category=SkillCategory.DEFENSE,
            parent_id=defense.skill_id,
            depth=2,
            primary_stat="dexterity" if tech == "Dodging" else "constitution",
            is_passive=True,
            level_requirement=5,
        )
        skills.append(skill)

    # =========================================================================
    # MAGIC SKILLS
    # =========================================================================
    magic = Skill(
        name="Magic",
        description="Arcane and divine magical abilities",
        category=SkillCategory.MAGIC,
        depth=0,
        primary_stat="intelligence",
        secondary_stat="wisdom",
    )
    skills.append(magic)

    # Elemental Magic
    elemental = Skill(
        name="Elemental Magic",
        description="Control over the elements",
        category=SkillCategory.ELEMENTAL,
        parent_id=magic.skill_id,
        depth=1,
        primary_stat="intelligence",
        stat_requirements={"intelligence": 12},
    )
    skills.append(elemental)

    for element, desc in [
        ("Fire Magic", "Control over flames"),
        ("Ice Magic", "Control over frost and cold"),
        ("Lightning Magic", "Control over electricity"),
        ("Earth Magic", "Control over stone and soil"),
        ("Water Magic", "Control over water"),
        ("Wind Magic", "Control over air"),
    ]:
        skill = Skill(
            name=element,
            description=desc,
            category=SkillCategory.ELEMENTAL,
            parent_id=elemental.skill_id,
            depth=2,
            primary_stat="intelligence",
            level_requirement=10,
        )
        skills.append(skill)

    # Divine Magic
    divine = Skill(
        name="Divine Magic",
        description="Magic granted by higher powers",
        category=SkillCategory.DIVINE,
        parent_id=magic.skill_id,
        depth=1,
        primary_stat="wisdom",
        stat_requirements={"wisdom": 12},
    )
    skills.append(divine)

    for school, desc in [
        ("Healing", "Restoring health and curing ailments"),
        ("Protection", "Defensive divine magic"),
        ("Smiting", "Offensive divine magic"),
        ("Blessing", "Empowering allies"),
    ]:
        skill = Skill(
            name=school,
            description=desc,
            category=SkillCategory.DIVINE,
            parent_id=divine.skill_id,
            depth=2,
            primary_stat="wisdom",
            level_requirement=10,
        )
        skills.append(skill)

    # Arcane Magic
    arcane = Skill(
        name="Arcane Magic",
        description="Pure magical manipulation",
        category=SkillCategory.ARCANE,
        parent_id=magic.skill_id,
        depth=1,
        primary_stat="intelligence",
        stat_requirements={"intelligence": 14},
    )
    skills.append(arcane)

    for school, desc in [
        ("Illusion", "Creating false images and sensations"),
        ("Transmutation", "Changing the nature of things"),
        ("Conjuration", "Summoning creatures and objects"),
        ("Divination", "Seeing the unseen"),
        ("Enchantment", "Influencing minds"),
    ]:
        skill = Skill(
            name=school,
            description=desc,
            category=SkillCategory.ARCANE,
            parent_id=arcane.skill_id,
            depth=2,
            primary_stat="intelligence",
            level_requirement=15,
        )
        skills.append(skill)

    # =========================================================================
    # CRAFTING SKILLS
    # =========================================================================
    crafting = Skill(
        name="Crafting",
        description="Creating items and goods",
        category=SkillCategory.CRAFTING,
        depth=0,
        primary_stat="intelligence",
        secondary_stat="dexterity",
    )
    skills.append(crafting)

    # Smithing
    smithing = Skill(
        name="Smithing",
        description="Working with metals",
        category=SkillCategory.SMITHING,
        parent_id=crafting.skill_id,
        depth=1,
        primary_stat="strength",
        secondary_stat="intelligence",
    )
    skills.append(smithing)

    for spec, desc in [
        ("Blacksmithing", "General metalwork"),
        ("Weaponsmithing", "Crafting weapons"),
        ("Armorsmithing", "Crafting armor"),
        ("Jewelrycrafting", "Working with precious metals"),
    ]:
        skill = Skill(
            name=spec,
            description=desc,
            category=SkillCategory.SMITHING,
            parent_id=smithing.skill_id,
            depth=2,
            primary_stat="strength",
            secondary_stat="dexterity",
            level_requirement=10,
        )
        skills.append(skill)

    # Alchemy
    alchemy = Skill(
        name="Alchemy",
        description="Creating potions and compounds",
        category=SkillCategory.ALCHEMY,
        parent_id=crafting.skill_id,
        depth=1,
        primary_stat="intelligence",
        secondary_stat="wisdom",
    )
    skills.append(alchemy)

    for spec, desc in [
        ("Potion Brewing", "Creating beneficial potions"),
        ("Poison Making", "Creating harmful substances"),
        ("Transmutation", "Changing base materials"),
    ]:
        skill = Skill(
            name=spec,
            description=desc,
            category=SkillCategory.ALCHEMY,
            parent_id=alchemy.skill_id,
            depth=2,
            primary_stat="intelligence",
            level_requirement=10,
        )
        skills.append(skill)

    # Other crafting
    for craft, desc, stat in [
        ("Woodworking", "Working with wood", "dexterity"),
        ("Leatherworking", "Working with leather", "dexterity"),
        ("Tailoring", "Creating cloth items", "dexterity"),
        ("Cooking", "Preparing food", "wisdom"),
    ]:
        skill = Skill(
            name=craft,
            description=desc,
            category=SkillCategory.CRAFTING,
            parent_id=crafting.skill_id,
            depth=1,
            primary_stat=stat,
        )
        skills.append(skill)

    # =========================================================================
    # GATHERING SKILLS
    # =========================================================================
    gathering = Skill(
        name="Gathering",
        description="Collecting resources from the world",
        category=SkillCategory.GATHERING,
        depth=0,
        primary_stat="constitution",
        secondary_stat="wisdom",
    )
    skills.append(gathering)

    for gather, desc, stat in [
        ("Mining", "Extracting ore and stone", "strength"),
        ("Herbalism", "Collecting plants and herbs", "wisdom"),
        ("Logging", "Harvesting wood", "strength"),
        ("Fishing", "Catching fish", "dexterity"),
        ("Foraging", "Finding food in the wild", "wisdom"),
    ]:
        skill = Skill(
            name=gather,
            description=desc,
            category=SkillCategory.GATHERING,
            parent_id=gathering.skill_id,
            depth=1,
            primary_stat=stat,
        )
        skills.append(skill)

    # =========================================================================
    # SOCIAL SKILLS
    # =========================================================================
    social = Skill(
        name="Social",
        description="Interacting with others",
        category=SkillCategory.SOCIAL,
        depth=0,
        primary_stat="charisma",
    )
    skills.append(social)

    for social_skill, desc in [
        ("Persuasion", "Convincing others through logic and charm"),
        ("Intimidation", "Coercing through fear"),
        ("Deception", "Misleading others"),
        ("Diplomacy", "Negotiating and mediating"),
        ("Leadership", "Inspiring and commanding"),
        ("Insight", "Reading people's intentions"),
        ("Performance", "Entertaining others"),
    ]:
        skill = Skill(
            name=social_skill,
            description=desc,
            category=SkillCategory.SOCIAL,
            parent_id=social.skill_id,
            depth=1,
            primary_stat="charisma",
            secondary_stat="wisdom" if social_skill == "Insight" else None,
        )
        skills.append(skill)

    # =========================================================================
    # KNOWLEDGE SKILLS
    # =========================================================================
    knowledge = Skill(
        name="Knowledge",
        description="Academic and practical knowledge",
        category=SkillCategory.KNOWLEDGE,
        depth=0,
        primary_stat="intelligence",
        is_passive=True,
    )
    skills.append(knowledge)

    for know, desc in [
        ("History", "Knowledge of past events"),
        ("Nature", "Knowledge of the natural world"),
        ("Arcana", "Knowledge of magical theory"),
        ("Religion", "Knowledge of deities and faith"),
        ("Medicine", "Knowledge of healing"),
        ("Engineering", "Knowledge of mechanisms"),
        ("Geography", "Knowledge of places"),
    ]:
        skill = Skill(
            name=know,
            description=desc,
            category=SkillCategory.KNOWLEDGE,
            parent_id=knowledge.skill_id,
            depth=1,
            primary_stat="intelligence",
            is_passive=True,
        )
        skills.append(skill)

    # =========================================================================
    # PHYSICAL SKILLS
    # =========================================================================
    physical = Skill(
        name="Physical",
        description="Physical abilities and athletics",
        category=SkillCategory.PHYSICAL,
        depth=0,
        primary_stat="constitution",
    )
    skills.append(physical)

    for phys, desc, stat in [
        ("Athletics", "Running, jumping, climbing", "strength"),
        ("Acrobatics", "Balance and agility", "dexterity"),
        ("Stealth", "Moving unseen", "dexterity"),
        ("Endurance", "Lasting stamina", "constitution"),
        ("Swimming", "Moving through water", "strength"),
        ("Riding", "Controlling mounts", "dexterity"),
    ]:
        skill = Skill(
            name=phys,
            description=desc,
            category=SkillCategory.PHYSICAL,
            parent_id=physical.skill_id,
            depth=1,
            primary_stat=stat,
        )
        skills.append(skill)

    # =========================================================================
    # SURVIVAL SKILLS
    # =========================================================================
    survival = Skill(
        name="Survival",
        description="Surviving in the wilderness",
        category=SkillCategory.SURVIVAL,
        parent_id=physical.skill_id,
        depth=1,
        primary_stat="wisdom",
    )
    skills.append(survival)

    for surv, desc in [
        ("Tracking", "Following trails"),
        ("Hunting", "Catching game"),
        ("Trapping", "Setting snares"),
        ("Navigation", "Finding your way"),
        ("Shelter Building", "Creating temporary homes"),
    ]:
        skill = Skill(
            name=surv,
            description=desc,
            category=SkillCategory.SURVIVAL,
            parent_id=survival.skill_id,
            depth=2,
            primary_stat="wisdom",
            level_requirement=5,
        )
        skills.append(skill)

    # =========================================================================
    # TRADE SKILLS
    # =========================================================================
    trade = Skill(
        name="Trade",
        description="Commerce and production",
        category=SkillCategory.TRADE,
        depth=0,
        primary_stat="charisma",
        secondary_stat="intelligence",
    )
    skills.append(trade)

    for trade_skill, desc, stat in [
        ("Mercantile", "Buying and selling", "charisma"),
        ("Farming", "Growing crops", "wisdom"),
        ("Animal Husbandry", "Raising animals", "wisdom"),
        ("Brewing", "Making drinks", "intelligence"),
    ]:
        skill = Skill(
            name=trade_skill,
            description=desc,
            category=SkillCategory.TRADE,
            parent_id=trade.skill_id,
            depth=1,
            primary_stat=stat,
        )
        skills.append(skill)

    return skills


def build_skill_tree() -> SkillTree:
    """Build the complete skill tree"""
    tree = SkillTree()
    for skill in get_base_skills():
        tree.add_skill(skill)
    return tree
