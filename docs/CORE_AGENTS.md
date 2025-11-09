# Agent Simulation Layer - Part 1: Core Agent System

## Overview

This document defines the foundational elements of the agent simulation system: core data models, morality and alignment, agent types, and the technical stack. The system transforms procedurally generated worlds into living ecosystems where autonomous agents develop personalities, skills, relationships, and histories.

**Design Philosophy:** "Agents have ultimate autonomy to develop and thrive in a world they directly impact."

---

## Table of Contents

1. [Core Data Models](#core-data-models)
2. [Morality & Alignment System](#morality--alignment-system)
3. [Agent Types & Architecture](#agent-types--architecture)
4. [Tech Stack](#tech-stack)
5. [Agent Lifecycle](#agent-lifecycle)
6. [Genetics & Procreation](#genetics--procreation)

---

## Core Data Models

### CoreStats

Physical and mental attributes that govern agent capabilities and skill performance.

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from uuid import UUID
from datetime import datetime
import numpy as np

class CoreStats(BaseModel):
    """
    Core physical/mental stats (1-20 scale)
    
    These stats:
    - Are inherited genetically from parents
    - Develop toward genetic potential through childhood
    - Can be improved through relevant skill usage
    - Act as modifiers for skill success rates
    """
    
    # Physical Stats
    strength: int = Field(default=10, ge=1, le=20)
    """Physical power - affects melee combat, carrying capacity, physical labor"""
    
    dexterity: int = Field(default=10, ge=1, le=20)
    """Agility and precision - affects ranged combat, stealth, crafting precision"""
    
    constitution: int = Field(default=10, ge=1, le=20)
    """Endurance and health - affects stamina, disease resistance, survival"""
    
    # Mental Stats
    intelligence: int = Field(default=10, ge=1, le=20)
    """Learning and reasoning - affects magic, crafting, knowledge skills"""
    
    wisdom: int = Field(default=10, ge=1, le=20)
    """Perception and intuition - affects survival, medicine, social reading"""
    
    charisma: int = Field(default=10, ge=1, le=20)
    """Social influence - affects persuasion, leadership, trading"""
    
    def get_stat_modifier(self, stat_name: str) -> float:
        """
        Calculate modifier for a stat (used in skill checks)
        
        Formula: (stat_value - 10) * 0.02
        - 10 is baseline (0% modifier)
        - Each point above/below 10 = Â±2% success chance
        - Range: -18% (stat=1) to +20% (stat=20)
        """
        stat_value = getattr(self, stat_name.lower(), 10)
        return (stat_value - 10) * 0.02
    
    def improve_stat(self, stat_name: str, amount: float = 0.1):
        """
        Improve a stat through use
        
        Stats improve slowly through skill usage
        More improvement at lower levels, diminishing returns at higher levels
        """
        current_value = getattr(self, stat_name.lower())
        
        # Diminishing returns - harder to improve at higher levels
        improvement_chance = 1.0 - (current_value / 20) * 0.7
        
        if np.random.random() < improvement_chance * amount:
            new_value = min(current_value + 1, 20)
            setattr(self, stat_name.lower(), new_value)
            return True
        return False
```

### CoreNeeds

Survival needs that motivate agent behavior.

```python
class CoreNeeds(BaseModel):
    """
    Basic survival needs (0.0 = satisfied, 1.0 = critical)
    
    These needs:
    - Increase over time automatically
    - Drive agent goal selection
    - Affect agent performance when unfulfilled
    - Create realistic survival behavior
    """
    
    hunger: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for food - increases 0.05/hour"""
    
    thirst: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for water - increases 0.08/hour (faster than hunger)"""
    
    rest: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for sleep - increases 0.03/hour"""
    
    warmth: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for appropriate temperature - environment dependent"""
    
    safety: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for security - increases near threats"""
    
    social: float = Field(default=0.0, ge=0.0, le=1.0)
    """Need for companionship - varies by personality"""
    
    def get_most_urgent(self) -> tuple[str, float]:
        """Get the most urgent need"""
        needs = {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "rest": self.rest,
            "warmth": self.warmth,
            "safety": self.safety,
            "social": self.social
        }
        return max(needs.items(), key=lambda x: x[1])
    
    def update_over_time(self, hours: float, environment: Optional[Dict] = None):
        """
        Update needs based on time passage and environment
        
        Args:
            hours: Time elapsed
            environment: Optional environment data affecting needs
        """
        self.hunger = min(self.hunger + 0.05 * hours, 1.0)
        self.thirst = min(self.thirst + 0.08 * hours, 1.0)
        self.rest = min(self.rest + 0.03 * hours, 1.0)
        
        # Environment affects warmth
        if environment:
            temp = environment.get("temperature", 20)
            if temp < 10 or temp > 30:
                self.warmth = min(self.warmth + 0.04 * hours, 1.0)
            else:
                self.warmth = max(self.warmth - 0.02 * hours, 0.0)
    
    def get_performance_penalty(self) -> float:
        """
        Calculate performance penalty from unfulfilled needs
        
        Returns: Multiplier (0.5 = half performance, 1.0 = full performance)
        """
        # Average of needs, weighted by severity
        avg_need = (self.hunger + self.thirst * 1.5 + self.rest) / 3.5
        
        # Penalty scales exponentially
        if avg_need > 0.8:
            return 0.5  # Severe penalty
        elif avg_need > 0.6:
            return 0.7
        elif avg_need > 0.4:
            return 0.85
        else:
            return 1.0  # No penalty
```

---

## Morality & Alignment System

Agents have moral alignments that affect their decision-making and behavior patterns. This models realistic human behavioral variation, from altruistic heroes to dangerous criminals.

### Alignment Axes

```python
from enum import Enum

class MoralAlignment(str, Enum):
    """Good-Evil axis: How the agent treats others"""
    GOOD = "good"           # Altruistic, helpful, protective
    NEUTRAL = "neutral"     # Self-interested, pragmatic
    EVIL = "evil"          # Harmful, exploitative, cruel

class EthicalAlignment(str, Enum):
    """Lawful-Chaotic axis: How the agent views rules and order"""
    LAWFUL = "lawful"      # Respects authority, follows rules, organized
    NEUTRAL = "neutral"    # Flexible, adapts to situation
    CHAOTIC = "chaotic"    # Rebels against authority, unpredictable

class Alignment(BaseModel):
    """
    Combined alignment system (9 possible alignments)
    
    Examples:
    - Lawful Good: Noble paladin, honorable leader
    - Chaotic Evil: Bandit, psychopath, destroyer
    - True Neutral: Survivalist, merchant, pragmatist
    - Lawful Evil: Tyrant, corrupt official, slaver
    - Chaotic Good: Robin Hood, rebel with a cause
    """
    
    moral_axis: MoralAlignment = MoralAlignment.NEUTRAL
    ethical_axis: EthicalAlignment = EthicalAlignment.NEUTRAL
    
    # Numeric scores for finer granularity (-1.0 to 1.0)
    good_evil_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    """
    -1.0 = Pure Evil (actively seeks to harm)
     0.0 = Neutral (self-interested)
     1.0 = Pure Good (selflessly helps others)
    """
    
    lawful_chaotic_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    """
    -1.0 = Pure Chaos (rejects all rules)
     0.0 = Neutral (pragmatic)
     1.0 = Pure Law (rigid adherence to rules)
    """
    
    @property
    def alignment_string(self) -> str:
        """Get traditional alignment string (e.g., 'Chaotic Evil')"""
        if abs(self.lawful_chaotic_score) < 0.3 and abs(self.good_evil_score) < 0.3:
            return "True Neutral"
        
        ethical = (
            "Lawful" if self.lawful_chaotic_score > 0.3 else
            "Chaotic" if self.lawful_chaotic_score < -0.3 else
            ""
        )
        
        moral = (
            "Good" if self.good_evil_score > 0.3 else
            "Evil" if self.good_evil_score < -0.3 else
            "Neutral"
        )
        
        if ethical:
            return f"{ethical} {moral}"
        return moral
    
    def shift_alignment(
        self,
        action_type: str,
        magnitude: float = 0.05
    ):
        """
        Shift alignment based on actions
        
        Examples:
        - Helping stranger â†’ shift toward Good
        - Stealing â†’ shift toward Chaotic
        - Following law even when costly â†’ shift toward Lawful
        - Harming innocent â†’ shift toward Evil
        """
        shifts = {
            "help_stranger": (magnitude, 0),
            "harm_innocent": (-magnitude * 2, 0),  # Stronger shift for evil
            "murder": (-magnitude * 3, 0),
            "steal": (0, -magnitude),
            "follow_law_costly": (0, magnitude),
            "break_law_for_good": (magnitude * 0.5, -magnitude * 0.5),
            "sacrifice_for_others": (magnitude * 2, 0),
            "betray_trust": (-magnitude * 1.5, 0),
            "keep_promise": (magnitude * 0.3, magnitude * 0.3),
            "lie_for_gain": (-magnitude * 0.5, -magnitude * 0.5)
        }
        
        if action_type in shifts:
            good_shift, law_shift = shifts[action_type]
            
            self.good_evil_score = np.clip(
                self.good_evil_score + good_shift,
                -1.0, 1.0
            )
            
            self.lawful_chaotic_score = np.clip(
                self.lawful_chaotic_score + law_shift,
                -1.0, 1.0
            )
            
            # Update categorical alignment
            self._update_categorical()
    
    def _update_categorical(self):
        """Update categorical alignment based on scores"""
        if self.good_evil_score > 0.4:
            self.moral_axis = MoralAlignment.GOOD
        elif self.good_evil_score < -0.4:
            self.moral_axis = MoralAlignment.EVIL
        else:
            self.moral_axis = MoralAlignment.NEUTRAL
        
        if self.lawful_chaotic_score > 0.4:
            self.ethical_axis = EthicalAlignment.LAWFUL
        elif self.lawful_chaotic_score < -0.4:
            self.ethical_axis = EthicalAlignment.CHAOTIC
        else:
            self.ethical_axis = EthicalAlignment.NEUTRAL
    
    def get_harm_likelihood(self, context: Dict[str, Any]) -> float:
        """
        Calculate likelihood of harming another agent
        
        Factors:
        - Evil alignment increases likelihood
        - Chaotic alignment adds unpredictability
        - Context matters (provoked? threatened?)
        
        Returns: Probability 0.0-1.0
        """
        base_likelihood = 0.05  # Everyone has 5% base chance under stress
        
        # Evil alignment increases harm likelihood
        evil_multiplier = 1.0
        if self.good_evil_score < -0.3:
            evil_multiplier = 2.0 + abs(self.good_evil_score) * 3
        elif self.good_evil_score < 0:
            evil_multiplier = 1.5
        
        # Chaotic adds randomness
        chaos_factor = 1.0
        if self.lawful_chaotic_score < -0.3:
            chaos_factor = 1.0 + abs(self.lawful_chaotic_score) * 0.5
        
        # Context modifiers
        provoked = context.get("provoked", False)
        threatened = context.get("threatened", False)
        desperate = context.get("desperate", False)
        
        context_multiplier = 1.0
        if threatened:
            context_multiplier = 4.0  # High threat â†’ more likely to harm
        elif provoked:
            context_multiplier = 2.0
        elif desperate:
            context_multiplier = 1.5
        
        # Good alignment reduces likelihood even when provoked
        if self.good_evil_score > 0.3:
            context_multiplier *= 0.5
        
        likelihood = base_likelihood * evil_multiplier * chaos_factor * context_multiplier
        
        return min(likelihood, 0.95)  # Cap at 95%
    
    def should_cooperate(self, relationship_strength: float) -> bool:
        """
        Decide whether to cooperate with another agent
        
        Good agents cooperate more readily
        Evil agents cooperate only when beneficial
        """
        base_cooperation = 0.5
        
        # Alignment modifiers
        alignment_modifier = self.good_evil_score * 0.3  # Â±30%
        
        # Relationship modifier
        relationship_modifier = relationship_strength * 0.4  # up to Â±40%
        
        cooperation_chance = base_cooperation + alignment_modifier + relationship_modifier
        
        return np.random.random() < cooperation_chance
```

### Genetic Inheritance of Alignment

```python
class AlignmentGenetics:
    """Alignment has genetic and environmental components"""
    
    @staticmethod
    def inherit_alignment(
        parent1_alignment: Alignment,
        parent2_alignment: Alignment
    ) -> Alignment:
        """
        Children inherit alignment tendencies from parents
        
        Nature (60%) + Nurture (40%)
        - Parents' average provides baseline
        - Random variance added
        - Environmental factors will shift during upbringing
        """
        # Average parents' scores
        avg_good_evil = (
            parent1_alignment.good_evil_score + 
            parent2_alignment.good_evil_score
        ) / 2
        
        avg_lawful_chaotic = (
            parent1_alignment.lawful_chaotic_score + 
            parent2_alignment.lawful_chaotic_score
        ) / 2
        
        # Add variance (regression to mean)
        variance = 0.3
        good_evil_score = np.clip(
            avg_good_evil * 0.6 + np.random.normal(0, variance),
            -1.0, 1.0
        )
        
        lawful_chaotic_score = np.clip(
            avg_lawful_chaotic * 0.6 + np.random.normal(0, variance),
            -1.0, 1.0
        )
        
        offspring_alignment = Alignment(
            good_evil_score=good_evil_score,
            lawful_chaotic_score=lawful_chaotic_score
        )
        
        offspring_alignment._update_categorical()
        
        return offspring_alignment
```

---

## Agent Types & Architecture

### AgentType Enumeration

```python
class AgentType(str, Enum):
    """Types of agents in the simulation"""
    
    # Individual agents
    HUMAN = "human"              # Intelligent individual (full reasoning)
    ANIMAL = "animal"            # Fauna with simple behavior
    PLANT = "plant"              # Flora (mostly procedural)
    
    # Collective agents
    SETTLEMENT = "settlement"    # Village, town, city
    GUILD = "guild"             # Professional organization
    COUNCIL = "council"         # Governing body
    TRIBE = "tribe"             # Nomadic group
    NATION = "nation"           # Large-scale civilization
    RELIGION = "religion"       # Belief system collective
    
    # Special agents
    SKILL_ARCHITECT = "skill_architect"  # Manages skill creation
```

### AgentState - Core Data Model

```python
class AgentState(BaseModel):
    """
    Core state for any agent
    
    This is the primary data structure representing an agent's
    current state, stored in PostgreSQL and loaded into memory
    when the agent is active.
    """
    
    # Identity
    agent_id: UUID = Field(default_factory=uuid.uuid4)
    agent_type: AgentType
    name: str
    
    # Location
    world_id: UUID
    position_x: int
    position_y: int
    chunk_id: Optional[str] = None
    
    # Core Stats
    stats: CoreStats = Field(default_factory=CoreStats)
    
    # Core Needs
    needs: CoreNeeds = Field(default_factory=CoreNeeds)
    
    # Alignment & Morality
    alignment: Alignment = Field(default_factory=Alignment)
    
    # Health and Status
    is_alive: bool = True
    health: float = Field(default=1.0, ge=0.0, le=1.0)
    max_health: float = 1.0
    stamina: float = Field(default=1.0, ge=0.0, le=1.0)
    age_days: int = 0
    
    # Memory & History
    memory_summary: Optional[str] = None  # Latest summary of memories
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Skills (will be populated from agent_skills table)
    skills: Dict[str, "SkillLevel"] = {}  # skill_id -> level info
    
    # Inventory
    inventory: Dict[str, int] = {}  # item_type -> quantity
    equipped_items: Dict[str, str] = {}  # slot -> item_type
    
    # Relationships
    faction_id: Optional[UUID] = None
    parent_agent_id: Optional[UUID] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    generation: int = 0
    
    # Agent-specific data (extensible JSON field)
    custom_state: Dict[str, Any] = {}
    
    def get_skill_level(self, skill_id: str) -> int:
        """Get level in a specific skill"""
        if skill_id not in self.skills:
            return 0
        return self.skills[skill_id].level
    
    def get_skill_bonus(self, skill_id: str) -> float:
        """Get proficiency bonus for a skill (0.0 to 0.5)"""
        if skill_id not in self.skills:
            return 0.0
        return self.skills[skill_id].proficiency_bonus
    
    def has_item(self, item_type: str, quantity: int = 1) -> bool:
        """Check if agent has item"""
        return self.inventory.get(item_type, 0) >= quantity
    
    def add_item(self, item_type: str, quantity: int = 1):
        """Add item to inventory"""
        self.inventory[item_type] = self.inventory.get(item_type, 0) + quantity
    
    def remove_item(self, item_type: str, quantity: int = 1) -> bool:
        """Remove item from inventory, return success"""
        if not self.has_item(item_type, quantity):
            return False
        self.inventory[item_type] -= quantity
        if self.inventory[item_type] <= 0:
            del self.inventory[item_type]
        return True
    
    def is_incapacitated(self) -> bool:
        """Check if agent is too weak to act"""
        return (
            self.health < 0.2 or
            self.stamina < 0.1 or
            self.needs.hunger > 0.9 or
            self.needs.thirst > 0.9
        )
    
    def get_effective_stats(self) -> CoreStats:
        """Get stats modified by current condition"""
        effective = self.stats.model_copy()
        
        # Performance penalty from unfulfilled needs
        penalty = self.needs.get_performance_penalty()
        
        if penalty < 1.0:
            # Reduce all stats proportionally
            for stat_name in ["strength", "dexterity", "constitution",
                             "intelligence", "wisdom", "charisma"]:
                current = getattr(effective, stat_name)
                reduced = max(1, int(current * penalty))
                setattr(effective, stat_name, reduced)
        
        return effective
```

---

## Genetics & DNA System

### DNA Encoding

Agents have a simplified DNA system with ~35 genetic markers per trait. Each marker contributes to the trait's potential value.

```python
class GeneMarker(BaseModel):
    """
    Single genetic marker (like a gene locus)
    
    Each marker has two alleles (one from each parent)
    and contributes to the trait's final value
    """
    allele_1: int = Field(ge=-2, le=2, description="Allele from parent 1")
    allele_2: int = Field(ge=-2, le=2, description="Allele from parent 2")
    dominance: str = Field(default="additive", description="How alleles combine")
    
    def express(self) -> int:
        """
        Calculate expressed value from alleles
        
        - Additive: Both alleles contribute equally
        - Dominant: Stronger allele dominates
        - Recessive: Weaker allele expresses
        """
        if self.dominance == "additive":
            return self.allele_1 + self.allele_2
        elif self.dominance == "dominant":
            return max(abs(self.allele_1), abs(self.allele_2))
        else:  # recessive
            return min(abs(self.allele_1), abs(self.allele_2))

class TraitGenome(BaseModel):
    """
    Genetic encoding for a single trait
    
    Contains 35 markers that collectively determine
    the genetic potential for this trait
    """
    trait_name: str
    markers: List[GeneMarker] = Field(default_factory=list)
    base_value: float = Field(description="Baseline value before genetic contribution")
    marker_weight: float = Field(default=0.05, description="How much each marker affects trait")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize 35 markers if not provided
        if not self.markers:
            self.markers = [
                GeneMarker(
                    allele_1=np.random.randint(-2, 3),
                    allele_2=np.random.randint(-2, 3),
                    dominance=np.random.choice(["additive", "dominant", "recessive"])
                )
                for _ in range(35)
            ]
    
    def calculate_potential(self) -> float:
        """
        Calculate genetic potential from all markers
        
        Formula: base_value + sum(marker_expression * marker_weight)
        """
        genetic_contribution = sum(
            marker.express() * self.marker_weight 
            for marker in self.markers
        )
        
        return self.base_value + genetic_contribution
    
    @staticmethod
    def create_from_parents(
        parent1_trait: "TraitGenome",
        parent2_trait: "TraitGenome",
        mutation_rate: float = 0.01
    ) -> "TraitGenome":
        """
        Create offspring trait through recombination
        
        - Takes one allele from each parent per marker
        - Small chance of mutation (change allele value)
        """
        offspring_markers = []
        
        for i in range(35):
            # Get markers from parents
            p1_marker = parent1_trait.markers[i]
            p2_marker = parent2_trait.markers[i]
            
            # Randomly select one allele from each parent
            allele_1 = np.random.choice([p1_marker.allele_1, p1_marker.allele_2])
            allele_2 = np.random.choice([p2_marker.allele_1, p2_marker.allele_2])
            
            # Mutation chance
            if np.random.random() < mutation_rate:
                allele_1 += np.random.randint(-1, 2)
                allele_1 = np.clip(allele_1, -2, 2)
            
            if np.random.random() < mutation_rate:
                allele_2 += np.random.randint(-1, 2)
                allele_2 = np.clip(allele_2, -2, 2)
            
            # Inherit dominance pattern (randomly from parents)
            dominance = np.random.choice([p1_marker.dominance, p2_marker.dominance])
            
            offspring_markers.append(GeneMarker(
                allele_1=allele_1,
                allele_2=allele_2,
                dominance=dominance
            ))
        
        return TraitGenome(
            trait_name=parent1_trait.trait_name,
            markers=offspring_markers,
            base_value=parent1_trait.base_value,
            marker_weight=parent1_trait.marker_weight
        )

class AgentGenome(BaseModel):
    """
    Complete genetic code for an agent
    
    Contains all genetic information that determines:
    - Stat potentials (6 core stats)
    - Physical appearance (height, coloring, etc.)
    - Inherited skill aptitudes
    """
    genome_id: UUID = Field(default_factory=uuid.uuid4)
    
    # Stat potentials (35 markers each)
    strength_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="strength",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    dexterity_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="dexterity",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    constitution_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="constitution",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    intelligence_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="intelligence",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    wisdom_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="wisdom",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    charisma_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="charisma",
            base_value=10.0,
            marker_weight=0.05
        )
    )
    
    # Physical trait genes (35 markers each)
    height_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="height",
            base_value=170.0,  # cm
            marker_weight=1.0   # Each marker = Â±1 cm
        )
    )
    weight_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="weight",
            base_value=70.0,  # kg
            marker_weight=0.5  # Each marker = Â±0.5 kg
        )
    )
    
    # Color genes (encoded as RGB-like values)
    skin_tone_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="skin_tone",
            base_value=150.0,  # Melanin level (0-255)
            marker_weight=2.0
        )
    )
    eye_color_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="eye_color",
            base_value=100.0,  # Color spectrum
            marker_weight=2.0
        )
    )
    hair_color_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="hair_color",
            base_value=50.0,  # Color spectrum
            marker_weight=2.0
        )
    )
    
    # Alignment predisposition genes
    empathy_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="empathy",
            base_value=0.0,  # Good-evil predisposition
            marker_weight=0.01
        )
    )
    order_genes: TraitGenome = Field(
        default_factory=lambda: TraitGenome(
            trait_name="order",
            base_value=0.0,  # Lawful-chaotic predisposition
            marker_weight=0.01
        )
    )
    
    # Skill aptitude genes (inherited faster learning)
    skill_aptitudes: Dict[str, float] = Field(
        default_factory=dict,
        description="Skill category -> learning rate multiplier (0.8-1.2)"
    )
    
    def calculate_stat_potential(self, stat_name: str) -> int:
        """Calculate genetic potential for a stat (8-20 range)"""
        trait_genome = getattr(self, f"{stat_name}_genes")
        potential = trait_genome.calculate_potential()
        
        # Clamp to valid stat range
        return int(np.clip(potential, 8, 20))
    
    def calculate_physical_trait(self, trait_name: str) -> float:
        """Calculate value for a physical trait"""
        trait_genome = getattr(self, f"{trait_name}_genes")
        return trait_genome.calculate_potential()
    
    @staticmethod
    def create_from_parents(
        parent1_genome: "AgentGenome",
        parent2_genome: "AgentGenome",
        mutation_rate: float = 0.01
    ) -> "AgentGenome":
        """
        Create offspring genome through sexual recombination
        
        Takes 50% genetic material from each parent
        Small chance of mutations
        """
        offspring = AgentGenome()
        
        # Recombine stat genes
        for stat in ["strength", "dexterity", "constitution", 
                     "intelligence", "wisdom", "charisma"]:
            parent1_trait = getattr(parent1_genome, f"{stat}_genes")
            parent2_trait = getattr(parent2_genome, f"{stat}_genes")
            
            offspring_trait = TraitGenome.create_from_parents(
                parent1_trait,
                parent2_trait,
                mutation_rate
            )
            
            setattr(offspring, f"{stat}_genes", offspring_trait)
        
        # Recombine physical trait genes
        for trait in ["height", "weight", "skin_tone", "eye_color", "hair_color"]:
            parent1_trait = getattr(parent1_genome, f"{trait}_genes")
            parent2_trait = getattr(parent2_genome, f"{trait}_genes")
            
            offspring_trait = TraitGenome.create_from_parents(
                parent1_trait,
                parent2_trait,
                mutation_rate
            )
            
            setattr(offspring, f"{trait}_genes", offspring_trait)
        
        # Recombine alignment predisposition
        for trait in ["empathy", "order"]:
            parent1_trait = getattr(parent1_genome, f"{trait}_genes")
            parent2_trait = getattr(parent2_genome, f"{trait}_genes")
            
            offspring_trait = TraitGenome.create_from_parents(
                parent1_trait,
                parent2_trait,
                mutation_rate
            )
            
            setattr(offspring, f"{trait}_genes", offspring_trait)
        
        # Inherit skill aptitudes (average of parents with variance)
        all_skills = set(parent1_genome.skill_aptitudes.keys()) | \
                     set(parent2_genome.skill_aptitudes.keys())
        
        for skill_cat in all_skills:
            p1_apt = parent1_genome.skill_aptitudes.get(skill_cat, 1.0)
            p2_apt = parent2_genome.skill_aptitudes.get(skill_cat, 1.0)
            
            # Average with small random variation
            avg_aptitude = (p1_apt + p2_apt) / 2
            variance = np.random.normal(0, 0.05)
            offspring.skill_aptitudes[skill_cat] = np.clip(
                avg_aptitude + variance,
                0.8, 1.2
            )
        
        return offspring
    
    @staticmethod
    def generate_random(biological_sex: Optional[str] = None) -> "AgentGenome":
        """
        Generate random genome for initial population
        
        Used for first-generation agents (not born from parents)
        """
        genome = AgentGenome()
        
        # All traits already initialized with random markers
        # Just need to ensure they're in valid ranges
        
        return genome
```

---

## Physical Attributes

Agents have detailed physical characteristics determined by genetics.

```python
class PhysicalAttributes(BaseModel):
    """
    Physical appearance and body characteristics
    
    All values determined by genetics at birth,
    though some (like weight) can change over time
    """
    
    # Biological
    biological_sex: str = Field(description="'male' or 'female'")
    
    # Dimensions
    height_cm: float = Field(ge=120.0, le=220.0, description="Height in centimeters")
    weight_kg: float = Field(ge=30.0, le=200.0, description="Weight in kilograms")
    
    # Body composition
    muscle_mass_percent: float = Field(default=35.0, ge=15.0, le=55.0)
    body_fat_percent: float = Field(default=20.0, ge=5.0, le=50.0)
    
    # Appearance
    skin_tone: int = Field(ge=0, le=255, description="Melanin level (0=lightest, 255=darkest)")
    eye_color: str = Field(description="Eye color descriptor")
    hair_color: str = Field(description="Hair color descriptor")
    hair_type: str = Field(default="straight", description="straight, wavy, curly, coily")
    
    @staticmethod
    def from_genome(
        genome: AgentGenome,
        biological_sex: str,
        age_days: int = 0
    ) -> "PhysicalAttributes":
        """
        Generate physical attributes from genome
        """
        # Calculate from genes
        base_height = genome.calculate_physical_trait("height")
        base_weight = genome.calculate_physical_trait("weight")
        
        # Sexual dimorphism adjustments
        if biological_sex == "male":
            height_cm = base_height * 1.08  # Males ~8% taller on average
            weight_kg = base_weight * 1.15  # Males ~15% heavier on average
            muscle_mass = 35.0 + np.random.normal(5, 3)
            body_fat = 18.0 + np.random.normal(3, 2)
        else:
            height_cm = base_height
            weight_kg = base_weight
            muscle_mass = 28.0 + np.random.normal(4, 2)
            body_fat = 25.0 + np.random.normal(4, 3)
        
        # Age adjustments (children are smaller)
        if age_days < 6570:  # Under 18 years
            age_factor = age_days / 6570
            height_cm *= (0.5 + 0.5 * age_factor)
            weight_kg *= (0.3 + 0.7 * age_factor)
        
        # Color traits from genes
        skin_tone = int(np.clip(genome.calculate_physical_trait("skin_tone"), 0, 255))
        
        # Convert eye color value to descriptor
        eye_value = genome.calculate_physical_trait("eye_color")
        if eye_value < 50:
            eye_color = "brown"
        elif eye_value < 100:
            eye_color = "hazel"
        elif eye_value < 150:
            eye_color = "green"
        elif eye_value < 200:
            eye_color = "blue"
        else:
            eye_color = "gray"
        
        # Convert hair color value to descriptor
        hair_value = genome.calculate_physical_trait("hair_color")
        if hair_value < 40:
            hair_color = "black"
        elif hair_value < 80:
            hair_color = "dark brown"
        elif hair_value < 120:
            hair_color = "brown"
        elif hair_value < 160:
            hair_color = "light brown"
        elif hair_value < 200:
            hair_color = "blonde"
        else:
            hair_color = "red"
        
        # Hair type (genetic but simplified)
        hair_type = np.random.choice(["straight", "wavy", "curly", "coily"])
        
        return PhysicalAttributes(
            biological_sex=biological_sex,
            height_cm=height_cm,
            weight_kg=weight_kg,
            muscle_mass_percent=np.clip(muscle_mass, 15, 55),
            body_fat_percent=np.clip(body_fat, 5, 50),
            skin_tone=skin_tone,
            eye_color=eye_color,
            hair_color=hair_color,
            hair_type=hair_type
        )
    
    def get_bmi(self) -> float:
        """Calculate Body Mass Index"""
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m ** 2)
    
    def get_strength_modifier(self) -> float:
        """Physical strength modifier based on body composition"""
        # More muscle = more strength
        muscle_factor = (self.muscle_mass_percent - 25) / 30  # Normalized
        return 1.0 + muscle_factor * 0.3  # Up to Â±30%
    
    def get_dexterity_modifier(self) -> float:
        """Dexterity modifier based on body composition"""
        # Lower body fat = more dexterous
        fat_factor = (35 - self.body_fat_percent) / 30
        return 1.0 + fat_factor * 0.2  # Up to Â±20%

class BodyHealth(BaseModel):
    """
    Health status of body parts and systems
    
    Values range from 0.0 (missing/destroyed) to 1.0 (perfect health)
    Injuries and conditions reduce these values, affecting performance
    """
    
    # Limbs
    left_arm_health: float = Field(default=1.0, ge=0.0, le=1.0)
    right_arm_health: float = Field(default=1.0, ge=0.0, le=1.0)
    left_leg_health: float = Field(default=1.0, ge=0.0, le=1.0)
    right_leg_health: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Vital organs
    heart_health: float = Field(default=1.0, ge=0.0, le=1.0)
    lung_health: float = Field(default=1.0, ge=0.0, le=1.0)
    liver_health: float = Field(default=1.0, ge=0.0, le=1.0)
    kidney_health: float = Field(default=1.0, ge=0.0, le=1.0)
    stomach_health: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Senses
    vision: float = Field(default=1.0, ge=0.0, le=1.0)
    hearing: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Conditions
    injuries: List[str] = Field(default_factory=list, description="Active injury descriptions")
    diseases: List[str] = Field(default_factory=list, description="Active diseases")
    
    def is_missing_limb(self, limb: str) -> bool:
        """Check if a limb is completely missing"""
        limb_health = getattr(self, f"{limb}_health", 1.0)
        return limb_health < 0.1
    
    def get_skill_modifier(self, skill_id: str) -> float:
        """
        Calculate skill performance modifier based on body health
        
        Different skills require different body parts:
        - Archery: Both arms, good vision
        - Sword fighting: At least one good arm
        - Running: Both legs
        - Reading: Good vision
        - Listening: Good hearing
        """
        modifier = 1.0
        
        # Archery and bow skills require both arms
        if any(x in skill_id for x in ["archery", "bow", "crossbow"]):
            arm_health = min(self.left_arm_health, self.right_arm_health)
            vision_health = self.vision
            
            if arm_health < 0.5:
                modifier *= 0.1  # Severe penalty if missing/damaged arm
            else:
                modifier *= arm_health
            
            modifier *= (0.5 + 0.5 * vision_health)  # Vision contributes 50%
        
        # Melee combat needs at least one good arm
        elif any(x in skill_id for x in ["melee", "sword", "axe", "spear", "unarmed"]):
            best_arm = max(self.left_arm_health, self.right_arm_health)
            
            if best_arm < 0.3:
                modifier *= 0.2  # Can barely fight with bad arms
            else:
                modifier *= (0.5 + 0.5 * best_arm)
        
        # Shield use requires good off-hand
        elif "shield" in skill_id:
            worst_arm = min(self.left_arm_health, self.right_arm_health)
            modifier *= worst_arm
        
        # Athletics and running require legs
        elif any(x in skill_id for x in ["running", "athletics", "climbing", "jumping"]):
            leg_health = min(self.left_leg_health, self.right_leg_health)
            
            if leg_health < 0.3:
                modifier *= 0.1  # Crippled
            else:
                modifier *= leg_health
        
        # Swimming needs arms and legs
        elif "swimming" in skill_id:
            avg_limb_health = (
                self.left_arm_health + self.right_arm_health +
                self.left_leg_health + self.right_leg_health
            ) / 4
            modifier *= avg_limb_health
        
        # Perception skills require senses
        elif any(x in skill_id for x in ["perception", "tracking", "navigation"]):
            sense_health = (self.vision + self.hearing) / 2
            modifier *= (0.7 + 0.3 * sense_health)
        
        # Reading requires vision
        elif any(x in skill_id for x in ["reading", "writing", "knowledge"]):
            modifier *= (0.5 + 0.5 * self.vision)
        
        # Fine motor skills (crafting) require good arms and vision
        elif any(x in skill_id for x in ["crafting", "smithing", "carving", "alchemy"]):
            best_arm = max(self.left_arm_health, self.right_arm_health)
            modifier *= (best_arm + self.vision) / 2
        
        # Organ health affects stamina-based activities
        cardio_health = (self.heart_health + self.lung_health) / 2
        if cardio_health < 0.7:
            modifier *= (0.5 + 0.5 * cardio_health)
        
        return np.clip(modifier, 0.05, 1.0)  # At least 5%, max 100%
    
    def apply_injury(
        self,
        injury_type: str,
        severity: float,
        target: Optional[str] = None
    ):
        """
        Apply an injury to the agent
        
        Args:
            injury_type: Type of injury (cut, broken_bone, disease, etc.)
            severity: How severe (0.0-1.0)
            target: Specific body part if applicable
        """
        if target:
            # Targeted injury
            current_health = getattr(self, f"{target}_health", 1.0)
            new_health = max(0.0, current_health - severity)
            setattr(self, f"{target}_health", new_health)
            
            self.injuries.append(f"{injury_type} to {target} (severity: {severity:.2f})")
        else:
            # General injury - affects overall health
            # Distribute across organs
            self.heart_health = max(0.0, self.heart_health - severity * 0.2)
            self.lung_health = max(0.0, self.lung_health - severity * 0.2)
            self.liver_health = max(0.0, self.liver_health - severity * 0.2)
            
            self.injuries.append(f"{injury_type} (severity: {severity:.2f})")
    
    def heal(self, amount: float):
        """Gradually heal injuries"""
        # Heal limbs
        self.left_arm_health = min(1.0, self.left_arm_health + amount * 0.1)
        self.right_arm_health = min(1.0, self.right_arm_health + amount * 0.1)
        self.left_leg_health = min(1.0, self.left_leg_health + amount * 0.1)
        self.right_leg_health = min(1.0, self.right_leg_health + amount * 0.1)
        
        # Heal organs (slower)
        self.heart_health = min(1.0, self.heart_health + amount * 0.05)
        self.lung_health = min(1.0, self.lung_health + amount * 0.05)
        self.liver_health = min(1.0, self.liver_health + amount * 0.05)
        self.kidney_health = min(1.0, self.kidney_health + amount * 0.05)
        
        # Heal senses (slowest)
        self.vision = min(1.0, self.vision + amount * 0.02)
        self.hearing = min(1.0, self.hearing + amount * 0.02)
        
        # Remove fully healed injuries
        self.injuries = [
            inj for inj in self.injuries
            if "severity: 0" not in inj
        ]
    
    def get_overall_health(self) -> float:
        """Calculate overall health score"""
        limb_health = (
            self.left_arm_health + self.right_arm_health +
            self.left_leg_health + self.right_leg_health
        ) / 4
        
        organ_health = (
            self.heart_health + self.lung_health +
            self.liver_health + self.kidney_health
        ) / 4
        
        sense_health = (self.vision + self.hearing) / 2
        
        # Weighted average
        return (limb_health * 0.3 + organ_health * 0.5 + sense_health * 0.2)
```

---

## Agent Lifecycle

### Life Stages

```python
class LifeStage(str, Enum):
    """Life stages with different capabilities"""
    INFANT = "infant"          # 0-2 years: helpless, needs care
    CHILD = "child"            # 2-12 years: learning, dependent
    ADOLESCENT = "adolescent"  # 12-18 years: rapid learning, semi-independent
    ADULT = "adult"            # 18-60 years: full capabilities
    ELDER = "elder"            # 60+ years: wisdom, declining physical

class AgentLifecycle(BaseModel):
    """Lifecycle and development information"""
    
    agent_id: UUID
    genome: "Genome"  # Genetic information
    
    # Age and stage
    age_days: int = 0
    birth_date: datetime
    life_stage: LifeStage = LifeStage.INFANT
    
    # Development tracking
    stat_potentials: Dict[str, int] = {}  # Genetic potential (1-20)
    stat_development: Dict[str, float] = {}  # How much realized (0.0-1.0)
    
    # Skill learning bonuses from parents
    inherited_skill_bonuses: Dict[str, float] = {}  # skill_id -> XP multiplier
    
    def update_age(self, days_passed: int):
        """Update age and life stage"""
        self.age_days += days_passed
        
        # Update life stage based on age
        if self.age_days < 730:  # 2 years
            self.life_stage = LifeStage.INFANT
        elif self.age_days < 4380:  # 12 years
            self.life_stage = LifeStage.CHILD
        elif self.age_days < 6570:  # 18 years
            self.life_stage = LifeStage.ADOLESCENT
        elif self.age_days < 21900:  # 60 years
            self.life_stage = LifeStage.ADULT
        else:
            self.life_stage = LifeStage.ELDER
    
    def develop_stats(self, nurture_quality: float = 1.0):
        """
        Gradually realize genetic potential through development
        
        Called during childhood/adolescence
        nurture_quality: 0-1, represents nutrition, care, training
        """
        if self.life_stage not in [LifeStage.INFANT, LifeStage.CHILD, LifeStage.ADOLESCENT]:
            return
        
        # Development rate depends on life stage
        development_rate = {
            LifeStage.INFANT: 0.01,
            LifeStage.CHILD: 0.02,
            LifeStage.ADOLESCENT: 0.03
        }[self.life_stage]
        
        # Apply development to each stat
        for stat_name in ["strength", "dexterity", "constitution",
                         "intelligence", "wisdom", "charisma"]:
            current_dev = self.stat_development.get(stat_name, 0.0)
            if current_dev < 1.0:
                growth = development_rate * nurture_quality
                self.stat_development[stat_name] = min(current_dev + growth, 1.0)
```

---

## Genetics & Procreation

### Genome System

```python
class Gene(BaseModel):
    """Represents a single genetic trait"""
    gene_id: str
    allele_1: str  # From parent 1
    allele_2: str  # From parent 2
    dominance: str  # "dominant", "recessive", "codominant"
    
    def express(self) -> str:
        """Determine expressed phenotype"""
        if self.dominance == "dominant":
            if "D" in [self.allele_1[0], self.allele_2[0]]:
                return next(a for a in [self.allele_1, self.allele_2] if a[0] == "D")
            return self.allele_1
        elif self.dominance == "codominant":
            return f"{self.allele_1}_{self.allele_2}"
        else:  # recessive
            if self.allele_1 == self.allele_2:
                return self.allele_1
            return "default"

class Genome(BaseModel):
    """Complete genetic information for an agent"""
    
    genes: Dict[str, Gene] = {}
    
    # Stat genes (polygenic - multiple genes affect each stat)
    strength_genes: List[str] = []
    dexterity_genes: List[str] = []
    constitution_genes: List[str] = []
    intelligence_genes: List[str] = []
    wisdom_genes: List[str] = []
    charisma_genes: List[str] = []
    
    # Alignment genes (genetic predisposition)
    alignment_genes: Dict[str, str] = {}  # e.g., {"empathy": "D2", "aggression": "r-1"}
    
    # Skill aptitude genes
    skill_aptitudes: Dict[str, float] = {}  # skill_category -> aptitude (0.8-1.2)
    
    def calculate_stat_potential(self, stat_name: str) -> int:
        """
        Calculate genetic potential for a stat (8-20)
        """
        gene_list = getattr(self, f"{stat_name}_genes", [])
        
        if not gene_list:
            return 10  # Average
        
        total = 10  # Base
        for gene_id in gene_list:
            gene = self.genes.get(gene_id)
            if gene:
                expressed = gene.express()
                value = self._parse_allele_value(expressed)
                total += value
        
        return int(np.clip(total + np.random.normal(0, 1), 8, 20))
    
    def _parse_allele_value(self, allele: str) -> int:
        """Extract numeric value from allele string"""
        try:
            return int(allele[1:]) if len(allele) > 1 else 0
        except:
            return 0
    
    @staticmethod
    def create_from_parents(
        parent1_genome: "Genome",
        parent2_genome: "Genome"
    ) -> "Genome":
        """
        Create offspring genome through Mendelian inheritance
        """
        offspring = Genome()
        
        # Combine genes from both parents (simplified)
        all_gene_ids = set(parent1_genome.genes.keys()) | set(parent2_genome.genes.keys())
        
        for gene_id in all_gene_ids:
            p1_gene = parent1_genome.genes.get(gene_id)
            p2_gene = parent2_genome.genes.get(gene_id)
            
            if p1_gene and p2_gene:
                allele_1 = np.random.choice([p1_gene.allele_1, p1_gene.allele_2])
                allele_2 = np.random.choice([p2_gene.allele_1, p2_gene.allele_2])
                
                offspring.genes[gene_id] = Gene(
                    gene_id=gene_id,
                    allele_1=allele_1,
                    allele_2=allele_2,
                    dominance=p1_gene.dominance
                )
        
        # Inherit stat gene lists
        for stat in ["strength", "dexterity", "constitution",
                     "intelligence", "wisdom", "charisma"]:
            p1_list = getattr(parent1_genome, f"{stat}_genes", [])
            p2_list = getattr(parent2_genome, f"{stat}_genes", [])
            combined = list(set(p1_list + p2_list))
            setattr(offspring, f"{stat}_genes", combined)
        
        # Inherit skill aptitudes (average with variance)
        all_skills = set(parent1_genome.skill_aptitudes.keys()) | \
                     set(parent2_genome.skill_aptitudes.keys())
        
        for skill_cat in all_skills:
            p1_apt = parent1_genome.skill_aptitudes.get(skill_cat, 1.0)
            p2_apt = parent2_genome.skill_aptitudes.get(skill_cat, 1.0)
            
            avg_aptitude = (p1_apt + p2_apt) / 2
            variance = np.random.normal(0, 0.05)
            offspring.skill_aptitudes[skill_cat] = np.clip(
                avg_aptitude + variance,
                0.8, 1.2
            )
        
        return offspring
```

---

## Tech Stack

### Core Technologies

- **Python 3.12+** - Primary language
- **FastAPI** - REST API framework
- **LangGraph** - Multi-agent orchestration
- **Ollama** - Local LLM inference
- **Supabase** - PostgreSQL + Storage
- **Redis** - Message queue & caching

### Agent & LLM Libraries

- **langgraph** - Agent orchestration and state management
- **langchain** - LLM abstractions and tool calling
- **langmem** - Long-term memory management
- **langsmith** - Agent evaluation and monitoring
- **ollama** - Local LLM API client

### MCP & Tool Systems

- **mcp** - Model Context Protocol implementation
- **pydantic** - Data validation and tool schemas
- **jsonschema** - Tool parameter validation

### Scientific Computing

- **numpy** - Array operations and statistics
- **scipy** - Advanced algorithms
- **numba** - JIT compilation for performance

### Multi-Processing & Concurrency

- **asyncio** - Async agent execution
- **multiprocessing** - Parallel Ollama instances
- **celery** - Background task processing
- **redis** - Task queue backend

### Memory & State

- **chromadb** - Vector embeddings for semantic memory
- **pgvector** - PostgreSQL vector extension
- **supabase** - ORM for relational data

### Evaluation & Monitoring

- **prometheus-client** - Metrics collection
- **grafana** (optional) - Metrics visualization
- **structlog** - Structured logging

---
## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   World Generation Engine                   │
│                  (Geological Foundation)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              Simulation Orchestrator                        │
│                                                             │
│    ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│    │  Time Loop  │  │ Event Queue  │  │ Resource Mgr   │  │
│    │  Simulator  │  │  & Priority  │  │ (GPU/CPU/Mem)  │  │
│    └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴───────────────────┐
        │                                      │
┌───────▼────────┐                  ┌─────────▼──────────┐
│  Flora System  │                  │   Fauna System     │
│  (Procedural)  │                  │  (Simple Agents)   │
└───────┬────────┘                  └─────────┬──────────┘
        │                                      │
        └──────────────────┬───────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────┐
        │     Civilization Simulation Layer               │
        │                                                 │
        │   ┌─────────────────────────────────────────┐   │
        │   │   LangGraph Multi-Agent System          │   │
        │   │                                         │   │
        │   │  ┌──────────┐    ┌──────────────────┐  │   │
        │   │  │Individual│◄──►│ Collective       │  │   │
        │   │  │  Agents  │    │ Agents           │  │   │
        │   │  │          │    │ (Settlements,    │  │   │
        │   │  │(Humans)  │    │  Guilds)         │  │   │
        │   │  └────┬─────┘    └────┬─────────────┘  │   │
        │   │       │               │                 │   │
        │   │       │    ┌──────────▼──────────┐     │   │
        │   │       │    │  Skill Architect    │     │   │
        │   │       │    │  (Skill Creation)   │     │   │
        │   │       │    └──────────┬──────────┘     │   │
        │   │       │               │                 │   │
        │   │       └───────────────┘                 │   │
        │   │                                         │   │
        │   │     ┌──────────────────────────┐       │   │
        │   │     │   Memory Systems         │       │   │
        │   │     │  - Episodic              │       │   │
        │   │     │  - Semantic              │       │   │
        │   │     │  - Procedural            │       │   │
        │   │     └──────────┬───────────────┘       │   │
        │   │                │                       │   │
        │   │     ┌──────────▼───────────────┐       │   │
        │   │     │  Tool Registry           │       │   │
        │   │     │  (Dynamic MCP)           │       │   │
        │   │     └──────────────────────────┘       │   │
        │   └─────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴───────────────────┐
        │                                      │
┌───────▼────────┐                  ┌─────────▼──────────┐
│  Ollama Pool   │                  │  Event Chronicle   │
│  (3-5 models)  │                  │  (History DB)      │
│  Load Balanced │                  │  PostgreSQL        │
└────────────────┘                  └────────────────────┘
```

---

## Summary

Part 1 establishes the foundation:

✅ **CoreStats** - Six attributes (1-20) governing all actions
✅ **CoreNeeds** - Six survival needs driving behavior
✅ **Alignment System** - Good-Evil and Lawful-Chaotic axes
  - Genetic inheritance with environmental influence
  - Shifts based on actions
  - Affects cooperation and harm likelihood
✅ **AgentState** - Complete agent data model
✅ **Lifecycle** - Age-based development stages
✅ **Genetics** - Mendelian inheritance with regression to mean
✅ **Tech Stack** - Complete tool chain for agent simulation

**Next:** Part 2 will cover the skill system with stat-skill integration and hierarchical navigation.