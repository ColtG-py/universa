"""
Agent State Models
Core data structures for agent state, stats, needs, and alignment
Based on CORE_AGENTS.md specification
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from agents.config import (
    STAT_MIN, STAT_MAX, STAT_DEFAULT,
    NEED_MIN, NEED_MAX,
    ALIGNMENT_MIN, ALIGNMENT_MAX,
    AgentType, LifeStage,
    HUNGER_DECAY_RATE, THIRST_DECAY_RATE,
    REST_DECAY_RATE, SOCIAL_DECAY_RATE
)


class CoreStats(BaseModel):
    """
    D&D-style core statistics for agents.
    Range: 1-20, default 10
    """
    strength: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                          description="Physical power and carrying capacity")
    dexterity: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                           description="Agility, reflexes, and fine motor skills")
    constitution: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                              description="Health, stamina, and resistance")
    intelligence: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                              description="Learning, memory, and reasoning")
    wisdom: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                        description="Perception, insight, and judgment")
    charisma: int = Field(default=STAT_DEFAULT, ge=STAT_MIN, le=STAT_MAX,
                          description="Social influence and force of personality")

    def get_modifier(self, stat_name: str) -> float:
        """
        Calculate stat modifier: (stat - 10) * 0.02
        Returns a modifier between -0.18 and +0.20
        """
        stat_value = getattr(self, stat_name, STAT_DEFAULT)
        return (stat_value - 10) * 0.02

    def get_all_modifiers(self) -> Dict[str, float]:
        """Get modifiers for all stats"""
        return {
            stat: self.get_modifier(stat)
            for stat in ["strength", "dexterity", "constitution",
                         "intelligence", "wisdom", "charisma"]
        }

    @classmethod
    def generate_random(cls, variance: float = 3.0) -> "CoreStats":
        """
        Generate random stats with normal distribution around 10.

        Args:
            variance: Standard deviation for stat generation

        Returns:
            New CoreStats instance with random values
        """
        import random
        def random_stat():
            value = int(random.gauss(10, variance))
            return max(STAT_MIN, min(STAT_MAX, value))

        return cls(
            strength=random_stat(),
            dexterity=random_stat(),
            constitution=random_stat(),
            intelligence=random_stat(),
            wisdom=random_stat(),
            charisma=random_stat()
        )


class CoreNeeds(BaseModel):
    """
    Survival needs for agents.
    Range: 0.0 (satisfied) to 1.0 (critical)
    """
    hunger: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                          description="Need for food (0=full, 1=starving)")
    thirst: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                          description="Need for water (0=hydrated, 1=dehydrated)")
    rest: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                        description="Need for sleep (0=rested, 1=exhausted)")
    warmth: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                          description="Need for warmth (0=comfortable, 1=freezing)")
    safety: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                          description="Feeling of security (0=safe, 1=endangered)")
    social: float = Field(default=0.0, ge=NEED_MIN, le=NEED_MAX,
                          description="Need for social interaction (0=fulfilled, 1=lonely)")

    def update(self, hours_elapsed: float, environment_temp: float = 20.0) -> None:
        """
        Update needs based on time elapsed.

        Args:
            hours_elapsed: Game hours since last update
            environment_temp: Current temperature in Celsius
        """
        self.hunger = min(NEED_MAX, self.hunger + HUNGER_DECAY_RATE * hours_elapsed)
        self.thirst = min(NEED_MAX, self.thirst + THIRST_DECAY_RATE * hours_elapsed)
        self.rest = min(NEED_MAX, self.rest + REST_DECAY_RATE * hours_elapsed)
        self.social = min(NEED_MAX, self.social + SOCIAL_DECAY_RATE * hours_elapsed)

        # Warmth based on environment
        if environment_temp < 10:
            warmth_change = 0.05 * hours_elapsed * ((10 - environment_temp) / 10)
            self.warmth = min(NEED_MAX, self.warmth + warmth_change)
        elif environment_temp > 15:
            warmth_change = 0.02 * hours_elapsed
            self.warmth = max(NEED_MIN, self.warmth - warmth_change)

    def satisfy(self, need_name: str, amount: float) -> None:
        """
        Reduce a need by a given amount.

        Args:
            need_name: Name of the need to satisfy
            amount: Amount to reduce (0.0 to 1.0)
        """
        current = getattr(self, need_name, 0.0)
        setattr(self, need_name, max(NEED_MIN, current - amount))

    def get_most_urgent(self) -> str:
        """Get the name of the most urgent need"""
        needs = {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "rest": self.rest,
            "warmth": self.warmth,
            "safety": self.safety,
            "social": self.social
        }
        return max(needs, key=needs.get)

    def get_critical_needs(self, threshold: float = 0.7) -> List[str]:
        """Get list of needs above the critical threshold"""
        critical = []
        for need_name in ["hunger", "thirst", "rest", "warmth", "safety", "social"]:
            if getattr(self, need_name) >= threshold:
                critical.append(need_name)
        return critical


class Alignment(BaseModel):
    """
    Moral alignment on two axes.
    Good/Evil: -1.0 (pure evil) to +1.0 (pure good)
    Lawful/Chaotic: -1.0 (chaotic) to +1.0 (lawful)
    """
    good_evil: float = Field(default=0.0, ge=ALIGNMENT_MIN, le=ALIGNMENT_MAX,
                             description="-1=evil, 0=neutral, +1=good")
    lawful_chaotic: float = Field(default=0.0, ge=ALIGNMENT_MIN, le=ALIGNMENT_MAX,
                                  description="-1=chaotic, 0=neutral, +1=lawful")

    def get_alignment_name(self) -> str:
        """Get the alignment as a descriptive string"""
        # Determine good/evil axis
        if self.good_evil > 0.33:
            ge_name = "Good"
        elif self.good_evil < -0.33:
            ge_name = "Evil"
        else:
            ge_name = "Neutral"

        # Determine lawful/chaotic axis
        if self.lawful_chaotic > 0.33:
            lc_name = "Lawful"
        elif self.lawful_chaotic < -0.33:
            lc_name = "Chaotic"
        else:
            lc_name = "Neutral"

        # Combine (avoid "Neutral Neutral")
        if ge_name == "Neutral" and lc_name == "Neutral":
            return "True Neutral"
        elif ge_name == "Neutral":
            return lc_name
        elif lc_name == "Neutral":
            return f"Neutral {ge_name}"
        else:
            return f"{lc_name} {ge_name}"

    def shift(self, good_evil_delta: float = 0.0, lawful_chaotic_delta: float = 0.0) -> None:
        """
        Shift alignment based on actions taken.

        Args:
            good_evil_delta: Change to good/evil axis
            lawful_chaotic_delta: Change to lawful/chaotic axis
        """
        self.good_evil = max(ALIGNMENT_MIN, min(ALIGNMENT_MAX,
                                                self.good_evil + good_evil_delta))
        self.lawful_chaotic = max(ALIGNMENT_MIN, min(ALIGNMENT_MAX,
                                                      self.lawful_chaotic + lawful_chaotic_delta))


class PhysicalAttributes(BaseModel):
    """Physical appearance and attributes derived from genetics"""
    height_cm: float = Field(default=170.0, ge=50.0, le=250.0)
    weight_kg: float = Field(default=70.0, ge=20.0, le=300.0)
    hair_color: str = Field(default="brown")
    eye_color: str = Field(default="brown")
    skin_tone: str = Field(default="medium")
    build: str = Field(default="average")  # thin, average, muscular, heavy

    # Calculated attributes
    bmi: Optional[float] = None

    def calculate_bmi(self) -> float:
        """Calculate Body Mass Index"""
        height_m = self.height_cm / 100.0
        self.bmi = self.weight_kg / (height_m * height_m)
        return self.bmi


class AgentState(BaseModel):
    """
    Complete state of an agent in the simulation.
    Combines all agent attributes into a single model.
    """
    # Identity
    agent_id: UUID = Field(default_factory=uuid4)
    world_id: Optional[UUID] = None
    name: str
    agent_type: AgentType = AgentType.HUMAN

    # Location
    position_x: int = 0
    position_y: int = 0
    chunk_id: Optional[str] = None

    # Core attributes
    stats: CoreStats = Field(default_factory=CoreStats)
    needs: CoreNeeds = Field(default_factory=CoreNeeds)
    alignment: Alignment = Field(default_factory=Alignment)

    # Health
    health: float = Field(default=1.0, ge=0.0, le=1.0,
                          description="Current health (0=dead, 1=full)")
    stamina: float = Field(default=1.0, ge=0.0, le=1.0,
                           description="Current stamina (0=exhausted, 1=full)")
    is_alive: bool = True

    # Age and development
    age_days: int = Field(default=0, ge=0)
    generation: int = Field(default=0, ge=0)

    # Physical attributes
    physical: PhysicalAttributes = Field(default_factory=PhysicalAttributes)

    # Genetics (stored as JSON for flexibility)
    genome: Optional[Dict[str, Any]] = None

    # Social
    faction_id: Optional[UUID] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

    # Current activity
    current_action: Optional[str] = None
    current_plan: Optional[str] = None

    class Config:
        use_enum_values = True

    @property
    def life_stage(self) -> LifeStage:
        """Calculate life stage based on age"""
        years = self.age_days / 365

        if years < 2:
            return LifeStage.INFANT
        elif years < 12:
            return LifeStage.CHILD
        elif years < 18:
            return LifeStage.ADOLESCENT
        elif years < 60:
            return LifeStage.ADULT
        else:
            return LifeStage.ELDER

    @property
    def age_years(self) -> float:
        """Age in years"""
        return self.age_days / 365.0

    def update_time(self, hours_elapsed: float, environment_temp: float = 20.0) -> None:
        """
        Update agent state based on time passage.

        Args:
            hours_elapsed: Game hours elapsed
            environment_temp: Current temperature
        """
        # Update needs
        self.needs.update(hours_elapsed, environment_temp)

        # Regenerate stamina (slower if needs are high)
        stamina_regen = 0.1 * hours_elapsed
        if self.needs.rest > 0.5:
            stamina_regen *= 0.5
        if self.needs.hunger > 0.5:
            stamina_regen *= 0.7
        self.stamina = min(1.0, self.stamina + stamina_regen)

        # Health damage from critical needs
        critical_needs = self.needs.get_critical_needs(threshold=0.9)
        if critical_needs:
            health_damage = 0.01 * hours_elapsed * len(critical_needs)
            self.health = max(0.0, self.health - health_damage)

        # Check for death
        if self.health <= 0:
            self.is_alive = False

        # Update last active time
        self.last_active = datetime.utcnow()

    def take_damage(self, amount: float) -> None:
        """Apply damage to agent"""
        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.is_alive = False

    def heal(self, amount: float) -> None:
        """Heal agent"""
        if self.is_alive:
            self.health = min(1.0, self.health + amount)

    def move_to(self, x: int, y: int) -> None:
        """Move agent to new position"""
        self.position_x = x
        self.position_y = y
        # Calculate chunk
        self.chunk_id = f"{x // 256}_{y // 256}"

    def to_database_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion"""
        return {
            "agent_id": str(self.agent_id),
            "world_id": str(self.world_id) if self.world_id else None,
            "agent_type": self.agent_type,
            "name": self.name,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "chunk_id": self.chunk_id,
            "strength": self.stats.strength,
            "dexterity": self.stats.dexterity,
            "constitution": self.stats.constitution,
            "intelligence": self.stats.intelligence,
            "wisdom": self.stats.wisdom,
            "charisma": self.stats.charisma,
            "hunger": self.needs.hunger,
            "thirst": self.needs.thirst,
            "rest": self.needs.rest,
            "warmth": self.needs.warmth,
            "safety": self.needs.safety,
            "social": self.needs.social,
            "good_evil_score": self.alignment.good_evil,
            "lawful_chaotic_score": self.alignment.lawful_chaotic,
            "health": self.health,
            "stamina": self.stamina,
            "is_alive": self.is_alive,
            "age_days": self.age_days,
            "genome": self.genome,
            "physical_attributes": self.physical.model_dump(),
            "faction_id": str(self.faction_id) if self.faction_id else None,
            "generation": self.generation,
        }

    @classmethod
    def from_database_row(cls, row: Dict[str, Any]) -> "AgentState":
        """Create AgentState from database row"""
        stats = CoreStats(
            strength=row.get("strength", 10),
            dexterity=row.get("dexterity", 10),
            constitution=row.get("constitution", 10),
            intelligence=row.get("intelligence", 10),
            wisdom=row.get("wisdom", 10),
            charisma=row.get("charisma", 10),
        )

        needs = CoreNeeds(
            hunger=row.get("hunger", 0.0),
            thirst=row.get("thirst", 0.0),
            rest=row.get("rest", 0.0),
            warmth=row.get("warmth", 0.0),
            safety=row.get("safety", 0.0),
            social=row.get("social", 0.0),
        )

        alignment = Alignment(
            good_evil=row.get("good_evil_score", 0.0),
            lawful_chaotic=row.get("lawful_chaotic_score", 0.0),
        )

        physical = PhysicalAttributes(**(row.get("physical_attributes") or {}))

        return cls(
            agent_id=UUID(row["agent_id"]),
            world_id=UUID(row["world_id"]) if row.get("world_id") else None,
            name=row["name"],
            agent_type=AgentType(row["agent_type"]),
            position_x=row["position_x"],
            position_y=row["position_y"],
            chunk_id=row.get("chunk_id"),
            stats=stats,
            needs=needs,
            alignment=alignment,
            health=row.get("health", 1.0),
            stamina=row.get("stamina", 1.0),
            is_alive=row.get("is_alive", True),
            age_days=row.get("age_days", 0),
            physical=physical,
            genome=row.get("genome"),
            faction_id=UUID(row["faction_id"]) if row.get("faction_id") else None,
            generation=row.get("generation", 0),
            created_at=row.get("created_at", datetime.utcnow()),
            last_active=row.get("last_active", datetime.utcnow()),
        )
