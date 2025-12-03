"""
Genetics Models
Mendelian inheritance system for agent traits
Based on CORE_AGENTS.md specification
"""

from typing import Optional, Dict, List, Tuple, Any
from pydantic import BaseModel, Field
import random


class GeneMarker(BaseModel):
    """
    Single gene with two alleles (maternal and paternal).
    Alleles are continuous values 0.0 to 1.0.
    """
    maternal: float = Field(ge=0.0, le=1.0, description="Maternal allele value")
    paternal: float = Field(ge=0.0, le=1.0, description="Paternal allele value")
    dominance: float = Field(default=0.5, ge=0.0, le=1.0,
                             description="Dominance factor (0=recessive, 1=dominant)")

    def express(self) -> float:
        """
        Express the gene based on alleles and dominance.
        Higher dominance means maternal allele has more effect.

        Returns:
            Expressed value between 0.0 and 1.0
        """
        if self.dominance > 0.5:
            # Maternal dominant
            weight = self.dominance
            return self.maternal * weight + self.paternal * (1 - weight)
        else:
            # Paternal dominant
            weight = 1 - self.dominance
            return self.paternal * weight + self.maternal * (1 - weight)

    @classmethod
    def random(cls) -> "GeneMarker":
        """Generate random gene marker"""
        return cls(
            maternal=random.random(),
            paternal=random.random(),
            dominance=random.random()
        )

    @classmethod
    def inherit(cls, parent1: "GeneMarker", parent2: "GeneMarker") -> "GeneMarker":
        """
        Create new gene by inheriting from two parents.
        Each parent contributes one allele.
        """
        # Randomly select which allele from each parent
        from_p1 = parent1.maternal if random.random() < 0.5 else parent1.paternal
        from_p2 = parent2.maternal if random.random() < 0.5 else parent2.paternal

        # Possible mutation (small random adjustment)
        if random.random() < 0.05:  # 5% mutation chance
            from_p1 = max(0, min(1, from_p1 + random.gauss(0, 0.1)))
        if random.random() < 0.05:
            from_p2 = max(0, min(1, from_p2 + random.gauss(0, 0.1)))

        # Dominance can also mutate slightly
        avg_dominance = (parent1.dominance + parent2.dominance) / 2
        if random.random() < 0.1:
            avg_dominance = max(0, min(1, avg_dominance + random.gauss(0, 0.1)))

        return cls(
            maternal=from_p1,
            paternal=from_p2,
            dominance=avg_dominance
        )


class TraitGenome(BaseModel):
    """
    Genome for a single trait, containing 35 gene markers.
    This allows for complex polygenic inheritance.
    """
    markers: List[GeneMarker] = Field(default_factory=list)
    num_markers: int = Field(default=35)

    def model_post_init(self, __context) -> None:
        """Initialize markers if empty"""
        if not self.markers:
            self.markers = [GeneMarker.random() for _ in range(self.num_markers)]

    def express(self) -> float:
        """
        Calculate trait expression as weighted average of all markers.

        Returns:
            Trait value between 0.0 and 1.0
        """
        if not self.markers:
            return 0.5

        total = sum(marker.express() for marker in self.markers)
        return total / len(self.markers)

    @classmethod
    def inherit(cls, parent1: "TraitGenome", parent2: "TraitGenome") -> "TraitGenome":
        """Inherit trait genome from two parents"""
        new_markers = []
        for i in range(len(parent1.markers)):
            new_markers.append(GeneMarker.inherit(parent1.markers[i], parent2.markers[i]))
        return cls(markers=new_markers)


class AgentGenome(BaseModel):
    """
    Complete genome for an agent.
    Contains trait genomes for all heritable characteristics.
    """
    # Physical traits
    height: TraitGenome = Field(default_factory=TraitGenome)
    build: TraitGenome = Field(default_factory=TraitGenome)
    metabolism: TraitGenome = Field(default_factory=TraitGenome)

    # Appearance
    hair_color_r: TraitGenome = Field(default_factory=TraitGenome)
    hair_color_g: TraitGenome = Field(default_factory=TraitGenome)
    hair_color_b: TraitGenome = Field(default_factory=TraitGenome)
    eye_color: TraitGenome = Field(default_factory=TraitGenome)
    skin_tone: TraitGenome = Field(default_factory=TraitGenome)

    # Stat tendencies (influence starting stats)
    strength_tendency: TraitGenome = Field(default_factory=TraitGenome)
    dexterity_tendency: TraitGenome = Field(default_factory=TraitGenome)
    constitution_tendency: TraitGenome = Field(default_factory=TraitGenome)
    intelligence_tendency: TraitGenome = Field(default_factory=TraitGenome)
    wisdom_tendency: TraitGenome = Field(default_factory=TraitGenome)
    charisma_tendency: TraitGenome = Field(default_factory=TraitGenome)

    # Personality tendencies
    aggression: TraitGenome = Field(default_factory=TraitGenome)
    sociability: TraitGenome = Field(default_factory=TraitGenome)
    curiosity: TraitGenome = Field(default_factory=TraitGenome)
    industriousness: TraitGenome = Field(default_factory=TraitGenome)

    # Longevity and health
    lifespan: TraitGenome = Field(default_factory=TraitGenome)
    disease_resistance: TraitGenome = Field(default_factory=TraitGenome)

    def express_physical_attributes(self) -> Dict[str, Any]:
        """
        Express all physical traits into attribute values.

        Returns:
            Dictionary of physical attributes
        """
        # Height: 150-200cm range
        height_expr = self.height.express()
        height_cm = 150 + (height_expr * 50)

        # Build affects weight
        build_expr = self.build.express()
        base_weight = (height_cm - 100) * 0.9  # Baseline weight
        weight_modifier = 0.7 + (build_expr * 0.6)  # 0.7 to 1.3
        weight_kg = base_weight * weight_modifier

        # Hair color (simplified to common colors)
        hair_r = self.hair_color_r.express()
        hair_g = self.hair_color_g.express()
        hair_b = self.hair_color_b.express()
        hair_color = self._express_hair_color(hair_r, hair_g, hair_b)

        # Eye color
        eye_expr = self.eye_color.express()
        eye_color = self._express_eye_color(eye_expr)

        # Skin tone
        skin_expr = self.skin_tone.express()
        skin_tone = self._express_skin_tone(skin_expr)

        # Build category
        if build_expr < 0.3:
            build = "thin"
        elif build_expr < 0.5:
            build = "average"
        elif build_expr < 0.7:
            build = "muscular"
        else:
            build = "heavy"

        return {
            "height_cm": round(height_cm, 1),
            "weight_kg": round(weight_kg, 1),
            "hair_color": hair_color,
            "eye_color": eye_color,
            "skin_tone": skin_tone,
            "build": build,
        }

    def express_stat_tendencies(self) -> Dict[str, int]:
        """
        Express genetic tendencies as stat modifiers.
        Returns values to add to base stats (can be negative).

        Returns:
            Dictionary of stat modifications (-4 to +4)
        """
        def tendency_to_modifier(tendency: float) -> int:
            # Map 0-1 to -4 to +4
            return int((tendency - 0.5) * 8)

        return {
            "strength": tendency_to_modifier(self.strength_tendency.express()),
            "dexterity": tendency_to_modifier(self.dexterity_tendency.express()),
            "constitution": tendency_to_modifier(self.constitution_tendency.express()),
            "intelligence": tendency_to_modifier(self.intelligence_tendency.express()),
            "wisdom": tendency_to_modifier(self.wisdom_tendency.express()),
            "charisma": tendency_to_modifier(self.charisma_tendency.express()),
        }

    def express_personality(self) -> Dict[str, float]:
        """
        Express personality tendencies.

        Returns:
            Dictionary of personality traits (0-1 scale)
        """
        return {
            "aggression": self.aggression.express(),
            "sociability": self.sociability.express(),
            "curiosity": self.curiosity.express(),
            "industriousness": self.industriousness.express(),
        }

    def get_lifespan_modifier(self) -> float:
        """
        Get lifespan modifier from genetics.

        Returns:
            Multiplier for base lifespan (0.7 to 1.3)
        """
        return 0.7 + (self.lifespan.express() * 0.6)

    def get_disease_resistance(self) -> float:
        """
        Get disease resistance from genetics.

        Returns:
            Resistance value (0-1)
        """
        return self.disease_resistance.express()

    @classmethod
    def inherit(cls, parent1: "AgentGenome", parent2: "AgentGenome") -> "AgentGenome":
        """
        Create new genome by combining two parent genomes.

        Args:
            parent1: First parent's genome
            parent2: Second parent's genome

        Returns:
            New child genome
        """
        return cls(
            height=TraitGenome.inherit(parent1.height, parent2.height),
            build=TraitGenome.inherit(parent1.build, parent2.build),
            metabolism=TraitGenome.inherit(parent1.metabolism, parent2.metabolism),
            hair_color_r=TraitGenome.inherit(parent1.hair_color_r, parent2.hair_color_r),
            hair_color_g=TraitGenome.inherit(parent1.hair_color_g, parent2.hair_color_g),
            hair_color_b=TraitGenome.inherit(parent1.hair_color_b, parent2.hair_color_b),
            eye_color=TraitGenome.inherit(parent1.eye_color, parent2.eye_color),
            skin_tone=TraitGenome.inherit(parent1.skin_tone, parent2.skin_tone),
            strength_tendency=TraitGenome.inherit(parent1.strength_tendency, parent2.strength_tendency),
            dexterity_tendency=TraitGenome.inherit(parent1.dexterity_tendency, parent2.dexterity_tendency),
            constitution_tendency=TraitGenome.inherit(parent1.constitution_tendency, parent2.constitution_tendency),
            intelligence_tendency=TraitGenome.inherit(parent1.intelligence_tendency, parent2.intelligence_tendency),
            wisdom_tendency=TraitGenome.inherit(parent1.wisdom_tendency, parent2.wisdom_tendency),
            charisma_tendency=TraitGenome.inherit(parent1.charisma_tendency, parent2.charisma_tendency),
            aggression=TraitGenome.inherit(parent1.aggression, parent2.aggression),
            sociability=TraitGenome.inherit(parent1.sociability, parent2.sociability),
            curiosity=TraitGenome.inherit(parent1.curiosity, parent2.curiosity),
            industriousness=TraitGenome.inherit(parent1.industriousness, parent2.industriousness),
            lifespan=TraitGenome.inherit(parent1.lifespan, parent2.lifespan),
            disease_resistance=TraitGenome.inherit(parent1.disease_resistance, parent2.disease_resistance),
        )

    def _express_hair_color(self, r: float, g: float, b: float) -> str:
        """Convert RGB expression to hair color name"""
        # Simplified mapping
        brightness = (r + g + b) / 3
        redness = r - (g + b) / 2

        if brightness > 0.8:
            return "white"
        elif brightness > 0.6:
            return "blonde"
        elif brightness > 0.4:
            if redness > 0.1:
                return "auburn"
            else:
                return "brown"
        elif brightness > 0.2:
            if redness > 0.1:
                return "red"
            else:
                return "dark_brown"
        else:
            return "black"

    def _express_eye_color(self, value: float) -> str:
        """Convert expression value to eye color"""
        if value < 0.2:
            return "blue"
        elif value < 0.35:
            return "green"
        elif value < 0.5:
            return "hazel"
        elif value < 0.7:
            return "amber"
        else:
            return "brown"

    def _express_skin_tone(self, value: float) -> str:
        """Convert expression value to skin tone"""
        if value < 0.2:
            return "pale"
        elif value < 0.4:
            return "fair"
        elif value < 0.6:
            return "medium"
        elif value < 0.8:
            return "olive"
        else:
            return "dark"

    def to_json(self) -> Dict[str, Any]:
        """Serialize genome to JSON-compatible dictionary"""
        return self.model_dump()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "AgentGenome":
        """Deserialize genome from JSON dictionary"""
        return cls.model_validate(data)
