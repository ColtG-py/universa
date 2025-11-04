# Agent Simulation Layer - Part 2: Skill System

## Overview

The skill system provides structured progression for agents through hierarchical skill trees. Skills are organized into 12 top-level categories with 100+ individual skills. Each skill has governing stats that affect performance and improve through use, creating a feedback loop between stats and skills.

**Key Design Principles:**
- **No Search/Embeddings**: Agents navigate taxonomy deterministically
- **Stat-Skill Integration**: Using skills improves governing stats
- **Parent-Child Inheritance**: Skill improvements benefit related skills
- **Natural Specialization**: Repeated practice creates experts

---

## Table of Contents

1. [Skill Taxonomy Structure](#skill-taxonomy-structure)
2. [Governing Stats System](#governing-stats-system)
3. [Top-Down Navigation](#top-down-navigation)
4. [Skill Progression & XP](#skill-progression--xp)
5. [Stat Improvement System](#stat-improvement-system)
6. [Skill Data Models](#skill-data-models)

---

## Skill Taxonomy Structure

### 12 Top-Level Categories

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Set
from uuid import UUID

class SkillCategory(str, Enum):
    """Top-level skill categories (12 total)"""
    SURVIVAL = "survival"           # Basic survival skills
    COMBAT = "combat"              # Fighting and defense
    CRAFTING = "crafting"          # Making items
    GATHERING = "gathering"        # Collecting resources
    SOCIAL = "social"              # Interpersonal skills
    KNOWLEDGE = "knowledge"        # Learning and wisdom
    LEADERSHIP = "leadership"      # Managing others
    TRADE = "trade"               # Commerce and negotiation
    MAGIC = "magic"               # Supernatural abilities
    ATHLETICS = "athletics"        # Physical prowess
    STEALTH = "stealth"           # Sneaking and deception
    EXPLORATION = "exploration"    # Navigation and discovery
```

### Complete Skill Tree

The full taxonomy is organized hierarchically with up to 3 levels of depth:
- **Level 1**: Category (12 categories)
- **Level 2**: Subcategory (~40 subcategories)
- **Level 3**: Specific Skill (~100 skills)

```python
# Complete skill tree definition
SKILL_TREE = {
    "survival": {
        "description": "Skills for staying alive in harsh conditions",
        "governing_stats": ["constitution", "wisdom"],
        "children": {
            "survival.foraging": {
                "name": "Foraging",
                "description": "Find edible plants and fungi",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.3,
                "children": {
                    "survival.foraging.plants": {
                        "name": "Plant Identification",
                        "description": "Identify edible vs poisonous plants",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "survival.foraging.mushrooms": {
                        "name": "Mushroom Identification",
                        "description": "Identify safe mushrooms",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "survival.foraging.berries": {
                        "name": "Berry Gathering",
                        "description": "Find and gather berries",
                        "governing_stats": ["wisdom", "dexterity"]
                    }
                }
            },
            "survival.hunting": {
                "name": "Hunting",
                "description": "Track and hunt animals",
                "governing_stats": ["dexterity", "wisdom"],
                "base_difficulty": 0.6,
                "children": {
                    "survival.hunting.tracking": {
                        "name": "Tracking",
                        "description": "Follow animal trails",
                        "governing_stats": ["wisdom", "intelligence"]
                    },
                    "survival.hunting.trapping": {
                        "name": "Trapping",
                        "description": "Set and maintain traps",
                        "governing_stats": ["intelligence", "dexterity"]
                    },
                    "survival.hunting.archery": {
                        "name": "Hunting with Bow",
                        "description": "Hunt using bow and arrow",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "survival.hunting.spear": {
                        "name": "Spear Hunting",
                        "description": "Hunt using spear",
                        "governing_stats": ["strength", "dexterity"]
                    }
                }
            },
            "survival.shelter": {
                "name": "Shelter Building",
                "description": "Construct shelters and buildings",
                "governing_stats": ["strength", "intelligence"],
                "base_difficulty": 0.5
            },
            "survival.fire": {
                "name": "Fire Making",
                "description": "Start and maintain fires",
                "governing_stats": ["dexterity", "intelligence"],
                "base_difficulty": 0.4
            },
            "survival.water": {
                "name": "Water Finding",
                "description": "Locate and purify water sources",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.5
            }
        }
    },
    
    "combat": {
        "description": "Skills for fighting and self-defense",
        "governing_stats": ["strength", "dexterity"],
        "children": {
            "combat.melee": {
                "name": "Melee Combat",
                "description": "Close-quarters fighting",
                "governing_stats": ["strength", "dexterity"],
                "base_difficulty": 0.6,
                "children": {
                    "combat.melee.swords": {
                        "name": "Sword Fighting",
                        "description": "Fight with swords",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "combat.melee.axes": {
                        "name": "Axe Combat",
                        "description": "Fight with axes",
                        "governing_stats": ["strength", "dexterity"]
                    },
                    "combat.melee.spears": {
                        "name": "Spear Combat",
                        "description": "Fight with spears",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "combat.melee.unarmed": {
                        "name": "Unarmed Combat",
                        "description": "Hand-to-hand fighting",
                        "governing_stats": ["strength", "dexterity"]
                    },
                    "combat.melee.shields": {
                        "name": "Shield Fighting",
                        "description": "Combat with shield",
                        "governing_stats": ["strength", "constitution"]
                    }
                }
            },
            "combat.ranged": {
                "name": "Ranged Combat",
                "description": "Fighting at distance",
                "governing_stats": ["dexterity", "wisdom"],
                "base_difficulty": 0.7,
                "children": {
                    "combat.ranged.bows": {
                        "name": "Archery",
                        "description": "Combat with bow and arrow",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "combat.ranged.crossbows": {
                        "name": "Crossbow Combat",
                        "description": "Combat with crossbow",
                        "governing_stats": ["dexterity", "intelligence"]
                    },
                    "combat.ranged.throwing": {
                        "name": "Throwing Weapons",
                        "description": "Throw spears, knives, etc.",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "combat.ranged.slings": {
                        "name": "Sling Combat",
                        "description": "Use slings for ranged attack",
                        "governing_stats": ["dexterity", "strength"]
                    }
                }
            },
            "combat.defense": {
                "name": "Defense",
                "description": "Block and evade attacks",
                "governing_stats": ["dexterity", "constitution"],
                "base_difficulty": 0.5,
                "children": {
                    "combat.defense.dodge": {
                        "name": "Dodging",
                        "description": "Evade attacks",
                        "governing_stats": ["dexterity", "wisdom"]
                    },
                    "combat.defense.parry": {
                        "name": "Parrying",
                        "description": "Deflect attacks with weapon",
                        "governing_stats": ["dexterity", "strength"]
                    },
                    "combat.defense.armor": {
                        "name": "Armor Use",
                        "description": "Effectively use armor",
                        "governing_stats": ["constitution", "strength"]
                    }
                }
            },
            "combat.tactics": {
                "name": "Combat Tactics",
                "description": "Strategic fighting",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.7
            }
        }
    },
    
    "crafting": {
        "description": "Skills for creating items and structures",
        "governing_stats": ["dexterity", "intelligence"],
        "children": {
            "crafting.woodworking": {
                "name": "Woodworking",
                "description": "Work with wood",
                "governing_stats": ["dexterity", "strength"],
                "base_difficulty": 0.5,
                "children": {
                    "crafting.woodworking.carpentry": {
                        "name": "Carpentry",
                        "description": "Build wooden structures",
                        "governing_stats": ["strength", "intelligence"]
                    },
                    "crafting.woodworking.carving": {
                        "name": "Wood Carving",
                        "description": "Carve decorative items",
                        "governing_stats": ["dexterity", "intelligence"]
                    },
                    "crafting.woodworking.bowmaking": {
                        "name": "Bow Making",
                        "description": "Craft bows and arrows",
                        "governing_stats": ["dexterity", "intelligence"]
                    }
                }
            },
            "crafting.smithing": {
                "name": "Smithing",
                "description": "Work with metal",
                "governing_stats": ["strength", "intelligence"],
                "base_difficulty": 0.8,
                "children": {
                    "crafting.smithing.blacksmithing": {
                        "name": "Blacksmithing",
                        "description": "Create metal tools and items",
                        "governing_stats": ["strength", "intelligence"]
                    },
                    "crafting.smithing.weaponsmithing": {
                        "name": "Weapon Smithing",
                        "description": "Forge weapons",
                        "governing_stats": ["strength", "dexterity"]
                    },
                    "crafting.smithing.armorsmithing": {
                        "name": "Armor Smithing",
                        "description": "Craft armor",
                        "governing_stats": ["strength", "intelligence"]
                    },
                    "crafting.smithing.jewelrymaking": {
                        "name": "Jewelry Making",
                        "description": "Craft jewelry and decorations",
                        "governing_stats": ["dexterity", "intelligence"]
                    }
                }
            },
            "crafting.leatherworking": {
                "name": "Leatherworking",
                "description": "Work with leather and hides",
                "governing_stats": ["dexterity", "strength"],
                "base_difficulty": 0.5,
                "children": {
                    "crafting.leatherworking.tanning": {
                        "name": "Tanning",
                        "description": "Process hides into leather",
                        "governing_stats": ["strength", "intelligence"]
                    },
                    "crafting.leatherworking.armormaking": {
                        "name": "Leather Armor Making",
                        "description": "Craft leather armor",
                        "governing_stats": ["dexterity", "intelligence"]
                    }
                }
            },
            "crafting.textiles": {
                "name": "Textile Work",
                "description": "Work with cloth and fabric",
                "governing_stats": ["dexterity", "intelligence"],
                "base_difficulty": 0.4,
                "children": {
                    "crafting.textiles.weaving": {
                        "name": "Weaving",
                        "description": "Weave cloth from thread",
                        "governing_stats": ["dexterity", "intelligence"]
                    },
                    "crafting.textiles.tailoring": {
                        "name": "Tailoring",
                        "description": "Sew clothing",
                        "governing_stats": ["dexterity", "intelligence"]
                    }
                }
            },
            "crafting.cooking": {
                "name": "Cooking",
                "description": "Prepare food",
                "governing_stats": ["intelligence", "dexterity"],
                "base_difficulty": 0.4,
                "children": {
                    "crafting.cooking.butchery": {
                        "name": "Butchery",
                        "description": "Process animal carcasses",
                        "governing_stats": ["strength", "intelligence"]
                    },
                    "crafting.cooking.preservation": {
                        "name": "Food Preservation",
                        "description": "Preserve food for storage",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "crafting.cooking.baking": {
                        "name": "Baking",
                        "description": "Bake bread and pastries",
                        "governing_stats": ["intelligence", "dexterity"]
                    }
                }
            },
            "crafting.pottery": {
                "name": "Pottery",
                "description": "Create ceramic items",
                "governing_stats": ["dexterity", "intelligence"],
                "base_difficulty": 0.6
            },
            "crafting.alchemy": {
                "name": "Alchemy",
                "description": "Create potions and medicines",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.7,
                "children": {
                    "crafting.alchemy.potions": {
                        "name": "Potion Making",
                        "description": "Brew potions",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "crafting.alchemy.poisons": {
                        "name": "Poison Making",
                        "description": "Create poisons",
                        "governing_stats": ["intelligence", "wisdom"]
                    }
                }
            }
        }
    },
    
    "gathering": {
        "description": "Skills for collecting natural resources",
        "governing_stats": ["strength", "wisdom"],
        "children": {
            "gathering.mining": {
                "name": "Mining",
                "description": "Extract minerals from earth",
                "governing_stats": ["strength", "constitution"],
                "base_difficulty": 0.6,
                "children": {
                    "gathering.mining.prospecting": {
                        "name": "Prospecting",
                        "description": "Find mineral deposits",
                        "governing_stats": ["wisdom", "intelligence"]
                    },
                    "gathering.mining.ore_extraction": {
                        "name": "Ore Extraction",
                        "description": "Extract ore efficiently",
                        "governing_stats": ["strength", "intelligence"]
                    }
                }
            },
            "gathering.logging": {
                "name": "Logging",
                "description": "Fell trees and harvest wood",
                "governing_stats": ["strength", "constitution"],
                "base_difficulty": 0.5
            },
            "gathering.fishing": {
                "name": "Fishing",
                "description": "Catch fish from water",
                "governing_stats": ["dexterity", "wisdom"],
                "base_difficulty": 0.4,
                "children": {
                    "gathering.fishing.net": {
                        "name": "Net Fishing",
                        "description": "Fish with nets",
                        "governing_stats": ["dexterity", "intelligence"]
                    },
                    "gathering.fishing.line": {
                        "name": "Line Fishing",
                        "description": "Fish with hook and line",
                        "governing_stats": ["dexterity", "wisdom"]
                    },
                    "gathering.fishing.spear": {
                        "name": "Spear Fishing",
                        "description": "Fish with spear",
                        "governing_stats": ["dexterity", "strength"]
                    }
                }
            },
            "gathering.herbalism": {
                "name": "Herbalism",
                "description": "Collect medicinal plants",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.5
            },
            "gathering.scavenging": {
                "name": "Scavenging",
                "description": "Find useful items and materials",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.4
            }
        }
    },
    
    "social": {
        "description": "Skills for interacting with others",
        "governing_stats": ["charisma", "wisdom"],
        "children": {
            "social.persuasion": {
                "name": "Persuasion",
                "description": "Convince others through reason",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.6
            },
            "social.deception": {
                "name": "Deception",
                "description": "Lie convincingly",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.7
            },
            "social.empathy": {
                "name": "Empathy",
                "description": "Understand others' emotions",
                "governing_stats": ["wisdom", "charisma"],
                "base_difficulty": 0.5
            },
            "social.intimidation": {
                "name": "Intimidation",
                "description": "Frighten or coerce others",
                "governing_stats": ["strength", "charisma"],
                "base_difficulty": 0.6
            },
            "social.performance": {
                "name": "Performance",
                "description": "Entertain through acting, music, etc.",
                "governing_stats": ["charisma", "dexterity"],
                "base_difficulty": 0.5,
                "children": {
                    "social.performance.music": {
                        "name": "Music",
                        "description": "Play musical instruments",
                        "governing_stats": ["dexterity", "charisma"]
                    },
                    "social.performance.storytelling": {
                        "name": "Storytelling",
                        "description": "Tell engaging stories",
                        "governing_stats": ["charisma", "intelligence"]
                    }
                }
            },
            "social.insight": {
                "name": "Insight",
                "description": "Read people's intentions",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.6
            }
        }
    },
    
    "knowledge": {
        "description": "Skills for learning and understanding",
        "governing_stats": ["intelligence", "wisdom"],
        "children": {
            "knowledge.medicine": {
                "name": "Medicine",
                "description": "Heal injuries and illness",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.8,
                "children": {
                    "knowledge.medicine.diagnosis": {
                        "name": "Diagnosis",
                        "description": "Identify illnesses",
                        "governing_stats": ["wisdom", "intelligence"]
                    },
                    "knowledge.medicine.treatment": {
                        "name": "Treatment",
                        "description": "Treat injuries and diseases",
                        "governing_stats": ["intelligence", "dexterity"]
                    },
                    "knowledge.medicine.surgery": {
                        "name": "Surgery",
                        "description": "Perform surgical procedures",
                        "governing_stats": ["dexterity", "intelligence"]
                    }
                }
            },
            "knowledge.history": {
                "name": "History",
                "description": "Knowledge of past events",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.6
            },
            "knowledge.nature": {
                "name": "Nature Lore",
                "description": "Understanding of natural world",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.5,
                "children": {
                    "knowledge.nature.animals": {
                        "name": "Animal Lore",
                        "description": "Knowledge about animals",
                        "governing_stats": ["wisdom", "intelligence"]
                    },
                    "knowledge.nature.plants": {
                        "name": "Plant Lore",
                        "description": "Knowledge about plants",
                        "governing_stats": ["wisdom", "intelligence"]
                    }
                }
            },
            "knowledge.magic_theory": {
                "name": "Magic Theory",
                "description": "Theoretical understanding of magic",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.9
            },
            "knowledge.engineering": {
                "name": "Engineering",
                "description": "Design and build complex structures",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.8
            }
        }
    },
    
    "leadership": {
        "description": "Skills for managing and leading others",
        "governing_stats": ["charisma", "intelligence"],
        "children": {
            "leadership.command": {
                "name": "Command",
                "description": "Lead groups effectively",
                "governing_stats": ["charisma", "wisdom"],
                "base_difficulty": 0.7
            },
            "leadership.tactics": {
                "name": "Tactics",
                "description": "Plan strategic actions",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.8
            },
            "leadership.diplomacy": {
                "name": "Diplomacy",
                "description": "Negotiate between groups",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.7
            },
            "leadership.inspiration": {
                "name": "Inspiration",
                "description": "Motivate others to action",
                "governing_stats": ["charisma", "wisdom"],
                "base_difficulty": 0.6
            },
            "leadership.administration": {
                "name": "Administration",
                "description": "Manage resources and people",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.7
            }
        }
    },
    
    "trade": {
        "description": "Skills for commerce and economics",
        "governing_stats": ["intelligence", "charisma"],
        "children": {
            "trade.bargaining": {
                "name": "Bargaining",
                "description": "Negotiate prices",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.5
            },
            "trade.appraisal": {
                "name": "Appraisal",
                "description": "Assess item value",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.6
            },
            "trade.accounting": {
                "name": "Accounting",
                "description": "Track finances and resources",
                "governing_stats": ["intelligence", "wisdom"],
                "base_difficulty": 0.6
            },
            "trade.merchant": {
                "name": "Merchant",
                "description": "Buy and sell goods profitably",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.6
            }
        }
    },
    
    "magic": {
        "description": "Skills for supernatural abilities",
        "governing_stats": ["intelligence", "wisdom"],
        "children": {
            "magic.evocation": {
                "name": "Evocation",
                "description": "Elemental magic (fire, ice, etc.)",
                "governing_stats": ["intelligence", "charisma"],
                "base_difficulty": 0.9,
                "children": {
                    "magic.evocation.fire": {
                        "name": "Fire Magic",
                        "description": "Control fire",
                        "governing_stats": ["intelligence", "charisma"]
                    },
                    "magic.evocation.ice": {
                        "name": "Ice Magic",
                        "description": "Control ice and cold",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "magic.evocation.lightning": {
                        "name": "Lightning Magic",
                        "description": "Control electricity",
                        "governing_stats": ["intelligence", "dexterity"]
                    }
                }
            },
            "magic.healing": {
                "name": "Healing Magic",
                "description": "Restore health with magic",
                "governing_stats": ["wisdom", "charisma"],
                "base_difficulty": 0.8
            },
            "magic.divination": {
                "name": "Divination",
                "description": "Perceive distant or future events",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.9
            },
            "magic.enchantment": {
                "name": "Enchantment",
                "description": "Imbue objects with magic",
                "governing_stats": ["intelligence", "dexterity"],
                "base_difficulty": 0.9
            }
        }
    },
    
    "athletics": {
        "description": "Physical prowess and endurance",
        "governing_stats": ["strength", "constitution"],
        "children": {
            "athletics.running": {
                "name": "Running",
                "description": "Run quickly and efficiently",
                "governing_stats": ["dexterity", "constitution"],
                "base_difficulty": 0.3
            },
            "athletics.climbing": {
                "name": "Climbing",
                "description": "Climb walls and surfaces",
                "governing_stats": ["strength", "dexterity"],
                "base_difficulty": 0.5
            },
            "athletics.swimming": {
                "name": "Swimming",
                "description": "Swim efficiently",
                "governing_stats": ["strength", "constitution"],
                "base_difficulty": 0.4
            },
            "athletics.jumping": {
                "name": "Jumping",
                "description": "Jump long distances",
                "governing_stats": ["strength", "dexterity"],
                "base_difficulty": 0.4
            },
            "athletics.acrobatics": {
                "name": "Acrobatics",
                "description": "Perform athletic feats",
                "governing_stats": ["dexterity", "strength"],
                "base_difficulty": 0.6
            }
        }
    },
    
    "stealth": {
        "description": "Skills for sneaking and deception",
        "governing_stats": ["dexterity", "intelligence"],
        "children": {
            "stealth.sneaking": {
                "name": "Sneaking",
                "description": "Move silently and unseen",
                "governing_stats": ["dexterity", "wisdom"],
                "base_difficulty": 0.6
            },
            "stealth.lockpicking": {
                "name": "Lockpicking",
                "description": "Pick locks",
                "governing_stats": ["dexterity", "intelligence"],
                "base_difficulty": 0.7
            },
            "stealth.pickpocketing": {
                "name": "Pickpocketing",
                "description": "Steal from people unnoticed",
                "governing_stats": ["dexterity", "intelligence"],
                "base_difficulty": 0.8
            },
            "stealth.disguise": {
                "name": "Disguise",
                "description": "Alter appearance",
                "governing_stats": ["charisma", "intelligence"],
                "base_difficulty": 0.7
            },
            "stealth.trap_detection": {
                "name": "Trap Detection",
                "description": "Find and disarm traps",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.7
            }
        }
    },
    
    "exploration": {
        "description": "Skills for navigation and discovery",
        "governing_stats": ["wisdom", "intelligence"],
        "children": {
            "exploration.navigation": {
                "name": "Navigation",
                "description": "Find your way using landmarks",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.5,
                "children": {
                    "exploration.navigation.celestial": {
                        "name": "Celestial Navigation",
                        "description": "Navigate by stars",
                        "governing_stats": ["intelligence", "wisdom"]
                    },
                    "exploration.navigation.map_reading": {
                        "name": "Map Reading",
                        "description": "Read and use maps",
                        "governing_stats": ["intelligence", "wisdom"]
                    }
                }
            },
            "exploration.cartography": {
                "name": "Cartography",
                "description": "Create maps",
                "governing_stats": ["intelligence", "dexterity"],
                "base_difficulty": 0.6
            },
            "exploration.perception": {
                "name": "Perception",
                "description": "Notice details in environment",
                "governing_stats": ["wisdom", "intelligence"],
                "base_difficulty": 0.5
            },
            "exploration.survival_instinct": {
                "name": "Survival Instinct",
                "description": "Sense danger and opportunity",
                "governing_stats": ["wisdom", "constitution"],
                "base_difficulty": 0.6
            }
        }
    }
}
```

---

## Governing Stats System

Every skill has **two governing stats**:
- **Primary Stat**: Main attribute (70% weight)
- **Secondary Stat**: Supporting attribute (30% weight)

### How Stats Affect Skills

```python
class SkillDefinition(BaseModel):
    """Complete definition of a skill"""
    
    skill_id: str = Field(description="Unique identifier (e.g., 'combat.melee.swords')")
    name: str = Field(description="Display name")
    description: str
    category: SkillCategory
    parent_skill: Optional[str] = None
    
    # Governing stats
    primary_stat: str = Field(description="Main governing stat (70% weight)")
    secondary_stat: str = Field(description="Supporting stat (30% weight)")
    
    # Difficulty and progression
    base_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    xp_per_use: float = 1.0
    xp_to_level: float = 100.0
    max_level: int = 100
    
    # Requirements
    required_skills: Dict[str, int] = {}  # skill_id -> min_level
    required_tools: List[str] = []
    required_materials: List[str] = []
    
    def calculate_success_chance(
        self,
        agent_stats: "CoreStats",
        skill_level: int,
        difficulty_modifier: float = 0.0
    ) -> float:
        """
        Calculate success chance combining stats and skill
        
        Formula:
        base_chance = (1.0 - base_difficulty - difficulty_modifier)
        stat_bonus = primary_stat_modifier * 0.7 + secondary_stat_modifier * 0.3
        skill_bonus = min(skill_level * 0.01, 0.5)  # Max +50% from skill
        
        total = base_chance + stat_bonus + skill_bonus
        clamped to [0.05, 0.95]
        """
        # Base success chance
        base_chance = 1.0 - self.base_difficulty - difficulty_modifier
        
        # Stat bonuses (weighted)
        primary_modifier = agent_stats.get_stat_modifier(self.primary_stat)
        secondary_modifier = agent_stats.get_stat_modifier(self.secondary_stat)
        stat_bonus = primary_modifier * 0.7 + secondary_modifier * 0.3
        
        # Skill proficiency bonus
        skill_bonus = min(skill_level * 0.01, 0.5)
        
        # Parent skill bonus (if exists)
        parent_bonus = 0.0
        # Would be calculated from parent skill level
        
        # Calculate total
        total_chance = base_chance + stat_bonus + skill_bonus + parent_bonus
        
        # Clamp to reasonable range (always some chance of failure/success)
        return np.clip(total_chance, 0.05, 0.95)
    
    def get_stat_improvement_chances(self) -> Dict[str, float]:
        """
        Get chances to improve stats when using this skill
        
        Returns dict of stat_name -> improvement_chance
        """
        return {
            self.primary_stat: 0.10,    # 10% chance to improve primary
            self.secondary_stat: 0.05   # 5% chance to improve secondary
        }
```

---

## Top-Down Navigation

Agents navigate the skill tree **deterministically** without search or embeddings.

### Navigation System

```python
class SkillNavigator:
    """
    Handles deterministic navigation of skill taxonomy
    
    Agents see categories → select → list subcategories → select → list skills
    """
    
    def __init__(self):
        self.skill_tree = SKILL_TREE
        self.skill_cache: Dict[str, SkillDefinition] = {}
        self._build_skill_cache()
    
    def _build_skill_cache(self):
        """Build flat cache of all skills for fast lookup"""
        
        def traverse(node: Dict, parent_id: Optional[str] = None, category: Optional[SkillCategory] = None):
            skill_id = node.get("skill_id")
            if not skill_id and parent_id:
                # This is a parent node, construct ID
                for key in node.keys():
                    if key.startswith(parent_id):
                        skill_id = key
                        break
            
            if "children" in node:
                # Has children - recurse
                for child_id, child_node in node["children"].items():
                    traverse(child_node, skill_id or parent_id, category)
            
            if skill_id and "name" in node:
                # This is a skill - cache it
                skill_def = SkillDefinition(
                    skill_id=skill_id,
                    name=node["name"],
                    description=node["description"],
                    category=category or SkillCategory(skill_id.split(".")[0]),
                    parent_skill=parent_id,
                    primary_stat=node["governing_stats"][0],
                    secondary_stat=node["governing_stats"][1] if len(node["governing_stats"]) > 1 else node["governing_stats"][0],
                    base_difficulty=node.get("base_difficulty", 0.5)
                )
                self.skill_cache[skill_id] = skill_def
        
        for category_id, category_data in self.skill_tree.items():
            traverse(category_data, None, SkillCategory(category_id))
    
    def get_categories(self) -> List[Dict[str, str]]:
        """
        Get top-level skill categories
        
        Returns: List of {id, name, description}
        """
        categories = []
        for cat_id, cat_data in self.skill_tree.items():
            categories.append({
                "id": cat_id,
                "name": cat_id.title(),
                "description": cat_data.get("description", ""),
                "governing_stats": cat_data.get("governing_stats", [])
            })
        return categories
    
    def get_subcategories(self, category_id: str) -> List[Dict[str, str]]:
        """
        Get subcategories within a category
        
        Args:
            category_id: e.g., "combat", "crafting"
        
        Returns: List of subcategory info
        """
        if category_id not in self.skill_tree:
            return []
        
        category_data = self.skill_tree[category_id]
        subcategories = []
        
        if "children" in category_data:
            for subcat_id, subcat_data in category_data["children"].items():
                subcategories.append({
                    "id": subcat_id,
                    "name": subcat_data.get("name", ""),
                    "description": subcat_data.get("description", ""),
                    "governing_stats": subcat_data.get("governing_stats", []),
                    "base_difficulty": subcat_data.get("base_difficulty", 0.5),
                    "has_children": "children" in subcat_data
                })
        
        return subcategories
    
    def get_skills_in_subcategory(self, subcategory_id: str) -> List[Dict[str, Any]]:
        """
        Get specific skills within a subcategory
        
        Args:
            subcategory_id: e.g., "combat.melee", "crafting.smithing"
        
        Returns: List of skill definitions
        """
        # Navigate to the subcategory
        parts = subcategory_id.split(".")
        
        if len(parts) < 2:
            return []
        
        current = self.skill_tree.get(parts[0])
        if not current:
            return []
        
        # Navigate down
        for part in parts[1:]:
            if "children" not in current:
                return []
            found = False
            for key, value in current["children"].items():
                if key.endswith(part):
                    current = value
                    found = True
                    break
            if not found:
                return []
        
        # Now we're at the subcategory, list its skills
        skills = []
        if "children" in current:
            for skill_id, skill_data in current["children"].items():
                skills.append({
                    "id": skill_id,
                    "name": skill_data.get("name", ""),
                    "description": skill_data.get("description", ""),
                    "governing_stats": skill_data.get("governing_stats", []),
                    "base_difficulty": skill_data.get("base_difficulty", 0.5)
                })
        
        return skills
    
    def get_skill_path(self, skill_id: str) -> List[str]:
        """
        Get full path from category to skill
        
        Example: "combat.melee.swords" → ["combat", "combat.melee", "combat.melee.swords"]
        """
        parts = skill_id.split(".")
        path = []
        
        for i in range(1, len(parts) + 1):
            path.append(".".join(parts[:i]))
        
        return path
    
    def get_skill_definition(self, skill_id: str) -> Optional[SkillDefinition]:
        """Get complete skill definition"""
        return self.skill_cache.get(skill_id)

# Global navigator
skill_navigator = SkillNavigator()
```

### Agent Interaction with Navigator

```python
class AgentSkillInterface:
    """Interface for agents to explore and use skills"""
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.navigator = skill_navigator
    
    async def present_skill_categories_to_llm(self) -> str:
        """
        Present top-level categories to agent's LLM
        
        Returns formatted string for LLM prompt
        """
        categories = self.navigator.get_categories()
        
        prompt = "Available Skill Categories:\n\n"
        for i, cat in enumerate(categories, 1):
            prompt += f"{i}. {cat['name']} - {cat['description']}\n"
            prompt += f"   Governing stats: {', '.join(cat['governing_stats'])}\n\n"
        
        prompt += "To explore a category, respond with: LIST [category_name]"
        
        return prompt
    
    async def expand_category(self, category_id: str) -> str:
        """
        Show subcategories within a category
        """
        subcats = self.navigator.get_subcategories(category_id)
        
        if not subcats:
            return f"No subcategories found in {category_id}"
        
        prompt = f"Subcategories in {category_id.title()}:\n\n"
        for i, subcat in enumerate(subcats, 1):
            prompt += f"{i}. {subcat['name']} - {subcat['description']}\n"
            prompt += f"   Governing stats: {', '.join(subcat['governing_stats'])}\n"
            prompt += f"   Difficulty: {subcat['base_difficulty']:.1f}\n"
            
            if subcat['has_children']:
                prompt += f"   (Has sub-skills - use LIST {subcat['id']})\n"
            
            prompt += "\n"
        
        prompt += "To use a skill, respond with: USE [skill_id]\n"
        prompt += "To see sub-skills, respond with: LIST [subcategory_id]"
        
        return prompt
    
    async def expand_subcategory(self, subcategory_id: str) -> str:
        """
        Show specific skills within a subcategory
        """
        skills = self.navigator.get_skills_in_subcategory(subcategory_id)
        
        if not skills:
            return f"No skills found in {subcategory_id}"
        
        prompt = f"Skills in {subcategory_id}:\n\n"
        for i, skill in enumerate(skills, 1):
            prompt += f"{i}. {skill['name']} - {skill['description']}\n"
            prompt += f"   Governing stats: {', '.join(skill['governing_stats'])}\n"
            prompt += f"   Difficulty: {skill['base_difficulty']:.1f}\n\n"
        
        prompt += "To use a skill, respond with: USE [skill_id] [description of action]"
        
        return prompt
```

---

## Skill Progression & XP

### SkillLevel Model

```python
class SkillLevel(BaseModel):
    """Agent's progress in a specific skill"""
    
    skill_id: str
    level: int = 0
    experience: float = 0.0
    times_used: int = 0
    success_count: int = 0
    fail_count: int = 0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate historical success rate"""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    @property
    def proficiency_bonus(self) -> float:
        """Bonus to success chance based on level (0.0 to 0.5)"""
        return min(self.level * 0.01, 0.5)
    
    def gain_experience(
        self,
        skill_def: SkillDefinition,
        success: bool,
        difficulty_modifier: float = 0.0
    ) -> Dict[str, Any]:
        """
        Grant XP and check for level up
        
        Returns info about XP gain and level ups
        """
        # Calculate XP
        base_xp = skill_def.xp_per_use
        
        # More XP for harder tasks
        difficulty_multiplier = 1.0 + (skill_def.base_difficulty + difficulty_modifier)
        
        # More XP for success, but still gain some on failure
        success_multiplier = 1.0 if success else 0.3
        
        xp_gained = base_xp * difficulty_multiplier * success_multiplier
        
        # Update tracking
        self.experience += xp_gained
        self.times_used += 1
        self.last_used = datetime.utcnow()
        
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        
        # Check for level up
        leveled_up = False
        old_level = self.level
        
        while (self.experience >= skill_def.xp_to_level and 
               self.level < skill_def.max_level):
            self.experience -= skill_def.xp_to_level
            self.level += 1
            leveled_up = True
        
        return {
            "xp_gained": xp_gained,
            "leveled_up": leveled_up,
            "old_level": old_level,
            "new_level": self.level
        }
```

---

## Stat Improvement System

Using skills improves governing stats over time.

```python
class StatSkillProgressionSystem:
    """Manages feedback loop between stats and skills"""
    
    @staticmethod
    async def process_skill_use(
        agent_state: "AgentState",
        skill_id: str,
        success: bool,
        difficulty_modifier: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process skill use with potential stat improvement
        
        Returns:
            - XP info
            - Stat improvement info
            - Updated skill level
        """
        skill_def = skill_navigator.get_skill_definition(skill_id)
        if not skill_def:
            return {"error": "Skill not found"}
        
        # Initialize skill if not present
        if skill_id not in agent_state.skills:
            agent_state.skills[skill_id] = SkillLevel(skill_id=skill_id)
        
        skill_level = agent_state.skills[skill_id]
        
        # Grant skill XP
        xp_info = skill_level.gain_experience(
            skill_def,
            success,
            difficulty_modifier
        )
        
        # Attempt stat improvements
        stat_improvements = {}
        
        # Check primary stat improvement
        improvement_chances = skill_def.get_stat_improvement_chances()
        
        for stat_name, chance in improvement_chances.items():
            # Successful skill use has higher chance to improve stats
            effective_chance = chance if success else chance * 0.5
            
            improved = agent_state.stats.improve_stat(stat_name, effective_chance)
            if improved:
                stat_improvements[stat_name] = {
                    "old_value": getattr(agent_state.stats, stat_name) - 1,
                    "new_value": getattr(agent_state.stats, stat_name)
                }
        
        # Parent skill XP sharing
        parent_xp_gained = 0
        if skill_def.parent_skill and skill_def.parent_skill in agent_state.skills:
            # Parent skill gains 30% of XP
            parent_skill_level = agent_state.skills[skill_def.parent_skill]
            parent_xp = xp_info["xp_gained"] * 0.3
            parent_skill_level.experience += parent_xp
            parent_xp_gained = parent_xp
        
        result = {
            "skill_progression": xp_info,
            "stat_improvements": stat_improvements,
            "parent_xp_gained": parent_xp_gained
        }
        
        # Store updated skill
        agent_state.skills[skill_id] = skill_level
        
        return result

# Global progression system
stat_skill_progression = StatSkillProgressionSystem()
```

---

## Skill Data Models Summary

All skill-related models for database storage:

```python
# Complete models for persistence

class SkillUsageLog(BaseModel):
    """Log entry for skill usage analytics"""
    log_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: UUID
    skill_id: str
    success: bool
    difficulty_modifier: float
    xp_gained: float
    stat_improvements: Dict[str, Dict] = {}
    level_before: int
    level_after: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentSkillState(BaseModel):
    """Agent's complete skill portfolio"""
    agent_id: UUID
    skills: Dict[str, SkillLevel] = {}  # skill_id -> level
    total_skills_learned: int = 0
    highest_skill_level: int = 0
    specialization_category: Optional[SkillCategory] = None
    
    def get_top_skills(self, limit: int = 5) -> List[tuple[str, SkillLevel]]:
        """Get agent's top skills by level"""
        sorted_skills = sorted(
            self.skills.items(),
            key=lambda x: x[1].level,
            reverse=True
        )
        return sorted_skills[:limit]
    
    def calculate_specialization(self) -> Optional[SkillCategory]:
        """
        Determine agent's specialization based on skill distribution
        
        Returns category with most high-level skills
        """
        category_levels: Dict[SkillCategory, int] = {}
        
        for skill_id, skill_level in self.skills.items():
            if skill_level.level < 5:  # Only count meaningful skills
                continue
            
            category = SkillCategory(skill_id.split(".")[0])
            category_levels[category] = category_levels.get(category, 0) + skill_level.level
        
        if not category_levels:
            return None
        
        return max(category_levels.items(), key=lambda x: x[1])[0]
```

---

## Summary

Part 2 establishes the skill system:

✅ **Hierarchical Taxonomy** - 12 categories, 40+ subcategories, 100+ skills
✅ **Governing Stats** - Every skill has primary (70%) and secondary (30%) stats
✅ **Top-Down Navigation** - Deterministic tree exploration (no search/embeddings)
✅ **Stat-Skill Feedback Loop** - Using skills improves governing stats
✅ **Parent-Child XP** - Child skills share 30% XP with parent skills
✅ **Natural Specialization** - Repeated practice creates expert professions

**Next:** Part 3 will cover the Skill Architect agent that creates new skills based on world context.