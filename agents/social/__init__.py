"""
Social Systems Module
Agent relationships, interactions, and information diffusion.
"""

from agents.social.relationships import (
    Relationship,
    RelationshipType,
    RelationshipManager,
)
from agents.social.interactions import (
    Interaction,
    InteractionType,
    InteractionManager,
)
from agents.social.information import (
    Information,
    InformationType,
    InformationNetwork,
)

__all__ = [
    "Relationship",
    "RelationshipType",
    "RelationshipManager",
    "Interaction",
    "InteractionType",
    "InteractionManager",
    "Information",
    "InformationType",
    "InformationNetwork",
]
