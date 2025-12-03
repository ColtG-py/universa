"""
Universa Agents - Generative Agent System
Based on the Stanford "Generative Agents" paper by Park et al.

This module provides autonomous agents that can:
- Inhabit and interact with the procedurally generated world
- Develop personalities, skills, and relationships over time
- Form memories, make plans, and reflect on experiences
- Coordinate with other agents and respond to environmental changes
"""

from agents.models.agent_state import AgentState, CoreStats, CoreNeeds, Alignment

__version__ = "0.1.0"
__all__ = ["AgentState", "CoreStats", "CoreNeeds", "Alignment"]
