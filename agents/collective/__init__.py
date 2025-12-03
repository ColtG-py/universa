"""
Collective Agents Module
Implements settlement and kingdom-level agents that make decisions
affecting groups of individual agents without requiring individual LLM calls.
"""

from .settlement_agent import SettlementAgent, SettlementModifiers
from .kingdom_agent import KingdomAgent, KingdomPolicy
from .collective_manager import CollectiveManager

__all__ = [
    'SettlementAgent',
    'SettlementModifiers',
    'KingdomAgent',
    'KingdomPolicy',
    'CollectiveManager',
]
