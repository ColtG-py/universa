"""
Behavior Consistency Checker
Evaluates agent behavior consistency over time.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
from collections import Counter

from agents.llm.ollama_client import OllamaClient


class ConsistencyDimension(str, Enum):
    """Dimensions of behavioral consistency"""
    PERSONALITY = "personality"
    OCCUPATION = "occupation"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    GOAL_ORIENTED = "goal_oriented"


@dataclass
class BehaviorPattern:
    """A detected behavioral pattern"""
    pattern_id: UUID = field(default_factory=uuid4)
    description: str = ""
    frequency: int = 0
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    contexts: List[str] = field(default_factory=list)


@dataclass
class ConsistencyResult:
    """Result of a consistency check"""
    result_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    agent_name: str = ""

    # Dimension scores (0-10)
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Detected patterns
    patterns: List[BehaviorPattern] = field(default_factory=list)

    # Inconsistencies found
    inconsistencies: List[str] = field(default_factory=list)

    # Overall score
    overall_consistency: float = 0.0

    # Analysis metadata
    actions_analyzed: int = 0
    time_span_hours: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "result_id": str(self.result_id),
            "agent_id": str(self.agent_id),
            "agent_name": self.agent_name,
            "dimension_scores": self.dimension_scores,
            "overall_consistency": self.overall_consistency,
            "patterns_found": len(self.patterns),
            "inconsistencies_found": len(self.inconsistencies),
            "actions_analyzed": self.actions_analyzed,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Behavior Consistency Report for {self.agent_name}",
            "=" * 40,
            f"Overall Consistency: {self.overall_consistency:.1f}/10",
            "",
            "Dimension Scores:",
        ]
        for dim, score in sorted(self.dimension_scores.items()):
            lines.append(f"  {dim}: {score:.1f}/10")

        if self.patterns:
            lines.append("")
            lines.append(f"Detected Patterns ({len(self.patterns)}):")
            for p in self.patterns[:5]:
                lines.append(f"  - {p.description} (freq: {p.frequency})")

        if self.inconsistencies:
            lines.append("")
            lines.append(f"Inconsistencies ({len(self.inconsistencies)}):")
            for inc in self.inconsistencies[:3]:
                lines.append(f"  ! {inc}")

        return "\n".join(lines)


class BehaviorConsistencyChecker:
    """
    Checks agent behavior consistency.

    Evaluates:
    - Personality consistency: Do actions match stated traits?
    - Occupation consistency: Does agent perform job-related tasks?
    - Social consistency: Are relationships maintained appropriately?
    - Temporal consistency: Do behaviors match time of day/season?
    - Goal consistency: Does agent work toward stated goals?
    """

    # Consistency evaluation prompt
    CONSISTENCY_PROMPT = """Evaluate behavioral consistency for this agent.

Agent Summary:
{agent_summary}

Recent Actions (chronological):
{actions}

Known Traits: {traits}
Occupation: {occupation}
Relationships: {relationships}

Evaluate consistency (0-10) for:
1. Personality: Do actions match the agent's personality traits?
2. Occupation: Are actions appropriate for their job/role?
3. Social: Are social interactions appropriate for relationships?
4. Temporal: Do actions make sense for the time of day?
5. Goal-oriented: Is the agent working toward any apparent goals?

Also identify:
- Behavioral patterns (repeated behaviors)
- Inconsistencies (behaviors that don't fit)

Format:
personality: X
occupation: X
social: X
temporal: X
goal_oriented: X
patterns: [pattern1, pattern2, ...]
inconsistencies: [inconsistency1, inconsistency2, ...]"""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize consistency checker.

        Args:
            ollama_client: LLM client for evaluation
        """
        self.client = ollama_client
        self._check_history: List[ConsistencyResult] = []

    async def check_consistency(
        self,
        agent_id: UUID,
        agent_name: str,
        agent_summary: str,
        actions: List[Dict[str, Any]],
        traits: Optional[List[str]] = None,
        occupation: Optional[str] = None,
        relationships: Optional[Dict[str, str]] = None,
    ) -> ConsistencyResult:
        """
        Check behavioral consistency for an agent.

        Args:
            agent_id: Agent's ID
            agent_name: Agent's name
            agent_summary: Agent's summary description
            actions: List of recent actions with timestamps
            traits: Known personality traits
            occupation: Agent's occupation
            relationships: Known relationships

        Returns:
            Consistency check result
        """
        result = ConsistencyResult(
            agent_id=agent_id,
            agent_name=agent_name,
            actions_analyzed=len(actions),
        )

        if not actions:
            result.overall_consistency = 5.0  # Neutral if no data
            return result

        # Calculate time span
        if len(actions) > 1:
            first_time = actions[0].get("timestamp", datetime.utcnow())
            last_time = actions[-1].get("timestamp", datetime.utcnow())
            if isinstance(first_time, str):
                first_time = datetime.fromisoformat(first_time)
            if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time)
            result.time_span_hours = (last_time - first_time).total_seconds() / 3600

        # Analyze with LLM if available
        if self.client:
            scores, patterns, inconsistencies = await self._llm_analyze(
                agent_summary=agent_summary,
                actions=actions,
                traits=traits or [],
                occupation=occupation or "unknown",
                relationships=relationships or {},
            )
            result.dimension_scores = scores
            result.patterns = patterns
            result.inconsistencies = inconsistencies
        else:
            result.dimension_scores = self._heuristic_analyze(
                actions=actions,
                traits=traits or [],
                occupation=occupation or "unknown",
            )
            result.patterns = self._detect_patterns(actions)

        # Calculate overall score
        if result.dimension_scores:
            result.overall_consistency = sum(result.dimension_scores.values()) / len(
                result.dimension_scores
            )

        self._check_history.append(result)
        return result

    async def _llm_analyze(
        self,
        agent_summary: str,
        actions: List[Dict[str, Any]],
        traits: List[str],
        occupation: str,
        relationships: Dict[str, str],
    ) -> Tuple[Dict[str, float], List[BehaviorPattern], List[str]]:
        """Analyze consistency using LLM"""
        actions_text = "\n".join(
            f"- [{a.get('timestamp', 'unknown')}] {a.get('description', a.get('action', str(a)))}"
            for a in actions[:20]  # Limit to recent actions
        )

        relationships_text = ", ".join(
            f"{name}: {rel}" for name, rel in relationships.items()
        ) or "None known"

        prompt = self.CONSISTENCY_PROMPT.format(
            agent_summary=agent_summary,
            actions=actions_text,
            traits=", ".join(traits) or "Not specified",
            occupation=occupation,
            relationships=relationships_text,
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=400,
            )
            return self._parse_llm_response(response.text)
        except Exception:
            return (
                self._heuristic_analyze(actions, traits, occupation),
                self._detect_patterns(actions),
                [],
            )

    def _parse_llm_response(
        self,
        response: str
    ) -> Tuple[Dict[str, float], List[BehaviorPattern], List[str]]:
        """Parse LLM response"""
        import re

        # Parse dimension scores
        scores = {}
        dimensions = ["personality", "occupation", "social", "temporal", "goal_oriented"]

        for dim in dimensions:
            match = re.search(rf"{dim}:\s*(\d+(?:\.\d+)?)", response.lower())
            if match:
                scores[dim] = min(10.0, float(match.group(1)))
            else:
                scores[dim] = 5.0

        # Parse patterns
        patterns = []
        pattern_match = re.search(r"patterns:\s*\[(.*?)\]", response.lower())
        if pattern_match:
            pattern_strs = pattern_match.group(1).split(",")
            for p in pattern_strs:
                p = p.strip().strip("'\"")
                if p:
                    patterns.append(BehaviorPattern(description=p, frequency=1))

        # Parse inconsistencies
        inconsistencies = []
        inc_match = re.search(r"inconsistencies:\s*\[(.*?)\]", response.lower())
        if inc_match:
            inc_strs = inc_match.group(1).split(",")
            for inc in inc_strs:
                inc = inc.strip().strip("'\"")
                if inc:
                    inconsistencies.append(inc)

        return scores, patterns, inconsistencies

    def _heuristic_analyze(
        self,
        actions: List[Dict[str, Any]],
        traits: List[str],
        occupation: str,
    ) -> Dict[str, float]:
        """Heuristic consistency analysis"""
        scores = {
            "personality": 5.0,
            "occupation": 5.0,
            "social": 5.0,
            "temporal": 5.0,
            "goal_oriented": 5.0,
        }

        if not actions:
            return scores

        # Extract action descriptions
        action_texts = [
            a.get("description", a.get("action", str(a))).lower()
            for a in actions
        ]

        # Personality: Check trait keywords in actions
        if traits:
            trait_matches = sum(
                1 for action in action_texts
                for trait in traits
                if trait.lower() in action
            )
            scores["personality"] = min(10.0, 5.0 + trait_matches)

        # Occupation: Check occupation keywords
        occupation_keywords = occupation.lower().split()
        occupation_matches = sum(
            1 for action in action_texts
            if any(kw in action for kw in occupation_keywords)
        )
        scores["occupation"] = min(10.0, 3.0 + occupation_matches * 2)

        # Social: Check social interaction keywords
        social_keywords = ["talk", "spoke", "met", "greeted", "said", "asked", "told"]
        social_count = sum(
            1 for action in action_texts
            if any(kw in action for kw in social_keywords)
        )
        scores["social"] = min(10.0, 4.0 + social_count)

        # Temporal: Check time-appropriate actions (simplified)
        # This would ideally check action times against appropriate activities
        scores["temporal"] = 6.0  # Default reasonable score

        # Goal-oriented: Check for progressive/purposeful actions
        purposeful_keywords = ["finished", "completed", "made", "created", "worked", "continued"]
        purposeful_count = sum(
            1 for action in action_texts
            if any(kw in action for kw in purposeful_keywords)
        )
        scores["goal_oriented"] = min(10.0, 4.0 + purposeful_count * 1.5)

        return scores

    def _detect_patterns(
        self,
        actions: List[Dict[str, Any]]
    ) -> List[BehaviorPattern]:
        """Detect behavioral patterns in actions"""
        patterns = []

        # Extract action types/categories
        action_types = []
        for action in actions:
            desc = action.get("description", action.get("action", "")).lower()

            # Categorize action
            if any(w in desc for w in ["eat", "drink", "breakfast", "lunch", "dinner"]):
                action_types.append("eating")
            elif any(w in desc for w in ["work", "forge", "craft", "make"]):
                action_types.append("working")
            elif any(w in desc for w in ["talk", "spoke", "conversation"]):
                action_types.append("socializing")
            elif any(w in desc for w in ["walk", "travel", "went", "arrived"]):
                action_types.append("moving")
            elif any(w in desc for w in ["rest", "sleep", "relax"]):
                action_types.append("resting")
            else:
                action_types.append("other")

        # Count frequencies
        type_counts = Counter(action_types)

        for action_type, count in type_counts.most_common(5):
            if count >= 2:
                patterns.append(BehaviorPattern(
                    description=f"Frequently {action_type}",
                    frequency=count,
                ))

        return patterns

    def detect_anomalies(
        self,
        agent_id: UUID,
        current_action: str,
        action_history: List[str],
    ) -> List[str]:
        """
        Detect anomalous behaviors.

        Args:
            agent_id: Agent's ID
            current_action: Current action to check
            action_history: Recent action history

        Returns:
            List of detected anomalies
        """
        anomalies = []

        current_lower = current_action.lower()
        history_lower = [a.lower() for a in action_history]

        # Check for sudden behavioral shifts
        # (This is a simplified check - could be more sophisticated)

        # Violent action after peaceful history
        violent_words = ["attack", "kill", "fight", "hurt", "destroy"]
        peaceful_words = ["help", "talk", "work", "rest", "eat"]

        is_violent = any(w in current_lower for w in violent_words)
        was_peaceful = all(
            any(p in h for p in peaceful_words) and
            not any(v in h for v in violent_words)
            for h in history_lower[-5:]
        ) if history_lower else True

        if is_violent and was_peaceful:
            anomalies.append("Sudden violent action after peaceful behavior")

        # Activity at unusual times
        # (Would need timestamp information for proper check)

        # Repeated identical actions (stuck in loop)
        if history_lower and len(history_lower) >= 3:
            if history_lower[-1] == history_lower[-2] == history_lower[-3]:
                if current_lower == history_lower[-1]:
                    anomalies.append("Repetitive behavior detected (possible stuck loop)")

        return anomalies

    def compare_to_baseline(
        self,
        result: ConsistencyResult,
        baseline: Optional[ConsistencyResult] = None,
    ) -> Dict[str, float]:
        """
        Compare current consistency to a baseline.

        Args:
            result: Current consistency result
            baseline: Baseline to compare against

        Returns:
            Dict of dimension changes
        """
        if not baseline:
            return {}

        changes = {}
        for dim in result.dimension_scores:
            if dim in baseline.dimension_scores:
                changes[dim] = result.dimension_scores[dim] - baseline.dimension_scores[dim]

        return changes

    def get_check_history(
        self,
        agent_id: Optional[UUID] = None
    ) -> List[ConsistencyResult]:
        """Get check history, optionally filtered by agent"""
        if agent_id:
            return [r for r in self._check_history if r.agent_id == agent_id]
        return self._check_history

    def get_consistency_trend(
        self,
        agent_id: UUID,
        dimension: str = "overall",
    ) -> List[Tuple[datetime, float]]:
        """
        Get consistency scores over time for an agent.

        Args:
            agent_id: Agent's ID
            dimension: Which dimension to track

        Returns:
            List of (timestamp, score) tuples
        """
        history = self.get_check_history(agent_id)

        trend = []
        for result in history:
            if dimension == "overall":
                score = result.overall_consistency
            else:
                score = result.dimension_scores.get(dimension, 0.0)

            trend.append((result.timestamp, score))

        return sorted(trend, key=lambda x: x[0])
