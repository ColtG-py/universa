"""
Believability Metrics
Comprehensive metrics for evaluating agent believability.
Based on Stanford Generative Agents paper success criteria.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


class MetricCategory(str, Enum):
    """Categories of believability metrics"""
    SELF_KNOWLEDGE = "self_knowledge"
    MEMORY = "memory"
    PLANNING = "planning"
    REACTIONS = "reactions"
    REFLECTIONS = "reflections"
    INFORMATION_DIFFUSION = "information_diffusion"
    RELATIONSHIP_FORMATION = "relationship_formation"
    COORDINATION = "coordination"
    SPECIALIZATION = "specialization"


@dataclass
class MetricScore:
    """A single metric score"""
    metric_id: UUID = field(default_factory=uuid4)
    category: MetricCategory = MetricCategory.SELF_KNOWLEDGE
    name: str = ""
    description: str = ""
    score: float = 0.0  # 0-10 scale
    max_score: float = 10.0
    weight: float = 1.0
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1"""
        return self.score / self.max_score if self.max_score > 0 else 0.0

    @property
    def weighted_score(self) -> float:
        """Score multiplied by weight"""
        return self.normalized_score * self.weight


@dataclass
class EvaluationReport:
    """Complete evaluation report for an agent"""
    report_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    agent_name: str = ""

    # Individual metrics
    metrics: List[MetricScore] = field(default_factory=list)

    # Category summaries
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Overall scores
    overall_believability: float = 0.0
    overall_emergence: float = 0.0
    overall_technical: float = 0.0

    # Metadata
    simulation_ticks: int = 0
    evaluation_duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": str(self.report_id),
            "agent_id": str(self.agent_id),
            "agent_name": self.agent_name,
            "overall_believability": self.overall_believability,
            "overall_emergence": self.overall_emergence,
            "overall_technical": self.overall_technical,
            "category_scores": self.category_scores,
            "metric_count": len(self.metrics),
            "simulation_ticks": self.simulation_ticks,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Evaluation Report for {self.agent_name}",
            "=" * 40,
            f"Overall Believability: {self.overall_believability:.1f}/10",
            f"Overall Emergence: {self.overall_emergence:.1f}/10",
            f"Overall Technical: {self.overall_technical:.1f}/10",
            "",
            "Category Scores:",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat}: {score:.1f}/10")

        return "\n".join(lines)


class BelievabilityMetrics:
    """
    Calculates believability metrics for agents.

    From the Stanford paper, believability is measured by:
    1. Self-knowledge: Can agents accurately describe themselves?
    2. Memory: Can agents recall specific past events and people?
    3. Planning: Do agents maintain coherent long-term plans?
    4. Reactions: Do agents respond appropriately to events?
    5. Reflections: Can agents synthesize higher-level insights?

    Emergent behavior is measured by:
    1. Information Diffusion: Does news spread through the population?
    2. Relationship Formation: Do new relationships form over time?
    3. Coordination: Can agents organize group activities?
    4. Specialization: Do agents develop distinct skill profiles?
    """

    # Metric definitions with descriptions and weights
    METRIC_DEFINITIONS = {
        # Believability metrics (from paper)
        "self_description_accuracy": {
            "category": MetricCategory.SELF_KNOWLEDGE,
            "description": "Agent can accurately describe their own traits, occupation, and relationships",
            "weight": 1.0,
        },
        "trait_consistency": {
            "category": MetricCategory.SELF_KNOWLEDGE,
            "description": "Agent's actions align with their stated personality traits",
            "weight": 1.0,
        },
        "memory_recall_accuracy": {
            "category": MetricCategory.MEMORY,
            "description": "Agent can recall specific events that occurred",
            "weight": 1.2,
        },
        "memory_person_accuracy": {
            "category": MetricCategory.MEMORY,
            "description": "Agent can recall details about people they've interacted with",
            "weight": 1.0,
        },
        "memory_temporal_ordering": {
            "category": MetricCategory.MEMORY,
            "description": "Agent correctly orders events in time",
            "weight": 0.8,
        },
        "plan_coherence": {
            "category": MetricCategory.PLANNING,
            "description": "Agent's plans are logical and achievable",
            "weight": 1.0,
        },
        "plan_execution": {
            "category": MetricCategory.PLANNING,
            "description": "Agent follows through on stated plans",
            "weight": 1.2,
        },
        "plan_adaptation": {
            "category": MetricCategory.PLANNING,
            "description": "Agent appropriately adjusts plans when circumstances change",
            "weight": 0.8,
        },
        "reaction_appropriateness": {
            "category": MetricCategory.REACTIONS,
            "description": "Agent reacts appropriately to events (not over/under-reacting)",
            "weight": 1.0,
        },
        "reaction_speed": {
            "category": MetricCategory.REACTIONS,
            "description": "Agent reacts in a timely manner to important events",
            "weight": 0.8,
        },
        "reflection_depth": {
            "category": MetricCategory.REFLECTIONS,
            "description": "Agent generates meaningful insights from experiences",
            "weight": 1.0,
        },
        "reflection_accuracy": {
            "category": MetricCategory.REFLECTIONS,
            "description": "Agent's reflections are grounded in actual experiences",
            "weight": 1.0,
        },

        # Emergent behavior metrics
        "information_spread": {
            "category": MetricCategory.INFORMATION_DIFFUSION,
            "description": "Information shared by agent propagates through network",
            "weight": 0.8,
        },
        "gossip_accuracy": {
            "category": MetricCategory.INFORMATION_DIFFUSION,
            "description": "Information remains accurate as it spreads",
            "weight": 0.6,
        },
        "new_relationships": {
            "category": MetricCategory.RELATIONSHIP_FORMATION,
            "description": "Agent forms new relationships over time",
            "weight": 0.8,
        },
        "relationship_depth": {
            "category": MetricCategory.RELATIONSHIP_FORMATION,
            "description": "Agent develops deeper relationships with some individuals",
            "weight": 0.8,
        },
        "group_participation": {
            "category": MetricCategory.COORDINATION,
            "description": "Agent participates in group activities",
            "weight": 0.6,
        },
        "initiative_taking": {
            "category": MetricCategory.COORDINATION,
            "description": "Agent initiates coordination with others",
            "weight": 0.6,
        },
        "skill_development": {
            "category": MetricCategory.SPECIALIZATION,
            "description": "Agent develops expertise in specific skills",
            "weight": 0.8,
        },
        "role_consistency": {
            "category": MetricCategory.SPECIALIZATION,
            "description": "Agent maintains consistent role/occupation",
            "weight": 0.6,
        },
    }

    def __init__(self):
        """Initialize metrics calculator"""
        self._evaluation_cache: Dict[UUID, EvaluationReport] = {}

    def calculate_metric(
        self,
        metric_name: str,
        evidence: List[str],
        score: float,
    ) -> MetricScore:
        """
        Calculate a single metric.

        Args:
            metric_name: Name of the metric
            evidence: Evidence supporting the score
            score: Score value (0-10)

        Returns:
            Calculated metric score
        """
        if metric_name not in self.METRIC_DEFINITIONS:
            raise ValueError(f"Unknown metric: {metric_name}")

        definition = self.METRIC_DEFINITIONS[metric_name]

        return MetricScore(
            category=definition["category"],
            name=metric_name,
            description=definition["description"],
            score=min(10.0, max(0.0, score)),
            weight=definition["weight"],
            evidence=evidence,
        )

    def evaluate_self_knowledge(
        self,
        agent_summary: str,
        self_description: str,
        action_history: List[str],
    ) -> List[MetricScore]:
        """
        Evaluate agent's self-knowledge.

        Args:
            agent_summary: True agent summary
            self_description: Agent's self-description
            action_history: Recent actions taken

        Returns:
            List of metric scores
        """
        metrics = []

        # Self-description accuracy
        # Check overlap between summary keywords and self-description
        summary_words = set(agent_summary.lower().split())
        desc_words = set(self_description.lower().split())
        overlap = len(summary_words & desc_words)
        accuracy_score = min(10.0, overlap / 5 * 10)

        metrics.append(self.calculate_metric(
            "self_description_accuracy",
            evidence=[
                f"Summary overlap: {overlap} words",
                f"Self-description length: {len(self_description)} chars",
            ],
            score=accuracy_score,
        ))

        # Trait consistency
        # This would ideally compare actions to stated traits
        # Simplified: check if actions seem varied and purposeful
        unique_actions = len(set(action_history))
        consistency_score = min(10.0, unique_actions / 3 * 10) if action_history else 5.0

        metrics.append(self.calculate_metric(
            "trait_consistency",
            evidence=[
                f"Unique actions: {unique_actions}",
                f"Total actions: {len(action_history)}",
            ],
            score=consistency_score,
        ))

        return metrics

    def evaluate_memory(
        self,
        actual_events: List[str],
        recalled_events: List[str],
        known_people: List[str],
        recalled_people: List[str],
    ) -> List[MetricScore]:
        """
        Evaluate agent's memory accuracy.

        Args:
            actual_events: Events that actually occurred
            recalled_events: Events the agent recalled
            known_people: People the agent should know
            recalled_people: People the agent recalled

        Returns:
            List of metric scores
        """
        metrics = []

        # Memory recall accuracy
        if actual_events:
            recall_overlap = sum(
                1 for r in recalled_events
                if any(a.lower() in r.lower() or r.lower() in a.lower()
                       for a in actual_events)
            )
            recall_score = min(10.0, recall_overlap / len(actual_events) * 10)
        else:
            recall_score = 5.0

        metrics.append(self.calculate_metric(
            "memory_recall_accuracy",
            evidence=[
                f"Actual events: {len(actual_events)}",
                f"Recalled events: {len(recalled_events)}",
                f"Overlap: {recall_overlap if actual_events else 'N/A'}",
            ],
            score=recall_score,
        ))

        # Person recall accuracy
        if known_people:
            person_overlap = sum(
                1 for r in recalled_people
                if any(p.lower() in r.lower() for p in known_people)
            )
            person_score = min(10.0, person_overlap / len(known_people) * 10)
        else:
            person_score = 5.0

        metrics.append(self.calculate_metric(
            "memory_person_accuracy",
            evidence=[
                f"Known people: {len(known_people)}",
                f"Recalled: {len(recalled_people)}",
            ],
            score=person_score,
        ))

        return metrics

    def evaluate_planning(
        self,
        stated_plans: List[str],
        executed_actions: List[str],
        plan_changes: int,
        circumstances_changed: int,
    ) -> List[MetricScore]:
        """
        Evaluate agent's planning abilities.

        Args:
            stated_plans: Plans the agent stated
            executed_actions: Actions actually taken
            plan_changes: Number of times plans changed
            circumstances_changed: Number of circumstance changes

        Returns:
            List of metric scores
        """
        metrics = []

        # Plan coherence (do plans make sense)
        coherence_score = min(10.0, len(stated_plans) * 2) if stated_plans else 3.0
        metrics.append(self.calculate_metric(
            "plan_coherence",
            evidence=[f"Stated {len(stated_plans)} plans"],
            score=coherence_score,
        ))

        # Plan execution (did agent follow through)
        if stated_plans and executed_actions:
            execution_ratio = len(executed_actions) / max(1, len(stated_plans))
            execution_score = min(10.0, execution_ratio * 8)
        else:
            execution_score = 5.0

        metrics.append(self.calculate_metric(
            "plan_execution",
            evidence=[
                f"Plans: {len(stated_plans)}",
                f"Actions: {len(executed_actions)}",
            ],
            score=execution_score,
        ))

        # Plan adaptation
        if circumstances_changed > 0:
            adaptation_ratio = plan_changes / circumstances_changed
            adaptation_score = min(10.0, adaptation_ratio * 10)
        else:
            adaptation_score = 7.0  # No need to adapt

        metrics.append(self.calculate_metric(
            "plan_adaptation",
            evidence=[
                f"Circumstance changes: {circumstances_changed}",
                f"Plan adjustments: {plan_changes}",
            ],
            score=adaptation_score,
        ))

        return metrics

    def evaluate_reactions(
        self,
        events: List[Dict[str, Any]],
        reactions: List[Dict[str, Any]],
    ) -> List[MetricScore]:
        """
        Evaluate agent's reactions to events.

        Args:
            events: Events that occurred (with importance)
            reactions: Agent's reactions (with timing)

        Returns:
            List of metric scores
        """
        metrics = []

        # Reaction appropriateness
        if events and reactions:
            # Check if high-importance events got reactions
            high_importance = [e for e in events if e.get("importance", 0) > 0.7]
            reacted_to = len([r for r in reactions if r.get("event_id")])
            appropriateness = min(10.0, reacted_to / max(1, len(high_importance)) * 10)
        else:
            appropriateness = 5.0

        metrics.append(self.calculate_metric(
            "reaction_appropriateness",
            evidence=[
                f"Events: {len(events)}",
                f"Reactions: {len(reactions)}",
            ],
            score=appropriateness,
        ))

        # Reaction speed
        if reactions:
            avg_delay = sum(r.get("delay_ms", 1000) for r in reactions) / len(reactions)
            speed_score = max(0.0, 10.0 - avg_delay / 1000)  # Penalize delays > 10s
        else:
            speed_score = 5.0

        metrics.append(self.calculate_metric(
            "reaction_speed",
            evidence=[f"Average reaction time: {avg_delay if reactions else 'N/A'}ms"],
            score=speed_score,
        ))

        return metrics

    def evaluate_reflections(
        self,
        reflections: List[str],
        source_memories: List[List[str]],
    ) -> List[MetricScore]:
        """
        Evaluate agent's reflection quality.

        Args:
            reflections: Generated reflections
            source_memories: Source memories for each reflection

        Returns:
            List of metric scores
        """
        metrics = []

        # Reflection depth (are reflections substantive)
        if reflections:
            avg_length = sum(len(r.split()) for r in reflections) / len(reflections)
            depth_score = min(10.0, avg_length / 10 * 10)  # Expect ~10 words
        else:
            depth_score = 0.0

        metrics.append(self.calculate_metric(
            "reflection_depth",
            evidence=[
                f"Reflections: {len(reflections)}",
                f"Avg length: {avg_length if reflections else 0} words",
            ],
            score=depth_score,
        ))

        # Reflection accuracy (grounded in memories)
        if reflections and source_memories:
            grounded = sum(1 for sources in source_memories if sources)
            accuracy_score = min(10.0, grounded / len(reflections) * 10)
        else:
            accuracy_score = 5.0

        metrics.append(self.calculate_metric(
            "reflection_accuracy",
            evidence=[
                f"Grounded reflections: {grounded if reflections else 0}",
            ],
            score=accuracy_score,
        ))

        return metrics

    def generate_report(
        self,
        agent_id: UUID,
        agent_name: str,
        metrics: List[MetricScore],
        simulation_ticks: int = 0,
    ) -> EvaluationReport:
        """
        Generate a complete evaluation report.

        Args:
            agent_id: Agent's ID
            agent_name: Agent's name
            metrics: List of calculated metrics
            simulation_ticks: Number of simulation ticks elapsed

        Returns:
            Complete evaluation report
        """
        report = EvaluationReport(
            agent_id=agent_id,
            agent_name=agent_name,
            metrics=metrics,
            simulation_ticks=simulation_ticks,
        )

        # Calculate category scores
        category_metrics: Dict[str, List[MetricScore]] = {}
        for metric in metrics:
            cat = metric.category.value
            if cat not in category_metrics:
                category_metrics[cat] = []
            category_metrics[cat].append(metric)

        for cat, cat_metrics in category_metrics.items():
            if cat_metrics:
                total_weight = sum(m.weight for m in cat_metrics)
                weighted_sum = sum(m.score * m.weight for m in cat_metrics)
                report.category_scores[cat] = weighted_sum / total_weight

        # Calculate overall scores
        believability_categories = [
            MetricCategory.SELF_KNOWLEDGE.value,
            MetricCategory.MEMORY.value,
            MetricCategory.PLANNING.value,
            MetricCategory.REACTIONS.value,
            MetricCategory.REFLECTIONS.value,
        ]
        emergence_categories = [
            MetricCategory.INFORMATION_DIFFUSION.value,
            MetricCategory.RELATIONSHIP_FORMATION.value,
            MetricCategory.COORDINATION.value,
            MetricCategory.SPECIALIZATION.value,
        ]

        believability_scores = [
            report.category_scores[c]
            for c in believability_categories
            if c in report.category_scores
        ]
        emergence_scores = [
            report.category_scores[c]
            for c in emergence_categories
            if c in report.category_scores
        ]

        if believability_scores:
            report.overall_believability = sum(believability_scores) / len(believability_scores)

        if emergence_scores:
            report.overall_emergence = sum(emergence_scores) / len(emergence_scores)

        # Technical score based on response times and completeness
        all_scores = [m.score for m in metrics]
        if all_scores:
            report.overall_technical = sum(all_scores) / len(all_scores)

        # Cache the report
        self._evaluation_cache[agent_id] = report

        return report

    def get_cached_report(self, agent_id: UUID) -> Optional[EvaluationReport]:
        """Get cached evaluation report for an agent"""
        return self._evaluation_cache.get(agent_id)
