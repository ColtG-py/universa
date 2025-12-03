"""
Memory Consolidation
Transfers information between memory systems.
Episodic → Semantic (episodes become facts)
Episodic → Procedural (episodes become procedures)
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from uuid import UUID
import asyncio

from agents.memory.episodic import EpisodicMemory, Episode
from agents.memory.semantic import SemanticMemory, Fact
from agents.memory.procedural import ProceduralMemory, Procedure
from agents.memory.memory_stream import MemoryStream
from agents.models.memory import Memory, Reflection
from agents.config import MemoryType


class MemoryConsolidator:
    """
    Consolidates memories across the three memory systems.

    This process:
    1. Extracts facts from episodic memories → semantic memory
    2. Extracts procedures from successful episodes → procedural memory
    3. Generates reflections from memory stream patterns
    4. Prunes old, low-importance memories
    """

    def __init__(
        self,
        agent_id: UUID,
        memory_stream: MemoryStream,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        procedural_memory: ProceduralMemory,
        fact_extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize consolidator.

        Args:
            agent_id: Agent ID
            memory_stream: Main memory stream
            episodic_memory: Episodic memory system
            semantic_memory: Semantic memory system
            procedural_memory: Procedural memory system
            fact_extractor: LLM function to extract facts from text
        """
        self.agent_id = agent_id
        self.stream = memory_stream
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.procedural = procedural_memory
        self._fact_extractor = fact_extractor

    def set_fact_extractor(
        self,
        extractor: Callable[[str], List[str]]
    ) -> None:
        """Set the fact extraction function (usually LLM-based)"""
        self._fact_extractor = extractor

    async def consolidate(
        self,
        hours_lookback: int = 24,
        extract_facts: bool = True,
        extract_procedures: bool = True,
        generate_reflections: bool = True,
        prune_old: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full memory consolidation.

        Args:
            hours_lookback: How far back to look
            extract_facts: Whether to extract semantic facts
            extract_procedures: Whether to extract procedures
            generate_reflections: Whether to generate reflections
            prune_old: Whether to prune old memories

        Returns:
            Consolidation results
        """
        results = {
            "facts_extracted": 0,
            "procedures_learned": 0,
            "reflections_generated": 0,
            "memories_pruned": 0,
        }

        # Get recent episodes
        episodes = await self.episodic.get_recent_episodes(hours_lookback)

        # Extract facts from episodes
        if extract_facts:
            for episode in episodes:
                facts = await self.extract_facts_from_episode(episode)
                results["facts_extracted"] += len(facts)

        # Extract procedures from successful episodes
        if extract_procedures:
            successful = [e for e in episodes if e.success and e.skills_used]
            for episode in successful:
                procedure = await self.extract_procedure_from_episode(episode)
                if procedure:
                    results["procedures_learned"] += 1

        # Generate reflections if threshold met
        if generate_reflections and self.stream.should_reflect():
            reflections = await self.generate_reflections()
            results["reflections_generated"] += len(reflections)

        # Prune old memories
        if prune_old:
            pruned = await self.prune_old_memories()
            results["memories_pruned"] = pruned

        return results

    async def extract_facts_from_episode(
        self,
        episode: Episode
    ) -> List[Fact]:
        """
        Extract semantic facts from an episodic memory.

        Args:
            episode: Episode to extract from

        Returns:
            List of extracted facts
        """
        facts = []

        # Use LLM extractor if available
        if self._fact_extractor:
            try:
                fact_texts = self._fact_extractor(episode.summary)
                for text in fact_texts:
                    fact = await self.semantic.store_fact(
                        fact_text=text,
                        source="experience",
                        confidence=0.8 if episode.success else 0.6
                    )
                    facts.append(fact)
                return facts
            except Exception:
                pass

        # Fallback: heuristic fact extraction
        facts_data = self._heuristic_fact_extraction(episode)
        for fact_text, category in facts_data:
            fact = await self.semantic.store_fact(
                fact_text=fact_text,
                category=category,
                source="experience",
                confidence=0.7 if episode.success else 0.5
            )
            facts.append(fact)

        return facts

    async def extract_procedure_from_episode(
        self,
        episode: Episode
    ) -> Optional[Procedure]:
        """
        Extract procedural knowledge from a successful episode.

        Args:
            episode: Successful episode

        Returns:
            Learned procedure if applicable
        """
        if not episode.success or not episode.skills_used:
            return None

        # Create procedure for each skill used
        for skill in episode.skills_used:
            procedure_name = f"how_to_{skill.replace('.', '_')}"

            procedure = await self.procedural.learn_procedure(
                procedure_name=procedure_name,
                procedure_prompt=f"From experience: {episode.summary}",
                related_skill_id=skill
            )

            return procedure  # Return first one for now

        return None

    async def generate_reflections(
        self,
        count: int = 5
    ) -> List[Reflection]:
        """
        Generate reflections from recent memories.

        Args:
            count: Number of reflections to generate

        Returns:
            Generated reflections
        """
        reflections = []

        # Get recent observations
        recent = await self.stream.retrieve_recent(
            hours=24,
            memory_types=[MemoryType.OBSERVATION]
        )

        if len(recent) < 5:
            return []  # Not enough memories

        # Group observations by topic (simplified)
        topics = self._cluster_by_topic(recent)

        for topic, memories in topics.items():
            if len(memories) >= 3:
                # Generate reflection for this topic
                reflection = await self._generate_topic_reflection(topic, memories)
                if reflection:
                    reflections.append(reflection)

                if len(reflections) >= count:
                    break

        return reflections

    async def prune_old_memories(
        self,
        days_old: int = 365,
        min_importance: float = 0.3
    ) -> int:
        """
        Prune old, low-importance memories.

        Args:
            days_old: Age threshold in days
            min_importance: Importance threshold

        Returns:
            Number of memories pruned
        """
        pruned = 0

        # Prune low-confidence facts
        pruned += await self.semantic.delete_low_confidence(threshold=0.2)

        # Could add pruning for other memory types

        return pruned

    async def strengthen_memory(
        self,
        memory: Memory
    ) -> None:
        """
        Strengthen a memory that was successfully used.

        Args:
            memory: Memory to strengthen
        """
        # Increase importance slightly
        new_importance = min(1.0, memory.importance + 0.05)
        memory.importance = new_importance

        # Update in database would go here

    async def decay_unused_memories(
        self,
        hours_since_access: int = 168  # 1 week
    ) -> int:
        """
        Decay memories that haven't been accessed.

        Args:
            hours_since_access: Hours threshold

        Returns:
            Number of memories decayed
        """
        # This would reduce importance of unused memories
        # Implementation depends on database capabilities
        return 0

    def _heuristic_fact_extraction(
        self,
        episode: Episode
    ) -> List[tuple]:
        """
        Extract facts using heuristics.

        Args:
            episode: Episode to extract from

        Returns:
            List of (fact_text, category) tuples
        """
        facts = []
        summary = episode.summary.lower()

        # Extract location facts
        if "at " in summary or "in " in summary:
            # Try to extract location
            pass

        # Extract relationship facts
        relationship_keywords = ["met", "talked to", "spoke with", "helped", "attacked"]
        for keyword in relationship_keywords:
            if keyword in summary:
                facts.append((
                    episode.summary,
                    "relationships"
                ))
                break

        # Extract skill facts
        if episode.skills_used:
            for skill in episode.skills_used:
                fact = f"I have used the skill {skill}"
                facts.append((fact, "skills"))

        # Success/failure facts
        if episode.success:
            if episode.skills_used:
                fact = f"I successfully used {episode.skills_used[0]}"
                facts.append((fact, "skills"))

        return facts

    def _cluster_by_topic(
        self,
        memories: List[Memory]
    ) -> Dict[str, List[Memory]]:
        """
        Cluster memories by topic for reflection.

        Args:
            memories: Memories to cluster

        Returns:
            Dictionary of topic -> memories
        """
        clusters = {}

        # Simple keyword-based clustering
        topic_keywords = {
            "social": ["met", "talked", "said", "asked", "told"],
            "work": ["made", "built", "crafted", "forged", "created"],
            "travel": ["went", "arrived", "traveled", "walked", "journey"],
            "conflict": ["fought", "attacked", "defended", "battle"],
            "learning": ["learned", "discovered", "realized", "understood"],
        }

        for memory in memories:
            desc_lower = memory.description.lower()
            assigned = False

            for topic, keywords in topic_keywords.items():
                if any(kw in desc_lower for kw in keywords):
                    if topic not in clusters:
                        clusters[topic] = []
                    clusters[topic].append(memory)
                    assigned = True
                    break

            if not assigned:
                if "general" not in clusters:
                    clusters["general"] = []
                clusters["general"].append(memory)

        return clusters

    async def _generate_topic_reflection(
        self,
        topic: str,
        memories: List[Memory]
    ) -> Optional[Reflection]:
        """
        Generate a reflection for a topic cluster.

        Args:
            topic: Topic name
            memories: Related memories

        Returns:
            Generated reflection
        """
        # Combine memory descriptions
        descriptions = [m.description for m in memories[:10]]
        combined = " ".join(descriptions)

        # Generate reflection text (would use LLM in production)
        reflection_text = self._heuristic_reflection(topic, descriptions)

        if not reflection_text:
            return None

        # Create reflection in memory stream
        source_ids = [m.memory_id for m in memories[:5]]
        reflection = await self.stream.add_reflection(
            description=reflection_text,
            source_memories=source_ids,
            importance=0.75
        )

        return reflection

    def _heuristic_reflection(
        self,
        topic: str,
        descriptions: List[str]
    ) -> Optional[str]:
        """
        Generate reflection using heuristics.

        Args:
            topic: Topic name
            descriptions: Memory descriptions

        Returns:
            Reflection text
        """
        if len(descriptions) < 3:
            return None

        templates = {
            "social": "I have been socially active lately, interacting with several people.",
            "work": "I have been productive, working on various tasks and projects.",
            "travel": "I have been moving around and exploring different places.",
            "conflict": "I have been involved in some conflicts or dangerous situations.",
            "learning": "I have been learning and discovering new things.",
            "general": "Many things have happened recently.",
        }

        return templates.get(topic, templates["general"])


class ConsolidationScheduler:
    """
    Schedules periodic memory consolidation.
    """

    def __init__(
        self,
        consolidator: MemoryConsolidator,
        interval_hours: int = 6
    ):
        """
        Initialize scheduler.

        Args:
            consolidator: Memory consolidator
            interval_hours: Hours between consolidations
        """
        self.consolidator = consolidator
        self.interval = timedelta(hours=interval_hours)
        self.last_run: Optional[datetime] = None
        self._running = False

    async def should_run(self) -> bool:
        """Check if consolidation should run"""
        if self.last_run is None:
            return True

        return datetime.utcnow() - self.last_run >= self.interval

    async def run_if_needed(self) -> Optional[Dict[str, Any]]:
        """
        Run consolidation if enough time has passed.

        Returns:
            Results if ran, None if skipped
        """
        if not await self.should_run():
            return None

        if self._running:
            return None

        self._running = True
        try:
            results = await self.consolidator.consolidate()
            self.last_run = datetime.utcnow()
            return results
        finally:
            self._running = False

    async def force_run(self) -> Dict[str, Any]:
        """Force immediate consolidation"""
        self._running = True
        try:
            results = await self.consolidator.consolidate()
            self.last_run = datetime.utcnow()
            return results
        finally:
            self._running = False
