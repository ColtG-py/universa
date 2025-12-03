"""
Prompt Templates
LLM prompts for agent operations based on Stanford Generative Agents paper.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class PromptTemplates:
    """
    Collection of prompt templates for generative agent operations.
    Based on the Stanford paper's prompt architecture.
    """

    # ==========================================================================
    # AGENT SUMMARY
    # ==========================================================================

    AGENT_SUMMARY = """Name: {name} (age: {age})
Innate traits: {traits}
{summary}

{occupation}

{recent_activities}"""

    # ==========================================================================
    # IMPORTANCE SCORING
    # ==========================================================================

    IMPORTANCE_SCORING = """On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, major discovery, death of a loved one), rate the likely poignancy of the following piece of memory.

Memory: {memory}

Respond with only a single number between 1 and 10.
Rating:"""

    # ==========================================================================
    # REFLECTION
    # ==========================================================================

    REFLECTION_QUESTIONS = """Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?

Statements about {agent_name}:
{statements}

Questions (one per line):
1)"""

    REFLECTION_INSIGHTS = """Statements about {agent_name}:
{numbered_statements}

What 5 high-level insights can you infer from the above statements? Format each insight with the statement numbers that support it.

Example format:
1. Insight here (because of 1, 5, 3)

Insights:
1."""

    # ==========================================================================
    # PLANNING
    # ==========================================================================

    DAILY_PLAN = """{agent_summary}

On {yesterday_date}, {agent_name} {yesterday_summary}.

Today is {today_date}. Here is {agent_name}'s plan today in broad strokes:
1)"""

    HOURLY_DECOMPOSITION = """{agent_summary}

Today is {date}. {agent_name}'s plan for today:
{daily_plan}

The current time is {current_time}. What is {agent_name} doing in the next hour? Be specific about the activity and location.

{agent_name} is:"""

    ACTION_DECOMPOSITION = """{agent_summary}

{agent_name}'s hourly plan: {hourly_plan}

Break this down into 5-15 minute actions. What specific action is {agent_name} taking right now ({current_time})?

Action:"""

    # ==========================================================================
    # REACTION
    # ==========================================================================

    REACTION_DECISION = """{agent_summary}

It is {current_time}.
{agent_name}'s status: {current_status}

Observation: {observation}

Summary of relevant context from {agent_name}'s memory:
{memory_context}

Should {agent_name} react to the observation, and if so, what would be an appropriate reaction? Answer in the format:
REACT: [yes/no]
REACTION: [description of reaction if yes, or "continue current activity" if no]"""

    WHAT_WOULD_SAY = """{agent_summary}

It is {current_time}.
{agent_name}'s status: {current_status}

{agent_name} sees: {observation}

Summary of relevant context from {agent_name}'s memory:
{memory_context}

What would {agent_name} say? Respond with just the dialogue.

{agent_name}:"""

    # ==========================================================================
    # DIALOGUE
    # ==========================================================================

    DIALOGUE_INITIATION = """{agent_summary}

It is {current_time}.
{agent_name}'s status: {current_status}

{agent_name} sees {other_agent_name}.

Summary of {agent_name}'s memory about {other_agent_name}:
{memory_about_other}

Should {agent_name} start a conversation with {other_agent_name}? If yes, what would they say?

START_CONVERSATION: [yes/no]
OPENING:"""

    DIALOGUE_RESPONSE = """{agent_summary}

It is {current_time}.
{agent_name}'s status: {current_status}

{other_agent_name} says to {agent_name}: "{other_utterance}"

Summary of {agent_name}'s memory about {other_agent_name}:
{memory_about_other}

How would {agent_name} respond?

{agent_name}:"""

    DIALOGUE_CONTINUE = """{agent_summary}

It is {current_time}.

{agent_name} is in a conversation with {other_agent_name}.

Conversation so far:
{dialogue_history}

Summary of relevant context from {agent_name}'s memory:
{memory_context}

How would {agent_name} continue the conversation? (Respond with just the dialogue, or "[END]" if the conversation should end)

{agent_name}:"""

    # ==========================================================================
    # OBSERVATION GENERATION
    # ==========================================================================

    GENERATE_OBSERVATION = """What would {agent_name} observe at this location?

Location: {location_description}
Nearby people: {nearby_agents}
Time: {current_time}
Weather: {weather}

Describe what {agent_name} perceives in one sentence from their perspective.

Observation:"""

    # ==========================================================================
    # SKILL USAGE
    # ==========================================================================

    SKILL_ATTEMPT = """{agent_summary}

{agent_name} is attempting to: {action_description}

Using skill: {skill_name} (level {skill_level})
Relevant stats: {relevant_stats}

Describe the attempt and outcome. Consider the skill level and stats.

Attempt:"""

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    @classmethod
    def format_agent_summary(
        cls,
        name: str,
        age: int,
        traits: List[str],
        summary: str,
        occupation: str = "",
        recent_activities: str = ""
    ) -> str:
        """Format an agent summary"""
        return cls.AGENT_SUMMARY.format(
            name=name,
            age=age,
            traits=", ".join(traits),
            summary=summary,
            occupation=occupation,
            recent_activities=recent_activities
        )

    @classmethod
    def format_reflection_questions(
        cls,
        agent_name: str,
        statements: List[str]
    ) -> str:
        """Format reflection question prompt"""
        stmt_text = "\n".join(f"- {s}" for s in statements)
        return cls.REFLECTION_QUESTIONS.format(
            agent_name=agent_name,
            statements=stmt_text
        )

    @classmethod
    def format_reflection_insights(
        cls,
        agent_name: str,
        statements: List[str]
    ) -> str:
        """Format reflection insight prompt"""
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        return cls.REFLECTION_INSIGHTS.format(
            agent_name=agent_name,
            numbered_statements=numbered
        )

    @classmethod
    def format_daily_plan(
        cls,
        agent_summary: str,
        agent_name: str,
        yesterday_date: str,
        yesterday_summary: str,
        today_date: str
    ) -> str:
        """Format daily planning prompt"""
        return cls.DAILY_PLAN.format(
            agent_summary=agent_summary,
            agent_name=agent_name,
            yesterday_date=yesterday_date,
            yesterday_summary=yesterday_summary,
            today_date=today_date
        )

    @classmethod
    def format_reaction(
        cls,
        agent_summary: str,
        agent_name: str,
        current_time: str,
        current_status: str,
        observation: str,
        memory_context: str
    ) -> str:
        """Format reaction decision prompt"""
        return cls.REACTION_DECISION.format(
            agent_summary=agent_summary,
            agent_name=agent_name,
            current_time=current_time,
            current_status=current_status,
            observation=observation,
            memory_context=memory_context
        )

    @classmethod
    def format_dialogue_response(
        cls,
        agent_summary: str,
        agent_name: str,
        current_time: str,
        current_status: str,
        other_agent_name: str,
        other_utterance: str,
        memory_about_other: str
    ) -> str:
        """Format dialogue response prompt"""
        return cls.DIALOGUE_RESPONSE.format(
            agent_summary=agent_summary,
            agent_name=agent_name,
            current_time=current_time,
            current_status=current_status,
            other_agent_name=other_agent_name,
            other_utterance=other_utterance,
            memory_about_other=memory_about_other
        )

    @classmethod
    def format_time(cls, dt: datetime) -> str:
        """Format datetime for prompts"""
        return dt.strftime("%I:%M %p on %A, %B %d, %Y")

    @classmethod
    def format_time_short(cls, dt: datetime) -> str:
        """Format short time for prompts"""
        return dt.strftime("%I:%M %p")
