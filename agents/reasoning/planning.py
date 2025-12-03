"""
Planning System
Hierarchical plan generation and management.
Based on Stanford Generative Agents paper.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from agents.memory.memory_stream import MemoryStream
from agents.models.memory import Plan
from agents.llm.ollama_client import OllamaClient
from agents.llm.prompts import PromptTemplates


class PlanGranularity(str, Enum):
    """Plan granularity levels"""
    DAY = "day"
    HOUR = "hour"
    ACTION = "action"


@dataclass
class DayPlan:
    """A day-level plan with broad strokes"""
    plan_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    date: datetime = None
    activities: List[str] = field(default_factory=list)
    is_complete: bool = False

    def to_string(self) -> str:
        """Convert to readable string"""
        return "\n".join(f"{i+1}) {a}" for i, a in enumerate(self.activities))


@dataclass
class HourPlan:
    """An hour-level plan with more detail"""
    plan_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    parent_plan_id: Optional[UUID] = None
    start_time: datetime = None
    end_time: datetime = None
    activity: str = ""
    location: Optional[str] = None
    is_complete: bool = False


@dataclass
class ActionPlan:
    """A 5-15 minute action plan with specific actions"""
    plan_id: UUID = field(default_factory=uuid4)
    agent_id: UUID = None
    parent_plan_id: Optional[UUID] = None
    start_time: datetime = None
    duration_minutes: int = 15
    action: str = ""
    is_complete: bool = False
    is_cancelled: bool = False


class PlanningSystem:
    """
    Generates and manages hierarchical plans.

    From the Stanford paper:
    "Plans describe a future sequence of actions for the agent...
    Plans are generated top-down: first creating a broad sketch,
    then recursively decomposing into finer-grained actions."

    Plan hierarchy:
    - Day plan: Broad strokes for the entire day
    - Hour plan: What to do in each hour
    - Action plan: Specific 5-15 minute actions
    """

    # Day planning prompt
    DAY_PLAN_PROMPT = """{agent_summary}

Today is {today_date}. {agent_name} woke up at {wake_time}.

Based on {agent_name}'s personality, occupation, and recent activities, what will {agent_name} do today?

List 5-8 activities in order, starting from waking up. Be specific about locations and times.

{agent_name}'s plan for today:
1)"""

    # Hour decomposition prompt
    HOUR_PLAN_PROMPT = """{agent_summary}

Today is {date}. {agent_name}'s plan for today:
{daily_plan}

It is now {current_time}. Based on the daily plan, what exactly is {agent_name} doing right now and for the next hour?

Be specific about the activity, location, and any people involved.

{agent_name} is currently:"""

    # Action decomposition prompt
    ACTION_PLAN_PROMPT = """{agent_summary}

{agent_name}'s current activity: {hourly_activity}

Break this down into specific actions. What is {agent_name} doing right now ({current_time}) for the next 10-15 minutes?

Current action:"""

    def __init__(
        self,
        agent_id: UUID,
        agent_name: str,
        memory_stream: MemoryStream,
        ollama_client: Optional[OllamaClient] = None,
    ):
        """
        Initialize planning system.

        Args:
            agent_id: Agent's UUID
            agent_name: Agent's name
            memory_stream: Agent's memory stream
            ollama_client: LLM client
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.memory_stream = memory_stream
        self.client = ollama_client

        # Current plans
        self._current_day_plan: Optional[DayPlan] = None
        self._current_hour_plan: Optional[HourPlan] = None
        self._current_action: Optional[ActionPlan] = None

        # Plan history
        self._day_plans: List[DayPlan] = []
        self._hour_plans: List[HourPlan] = []
        self._action_plans: List[ActionPlan] = []

    @property
    def current_day_plan(self) -> Optional[DayPlan]:
        """Get current day plan"""
        return self._current_day_plan

    @property
    def current_hour_plan(self) -> Optional[HourPlan]:
        """Get current hour plan"""
        return self._current_hour_plan

    @property
    def current_action(self) -> Optional[ActionPlan]:
        """Get current action"""
        return self._current_action

    async def create_day_plan(
        self,
        agent_summary: str,
        wake_time: str = "7:00 AM",
        date: Optional[datetime] = None
    ) -> DayPlan:
        """
        Create a day-level plan.

        Args:
            agent_summary: Description of agent
            wake_time: Time agent wakes up
            date: Date for the plan

        Returns:
            Generated day plan
        """
        plan_date = date or datetime.utcnow()

        if self.client:
            activities = await self._generate_day_activities(
                agent_summary, wake_time, plan_date
            )
        else:
            activities = self._default_day_activities()

        plan = DayPlan(
            agent_id=self.agent_id,
            date=plan_date,
            activities=activities
        )

        self._current_day_plan = plan
        self._day_plans.append(plan)

        # Store in memory stream
        plan_text = f"Today I plan to: {'; '.join(activities)}"
        await self.memory_stream.add_plan(
            description=plan_text,
            granularity="day",
            start_time=plan_date.replace(hour=0, minute=0),
            end_time=plan_date.replace(hour=23, minute=59),
            importance=0.6
        )

        return plan

    async def create_hour_plan(
        self,
        agent_summary: str,
        current_time: Optional[datetime] = None
    ) -> HourPlan:
        """
        Create an hour-level plan by decomposing day plan.

        Args:
            agent_summary: Description of agent
            current_time: Current time

        Returns:
            Generated hour plan
        """
        now = current_time or datetime.utcnow()

        if not self._current_day_plan:
            await self.create_day_plan(agent_summary)

        if self.client:
            activity, location = await self._generate_hour_activity(
                agent_summary, now
            )
        else:
            activity = self._get_default_hour_activity(now.hour)
            location = None

        plan = HourPlan(
            agent_id=self.agent_id,
            parent_plan_id=self._current_day_plan.plan_id if self._current_day_plan else None,
            start_time=now,
            end_time=now + timedelta(hours=1),
            activity=activity,
            location=location
        )

        self._current_hour_plan = plan
        self._hour_plans.append(plan)

        # Store in memory stream
        await self.memory_stream.add_plan(
            description=f"For the next hour: {activity}",
            granularity="hour",
            start_time=now,
            end_time=now + timedelta(hours=1),
            importance=0.5
        )

        return plan

    async def create_action_plan(
        self,
        agent_summary: str,
        current_time: Optional[datetime] = None,
        duration_minutes: int = 15
    ) -> ActionPlan:
        """
        Create a specific action plan.

        Args:
            agent_summary: Description of agent
            current_time: Current time
            duration_minutes: Duration of action

        Returns:
            Generated action plan
        """
        now = current_time or datetime.utcnow()

        if not self._current_hour_plan:
            await self.create_hour_plan(agent_summary, now)

        if self.client:
            action = await self._generate_action(agent_summary, now)
        else:
            action = self._current_hour_plan.activity if self._current_hour_plan else "resting"

        plan = ActionPlan(
            agent_id=self.agent_id,
            parent_plan_id=self._current_hour_plan.plan_id if self._current_hour_plan else None,
            start_time=now,
            duration_minutes=duration_minutes,
            action=action
        )

        self._current_action = plan
        self._action_plans.append(plan)

        # Store in memory stream
        await self.memory_stream.add_plan(
            description=f"Right now: {action}",
            granularity="action",
            start_time=now,
            end_time=now + timedelta(minutes=duration_minutes),
            importance=0.4
        )

        return plan

    async def complete_action(self, success: bool = True) -> None:
        """
        Mark current action as complete.

        Args:
            success: Whether action was successful
        """
        if self._current_action:
            self._current_action.is_complete = True

    async def cancel_action(self, reason: str = "") -> None:
        """
        Cancel current action.

        Args:
            reason: Reason for cancellation
        """
        if self._current_action:
            self._current_action.is_cancelled = True

    async def replan_from_event(
        self,
        event_description: str,
        agent_summary: str,
        current_time: Optional[datetime] = None
    ) -> ActionPlan:
        """
        Re-plan after an unexpected event.

        Args:
            event_description: What happened
            agent_summary: Agent description
            current_time: Current time

        Returns:
            New action plan
        """
        # Cancel current action if exists
        await self.cancel_action(f"Interrupted by: {event_description}")

        # Create new plan considering the event
        now = current_time or datetime.utcnow()

        if self.client:
            action = await self._generate_reaction_action(
                agent_summary, event_description, now
            )
        else:
            action = f"Responding to: {event_description}"

        plan = ActionPlan(
            agent_id=self.agent_id,
            start_time=now,
            duration_minutes=15,
            action=action
        )

        self._current_action = plan
        self._action_plans.append(plan)

        return plan

    def get_current_activity(self) -> str:
        """Get description of current activity"""
        if self._current_action:
            return self._current_action.action
        if self._current_hour_plan:
            return self._current_hour_plan.activity
        return "idle"

    def get_plan_context(self) -> Dict[str, Any]:
        """Get current planning context"""
        return {
            "day_plan": self._current_day_plan.to_string() if self._current_day_plan else None,
            "hour_activity": self._current_hour_plan.activity if self._current_hour_plan else None,
            "current_action": self._current_action.action if self._current_action else None,
        }

    # Private generation methods

    async def _generate_day_activities(
        self,
        agent_summary: str,
        wake_time: str,
        date: datetime
    ) -> List[str]:
        """Generate day activities using LLM"""
        prompt = self.DAY_PLAN_PROMPT.format(
            agent_summary=agent_summary,
            today_date=date.strftime("%A, %B %d"),
            agent_name=self.agent_name,
            wake_time=wake_time
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=400
            )
            return self._parse_activities(response.text)
        except Exception:
            return self._default_day_activities()

    async def _generate_hour_activity(
        self,
        agent_summary: str,
        current_time: datetime
    ) -> tuple:
        """Generate hour activity using LLM"""
        daily_plan = self._current_day_plan.to_string() if self._current_day_plan else "No specific plans"

        prompt = self.HOUR_PLAN_PROMPT.format(
            agent_summary=agent_summary,
            date=current_time.strftime("%A, %B %d"),
            agent_name=self.agent_name,
            daily_plan=daily_plan,
            current_time=current_time.strftime("%I:%M %p")
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=200
            )
            activity = response.text.strip()
            # Extract location if mentioned
            location = None
            if " at " in activity.lower():
                parts = activity.lower().split(" at ")
                if len(parts) > 1:
                    location = parts[-1].strip()
            return activity, location
        except Exception:
            return self._get_default_hour_activity(current_time.hour), None

    async def _generate_action(
        self,
        agent_summary: str,
        current_time: datetime
    ) -> str:
        """Generate specific action using LLM"""
        hourly_activity = self._current_hour_plan.activity if self._current_hour_plan else "general activities"

        prompt = self.ACTION_PLAN_PROMPT.format(
            agent_summary=agent_summary,
            agent_name=self.agent_name,
            hourly_activity=hourly_activity,
            current_time=current_time.strftime("%I:%M %p")
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=100
            )
            return response.text.strip()
        except Exception:
            return hourly_activity

    async def _generate_reaction_action(
        self,
        agent_summary: str,
        event: str,
        current_time: datetime
    ) -> str:
        """Generate action in response to event"""
        prompt = f"""{agent_summary}

{self.agent_name} was {self.get_current_activity()}.

Suddenly: {event}

What does {self.agent_name} do now?

{self.agent_name}:"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=100
            )
            return response.text.strip()
        except Exception:
            return f"Responding to {event}"

    def _parse_activities(self, response: str) -> List[str]:
        """Parse activities from LLM response"""
        activities = []
        lines = response.strip().split('\n')

        for line in lines:
            # Remove numbering
            import re
            line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if line and len(line) > 5:
                activities.append(line)

        return activities[:8] if activities else self._default_day_activities()

    def _default_day_activities(self) -> List[str]:
        """Default day activities as fallback"""
        return [
            "Wake up and have breakfast",
            "Morning work or activities",
            "Midday meal and rest",
            "Afternoon activities",
            "Evening meal",
            "Relaxation and social time",
            "Prepare for bed and sleep"
        ]

    def _get_default_hour_activity(self, hour: int) -> str:
        """Get default activity based on time of day"""
        if 6 <= hour < 8:
            return "morning routine and breakfast"
        elif 8 <= hour < 12:
            return "morning work"
        elif 12 <= hour < 14:
            return "lunch and rest"
        elif 14 <= hour < 18:
            return "afternoon work"
        elif 18 <= hour < 20:
            return "dinner and relaxation"
        elif 20 <= hour < 22:
            return "evening leisure"
        else:
            return "sleeping"
