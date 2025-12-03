"""
Time Management System
Simulation time, day/night cycles, and seasons.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TimeOfDay(str, Enum):
    """Periods of the day"""
    DAWN = "dawn"           # 5-7
    MORNING = "morning"     # 7-12
    NOON = "noon"           # 12-14
    AFTERNOON = "afternoon" # 14-18
    EVENING = "evening"     # 18-21
    NIGHT = "night"         # 21-5


class Season(str, Enum):
    """Seasons of the year"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


@dataclass
class SimulationTime:
    """
    Represents time in the simulation.

    Simulation time can run faster or slower than real time.
    """
    year: int = 1
    month: int = 1
    day: int = 1
    hour: int = 8
    minute: int = 0

    # Days per month (simplified calendar)
    DAYS_PER_MONTH = 30
    MONTHS_PER_YEAR = 12

    def advance(self, minutes: int) -> None:
        """Advance time by minutes"""
        self.minute += minutes

        # Handle overflow
        while self.minute >= 60:
            self.minute -= 60
            self.hour += 1

        while self.hour >= 24:
            self.hour -= 24
            self.day += 1

        while self.day > self.DAYS_PER_MONTH:
            self.day -= self.DAYS_PER_MONTH
            self.month += 1

        while self.month > self.MONTHS_PER_YEAR:
            self.month -= self.MONTHS_PER_YEAR
            self.year += 1

    def get_time_of_day(self) -> TimeOfDay:
        """Get current period of day"""
        if 5 <= self.hour < 7:
            return TimeOfDay.DAWN
        elif 7 <= self.hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= self.hour < 14:
            return TimeOfDay.NOON
        elif 14 <= self.hour < 18:
            return TimeOfDay.AFTERNOON
        elif 18 <= self.hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def get_season(self) -> Season:
        """Get current season"""
        if self.month in [3, 4, 5]:
            return Season.SPRING
        elif self.month in [6, 7, 8]:
            return Season.SUMMER
        elif self.month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER

    def is_daytime(self) -> bool:
        """Check if it's daytime (6am-8pm)"""
        return 6 <= self.hour < 20

    def is_nighttime(self) -> bool:
        """Check if it's nighttime"""
        return not self.is_daytime()

    def to_string(self) -> str:
        """Convert to readable string"""
        return f"Year {self.year}, Month {self.month}, Day {self.day}, {self.hour:02d}:{self.minute:02d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute,
            "time_of_day": self.get_time_of_day().value,
            "season": self.get_season().value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationTime":
        """Create from dictionary"""
        return cls(
            year=data.get("year", 1),
            month=data.get("month", 1),
            day=data.get("day", 1),
            hour=data.get("hour", 8),
            minute=data.get("minute", 0),
        )

    def copy(self) -> "SimulationTime":
        """Create a copy"""
        return SimulationTime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
        )


class TimeManager:
    """
    Manages simulation time.

    Features:
    - Time advancement with configurable speed
    - Day/night cycles
    - Seasonal effects
    - Time-based events
    """

    def __init__(
        self,
        start_time: Optional[SimulationTime] = None,
        tick_minutes: int = 15,
        time_multiplier: float = 1.0,
    ):
        """
        Initialize time manager.

        Args:
            start_time: Starting simulation time
            tick_minutes: Minutes per simulation tick
            time_multiplier: Speed multiplier (2.0 = twice as fast)
        """
        self.current_time = start_time or SimulationTime()
        self.tick_minutes = tick_minutes
        self.time_multiplier = time_multiplier

        # Tracking
        self.total_ticks = 0
        self.start_real_time = datetime.utcnow()

        # Time-based callbacks
        self._dawn_callbacks: List[Callable] = []
        self._dusk_callbacks: List[Callable] = []
        self._hourly_callbacks: List[Callable] = []
        self._daily_callbacks: List[Callable] = []
        self._season_callbacks: List[Callable] = []

        # Previous state for detecting transitions
        self._last_hour = self.current_time.hour
        self._last_day = self.current_time.day
        self._last_season = self.current_time.get_season()

    def tick(self) -> Dict[str, Any]:
        """
        Advance time by one tick.

        Returns:
            Dict with time info and any triggered events
        """
        previous_time = self.current_time.copy()

        # Advance time
        minutes_to_advance = int(self.tick_minutes * self.time_multiplier)
        self.current_time.advance(minutes_to_advance)
        self.total_ticks += 1

        # Check for transitions
        events = []

        # Hour change
        if self.current_time.hour != self._last_hour:
            events.append("hour_change")
            for callback in self._hourly_callbacks:
                callback(self.current_time)

            # Dawn (5am)
            if self.current_time.hour == 5:
                events.append("dawn")
                for callback in self._dawn_callbacks:
                    callback(self.current_time)

            # Dusk (20pm/8pm)
            if self.current_time.hour == 20:
                events.append("dusk")
                for callback in self._dusk_callbacks:
                    callback(self.current_time)

        # Day change
        if self.current_time.day != self._last_day:
            events.append("day_change")
            for callback in self._daily_callbacks:
                callback(self.current_time)

        # Season change
        current_season = self.current_time.get_season()
        if current_season != self._last_season:
            events.append(f"season_change:{current_season.value}")
            for callback in self._season_callbacks:
                callback(self.current_time, current_season)

        # Update tracking
        self._last_hour = self.current_time.hour
        self._last_day = self.current_time.day
        self._last_season = current_season

        return {
            "previous": previous_time.to_dict(),
            "current": self.current_time.to_dict(),
            "tick": self.total_ticks,
            "events": events,
        }

    def on_dawn(self, callback: Callable) -> None:
        """Register callback for dawn"""
        self._dawn_callbacks.append(callback)

    def on_dusk(self, callback: Callable) -> None:
        """Register callback for dusk"""
        self._dusk_callbacks.append(callback)

    def on_hour(self, callback: Callable) -> None:
        """Register callback for each hour"""
        self._hourly_callbacks.append(callback)

    def on_day(self, callback: Callable) -> None:
        """Register callback for each day"""
        self._daily_callbacks.append(callback)

    def on_season(self, callback: Callable) -> None:
        """Register callback for season changes"""
        self._season_callbacks.append(callback)

    def get_environment_modifiers(self) -> Dict[str, float]:
        """
        Get environmental modifiers based on time.

        Returns:
            Dict of modifier_name -> value
        """
        time_of_day = self.current_time.get_time_of_day()
        season = self.current_time.get_season()

        modifiers = {
            "visibility": 1.0,
            "temperature": 1.0,
            "activity_level": 1.0,
            "danger_level": 1.0,
        }

        # Time of day effects
        if time_of_day == TimeOfDay.NIGHT:
            modifiers["visibility"] = 0.3
            modifiers["danger_level"] = 1.5
            modifiers["activity_level"] = 0.2
        elif time_of_day == TimeOfDay.DAWN or time_of_day == TimeOfDay.EVENING:
            modifiers["visibility"] = 0.7
            modifiers["activity_level"] = 0.6
        elif time_of_day == TimeOfDay.NOON:
            modifiers["temperature"] = 1.2
            modifiers["activity_level"] = 0.8  # Too hot

        # Season effects
        if season == Season.WINTER:
            modifiers["temperature"] = 0.5
            modifiers["activity_level"] *= 0.8
        elif season == Season.SUMMER:
            modifiers["temperature"] = 1.4
        elif season == Season.SPRING:
            modifiers["activity_level"] *= 1.1

        return modifiers

    def get_appropriate_activities(self) -> List[str]:
        """Get activities appropriate for current time"""
        time_of_day = self.current_time.get_time_of_day()

        if time_of_day == TimeOfDay.NIGHT:
            return ["sleep", "guard", "study", "rest"]
        elif time_of_day == TimeOfDay.DAWN:
            return ["wake", "breakfast", "prepare"]
        elif time_of_day == TimeOfDay.MORNING:
            return ["work", "train", "travel", "gather"]
        elif time_of_day == TimeOfDay.NOON:
            return ["lunch", "rest", "socialize"]
        elif time_of_day == TimeOfDay.AFTERNOON:
            return ["work", "train", "craft", "trade"]
        elif time_of_day == TimeOfDay.EVENING:
            return ["dinner", "socialize", "entertainment", "prepare_for_bed"]
        else:
            return ["rest"]

    def set_speed(self, multiplier: float) -> None:
        """Set time speed multiplier"""
        self.time_multiplier = max(0.1, min(100.0, multiplier))

    def pause(self) -> None:
        """Pause time (set multiplier to 0)"""
        self.time_multiplier = 0.0

    def resume(self, multiplier: float = 1.0) -> None:
        """Resume time"""
        self.time_multiplier = multiplier

    def get_elapsed_real_time(self) -> timedelta:
        """Get real time elapsed since simulation start"""
        return datetime.utcnow() - self.start_real_time

    def get_elapsed_simulation_time(self) -> Dict[str, int]:
        """Get simulation time elapsed"""
        total_minutes = self.total_ticks * self.tick_minutes
        hours = total_minutes // 60
        days = hours // 24

        return {
            "ticks": self.total_ticks,
            "minutes": total_minutes,
            "hours": hours,
            "days": days,
        }
