"""
Base classes for generation passes.
Provides common interface and utilities for all generation passes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from datetime import datetime

from models.world import WorldState


@dataclass
class PassConfig:
    """Configuration for a generation pass."""
    
    name: str
    """Name of the pass (e.g., 'Planetary Foundation')"""
    
    pass_number: int
    """Pass number (1-14)"""
    
    description: str
    """Brief description of what this pass does"""
    
    dependencies: list[int]
    """List of pass numbers that must complete before this pass"""
    
    weight: float = 1.0
    """Relative computational weight (for progress reporting)"""
    
    enabled: bool = True
    """Whether this pass is enabled"""
    
    validation_fn: Optional[Callable[[WorldState], bool]] = None
    """Optional validation function to run after pass completes"""


class BasePass(ABC):
    """
    Base class for all generation passes.
    
    Each pass should:
    1. Inherit from this class
    2. Implement the execute() method
    3. Define a config property
    4. Optionally implement validate()
    """
    
    def __init__(self):
        """Initialize the base pass."""
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._error: Optional[Exception] = None
    
    @property
    @abstractmethod
    def config(self) -> PassConfig:
        """Return the configuration for this pass."""
        pass
    
    @abstractmethod
    def execute(self, world_state: WorldState, params: Any) -> None:
        """
        Execute the generation pass.
        
        Args:
            world_state: The world state to modify
            params: Generation parameters (typically WorldGenerationParams)
            
        Raises:
            Exception: If pass execution fails
        """
        pass
    
    def validate(self, world_state: WorldState) -> bool:
        """
        Validate the world state after pass execution.
        
        Args:
            world_state: The world state to validate
            
        Returns:
            True if validation passes, False otherwise
            
        Note:
            Default implementation always returns True.
            Override to add custom validation logic.
        """
        if self.config.validation_fn:
            return self.config.validation_fn(world_state)
        return True
    
    def run(self, world_state: WorldState, params: Any) -> Dict[str, Any]:
        """
        Run the pass with timing and error handling.
        
        Args:
            world_state: The world state to modify
            params: Generation parameters
            
        Returns:
            Dictionary with execution metadata:
                - success: bool
                - duration: float (seconds)
                - error: Optional[str]
                
        Raises:
            Exception: Re-raises any exception from execute()
        """
        import time
        
        self._start_time = time.time()
        self._error = None
        
        try:
            # Execute the pass
            self.execute(world_state, params)
            
            # Validate if enabled
            if self.config.enabled:
                is_valid = self.validate(world_state)
                if not is_valid:
                    raise ValueError(f"Validation failed for pass {self.config.name}")
            
            self._end_time = time.time()
            
            return {
                "success": True,
                "duration": self._end_time - self._start_time,
                "error": None
            }
            
        except Exception as e:
            self._end_time = time.time()
            self._error = e
            
            return {
                "success": False,
                "duration": self._end_time - self._start_time,
                "error": str(e)
            }
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the last execution in seconds."""
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return None
    
    @property
    def last_error(self) -> Optional[Exception]:
        """Get the last error that occurred during execution."""
        return self._error
    
    def __str__(self) -> str:
        """String representation of the pass."""
        return f"Pass {self.config.pass_number}: {self.config.name}"
    
    def __repr__(self) -> str:
        """Detailed representation of the pass."""
        return f"<{self.__class__.__name__} {self.config.pass_number}: {self.config.name}>"


class PassRegistry:
    """
    Registry for generation passes.
    Manages pass registration, ordering, and execution.
    """
    
    def __init__(self):
        """Initialize the pass registry."""
        self._passes: Dict[int, BasePass] = {}
        self._execution_order: list[int] = []
    
    def register(self, pass_instance: BasePass) -> None:
        """
        Register a pass.
        
        Args:
            pass_instance: Instance of a pass to register
            
        Raises:
            ValueError: If pass number is already registered
        """
        pass_num = pass_instance.config.pass_number
        
        if pass_num in self._passes:
            raise ValueError(f"Pass {pass_num} already registered")
        
        self._passes[pass_num] = pass_instance
        self._update_execution_order()
    
    def unregister(self, pass_number: int) -> None:
        """
        Unregister a pass.
        
        Args:
            pass_number: Pass number to unregister
        """
        if pass_number in self._passes:
            del self._passes[pass_number]
            self._update_execution_order()
    
    def get_pass(self, pass_number: int) -> Optional[BasePass]:
        """
        Get a pass by number.
        
        Args:
            pass_number: Pass number to retrieve
            
        Returns:
            Pass instance or None if not found
        """
        return self._passes.get(pass_number)
    
    def get_all_passes(self) -> list[BasePass]:
        """
        Get all registered passes in execution order.
        
        Returns:
            List of pass instances in execution order
        """
        return [self._passes[num] for num in self._execution_order if num in self._passes]
    
    def _update_execution_order(self) -> None:
        """
        Update the execution order based on dependencies.
        Uses topological sort to resolve dependencies.
        """
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(pass_num: int):
            if pass_num in visited:
                return
            visited.add(pass_num)
            
            pass_instance = self._passes.get(pass_num)
            if pass_instance:
                # Visit dependencies first
                for dep in pass_instance.config.dependencies:
                    if dep in self._passes:
                        visit(dep)
                order.append(pass_num)
        
        # Visit all passes
        for pass_num in sorted(self._passes.keys()):
            visit(pass_num)
        
        self._execution_order = order
    
    @property
    def execution_order(self) -> list[int]:
        """Get the execution order of passes."""
        return self._execution_order.copy()
    
    def __len__(self) -> int:
        """Get the number of registered passes."""
        return len(self._passes)
    
    def __contains__(self, pass_number: int) -> bool:
        """Check if a pass is registered."""
        return pass_number in self._passes


# Global registry instance
_global_registry = PassRegistry()


def get_registry() -> PassRegistry:
    """Get the global pass registry."""
    return _global_registry


def register_pass(pass_instance: BasePass) -> None:
    """
    Register a pass in the global registry.
    
    Args:
        pass_instance: Pass instance to register
    """
    _global_registry.register(pass_instance)


def get_pass(pass_number: int) -> Optional[BasePass]:
    """
    Get a pass from the global registry.
    
    Args:
        pass_number: Pass number to retrieve
        
    Returns:
        Pass instance or None if not found
    """
    return _global_registry.get_pass(pass_number)