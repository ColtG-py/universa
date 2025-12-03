"""
World Generation Runner
Handles async world generation with progress tracking.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from uuid import uuid4
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound generation work
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="world_gen")

# Global storage for generation state (in production, use Redis)
_generation_state: Dict[str, Dict[str, Any]] = {}
_generation_lock = threading.Lock()


class GenerationRunner:
    """
    Manages world generation in background threads with progress tracking.
    """

    def __init__(self):
        self._active_generations: Dict[str, asyncio.Task] = {}

    async def start_generation(
        self,
        world_id: str,
        params: Dict[str, Any],
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> str:
        """
        Start world generation in a background thread.

        Returns immediately with the world_id for progress tracking.
        """
        # Initialize state
        with _generation_lock:
            _generation_state[world_id] = {
                "status": "initializing",
                "progress_percent": 0.0,
                "current_pass": None,
                "current_pass_number": 0,
                "total_passes": 18,
                "started_at": datetime.utcnow(),
                "completed_at": None,
                "error": None,
                "world_state": None
            }

        # Run generation in thread pool
        loop = asyncio.get_event_loop()

        def run_generation():
            try:
                return self._run_generation_sync(world_id, params)
            except Exception as e:
                logger.error(f"Generation failed for {world_id}: {e}")
                with _generation_lock:
                    _generation_state[world_id]["status"] = "failed"
                    _generation_state[world_id]["error"] = str(e)
                if on_error:
                    on_error(world_id, e)
                raise

        future = loop.run_in_executor(_executor, run_generation)

        # Handle completion
        async def await_completion():
            try:
                result = await future
                if on_complete:
                    on_complete(world_id, result)
                return result
            except Exception as e:
                logger.error(f"Generation task failed: {e}")

        task = asyncio.create_task(await_completion())
        self._active_generations[world_id] = task

        return world_id

    def _run_generation_sync(self, world_id: str, params: Dict[str, Any]):
        """
        Synchronous generation runner (runs in thread pool).
        """
        try:
            # Import builder modules
            import sys
            sys.path.insert(0, '/home/colt/Projects/universa/builder')

            from config import WorldGenerationParams, WorldSize

            # Convert API params to builder params
            size_map = {
                "SMALL": WorldSize.SMALL,
                "MEDIUM": WorldSize.MEDIUM,
                "LARGE": WorldSize.LARGE,
                "HUGE": WorldSize.HUGE
            }

            builder_params = WorldGenerationParams(
                seed=params.get("seed", int(datetime.utcnow().timestamp())),
                size=size_map.get(params.get("size", "MEDIUM"), WorldSize.MEDIUM),
                planet_radius_km=params.get("planet_radius_km", 6371.0),
                axial_tilt=params.get("axial_tilt", 23.5),
                ocean_percentage=params.get("ocean_percentage", 0.7),
                num_plates=params.get("num_plates", 12),
                enable_caves=params.get("enable_caves", True),
            )

            # Progress callback
            def on_progress(pass_name: str, percent: float):
                with _generation_lock:
                    state = _generation_state.get(world_id)
                    if state:
                        state["status"] = "generating"
                        state["current_pass"] = pass_name
                        state["progress_percent"] = percent
                        # Extract pass number from name like "pass_01_planetary"
                        try:
                            pass_num = int(pass_name.split("_")[1])
                            state["current_pass_number"] = pass_num
                        except (IndexError, ValueError):
                            pass
                logger.debug(f"World {world_id}: {pass_name} - {percent:.1f}%")

            # Create and run pipeline
            logger.info(f"Starting generation for world {world_id}")

            # Import GenerationPipeline directly to use progress callback
            from generation.pipeline import GenerationPipeline

            # Create pipeline with progress callback
            pipeline = GenerationPipeline(builder_params, progress_callback=on_progress)

            # Register all passes manually (same as create_pipeline does)
            from generation import pass_01_planetary
            from generation import pass_02_tectonics
            from generation import pass_03_topography
            from generation import pass_04_geology
            from generation import pass_05_atmosphere
            from generation import pass_06_oceans
            from generation import pass_07_climate
            from generation import pass_08_erosion
            from generation import pass_09_groundwater
            from generation import pass_10_rivers
            from generation import pass_11_soil
            from generation import pass_12_biomes
            from generation import pass_13_fauna
            from generation import pass_14_resources
            from generation import pass_15_magic
            from generation import pass_16_settlements
            from generation import pass_17_roads
            from generation import pass_18_politics

            pipeline.register_pass("pass_01_planetary", pass_01_planetary)
            pipeline.register_pass("pass_02_tectonics", pass_02_tectonics)
            pipeline.register_pass("pass_03_topography", pass_03_topography)
            pipeline.register_pass("pass_04_geology", pass_04_geology)
            pipeline.register_pass("pass_05_atmosphere", pass_05_atmosphere)
            pipeline.register_pass("pass_06_oceans", pass_06_oceans)
            pipeline.register_pass("pass_07_climate", pass_07_climate)
            pipeline.register_pass("pass_08_erosion", pass_08_erosion)
            pipeline.register_pass("pass_09_groundwater", pass_09_groundwater)
            pipeline.register_pass("pass_10_rivers", pass_10_rivers)
            pipeline.register_pass("pass_11_soil", pass_11_soil)
            pipeline.register_pass("pass_12_biomes", pass_12_biomes)
            pipeline.register_pass("pass_13_fauna", pass_13_fauna)
            pipeline.register_pass("pass_14_resources", pass_14_resources)
            pipeline.register_pass("pass_15_magic", pass_15_magic)
            pipeline.register_pass("pass_16_settlements", pass_16_settlements)
            pipeline.register_pass("pass_17_roads", pass_17_roads)
            pipeline.register_pass("pass_18_politics", pass_18_politics)

            world_state = pipeline.generate()

            # Update completion state
            with _generation_lock:
                state = _generation_state.get(world_id)
                if state:
                    state["status"] = "ready"
                    state["progress_percent"] = 100.0
                    state["completed_at"] = datetime.utcnow()
                    state["world_state"] = world_state
                    state["current_pass"] = "Complete"
                    state["current_pass_number"] = 18

            logger.info(f"Generation complete for world {world_id}")
            return world_state

        except ImportError as e:
            logger.error(f"Failed to import builder modules: {e}")
            raise RuntimeError(f"Builder modules not available: {e}")

    def get_progress(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get current generation progress."""
        with _generation_lock:
            state = _generation_state.get(world_id)
            if not state:
                return None

            # Return copy without world_state (too large)
            return {
                "world_id": world_id,
                "status": state["status"],
                "progress_percent": state["progress_percent"],
                "current_pass": state["current_pass"],
                "current_pass_number": state["current_pass_number"],
                "total_passes": state["total_passes"],
                "started_at": state["started_at"],
                "completed_at": state["completed_at"],
                "error": state["error"]
            }

    def get_world_state(self, world_id: str):
        """Get the completed world state."""
        with _generation_lock:
            state = _generation_state.get(world_id)
            if state and state["status"] == "ready":
                return state.get("world_state")
        return None

    def is_complete(self, world_id: str) -> bool:
        """Check if generation is complete."""
        with _generation_lock:
            state = _generation_state.get(world_id)
            return state is not None and state["status"] == "ready"

    def cancel_generation(self, world_id: str) -> bool:
        """Cancel an in-progress generation."""
        task = self._active_generations.get(world_id)
        if task and not task.done():
            task.cancel()
            with _generation_lock:
                state = _generation_state.get(world_id)
                if state:
                    state["status"] = "cancelled"
            return True
        return False

    def cleanup(self, world_id: str):
        """Clean up generation state."""
        with _generation_lock:
            _generation_state.pop(world_id, None)
        self._active_generations.pop(world_id, None)


# Global instance
generation_runner = GenerationRunner()
