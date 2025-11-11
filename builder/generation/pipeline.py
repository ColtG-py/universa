"""
World Builder - Generation Pipeline
Main orchestrator for procedural world generation.
Executes all generation passes in sequence.
"""

import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import numpy as np

from config import (
    WorldGenerationParams,
    GENERATION_PASSES,
    PASS_WEIGHTS,
    CHUNK_SIZE,
)
from models.world import WorldState, WorldMetadata, WorldChunk


class GenerationPipeline:
    """
    Main pipeline for executing all world generation passes.
    Manages state, progress tracking, and pass orchestration.
    """
    
    def __init__(
        self,
        params: WorldGenerationParams,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize generation pipeline.
        
        Args:
            params: World generation parameters
            progress_callback: Optional callback for progress updates (pass_name, percent)
        """
        self.params = params
        self.progress_callback = progress_callback
        
        # Create metadata
        size_int = int(params.size) if isinstance(params.size, str) else params.size.to_int()
        self.metadata = WorldMetadata(
            seed=params.seed,
            size=size_int,
            generation_params=params.dict(),
        )
        
        # Create world state
        self.world_state = WorldState(self.metadata, params)
        
        # Pass registry - will be populated with pass modules
        self.pass_registry: Dict[str, Any] = {}
        
        # Track timing for each pass
        self.pass_timings: Dict[str, float] = {}
    
    def register_pass(self, pass_name: str, pass_module):
        """Register a generation pass module"""
        self.pass_registry[pass_name] = pass_module
    
    def generate(self) -> WorldState:
        """
        Execute complete world generation pipeline.
        
        Returns:
            Generated WorldState
        """
        self.metadata.status = "generating"
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting World Generation")
        print(f"Seed: {self.params.seed}")
        print(f"Size: {self.params.size} ({self.world_state.size}x{self.world_state.size})")
        print(f"Chunks: {self.world_state.num_chunks}x{self.world_state.num_chunks}")
        print(f"{'='*60}\n")
        
        try:
            # Calculate total weight for progress
            total_weight = sum(PASS_WEIGHTS.values())
            accumulated_weight = 0
            
            # Execute each pass in sequence
            for pass_name in GENERATION_PASSES:
                if pass_name not in self.pass_registry:
                    print(f"⚠ Warning: Pass '{pass_name}' not registered, skipping")
                    continue
                
                pass_start = time.time()
                self.metadata.current_pass = pass_name
                
                # Calculate progress percentage
                progress = (accumulated_weight / total_weight) * 100
                self.metadata.progress_percent = progress
                
                print(f"[{progress:5.1f}%] Executing {pass_name}...")
                
                # Execute the pass
                pass_module = self.pass_registry[pass_name]
                pass_module.execute(self.world_state, self.params)
                
                # Track timing
                pass_duration = time.time() - pass_start
                self.pass_timings[pass_name] = pass_duration
                
                print(f"        ✓ Completed in {pass_duration:.2f}s")
                
                # Update progress
                accumulated_weight += PASS_WEIGHTS[pass_name]
                
                # Call progress callback
                if self.progress_callback:
                    self.progress_callback(pass_name, (accumulated_weight / total_weight) * 100)
            
            # Mark as complete
            self.metadata.status = "ready"
            self.metadata.progress_percent = 100.0
            self.metadata.completed_at = datetime.utcnow()
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"World Generation Complete!")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*60}\n")
            
            # Print timing breakdown
            print("Pass Timings:")
            for pass_name, duration in self.pass_timings.items():
                percentage = (duration / total_time) * 100
                print(f"  {pass_name:30s} {duration:8.2f}s ({percentage:5.1f}%)")
            
            return self.world_state
            
        except Exception as e:
            self.metadata.status = "failed"
            self.metadata.error_message = str(e)
            print(f"\n❌ Generation failed: {e}")
            raise
    
    def generate_chunk(self, chunk_x: int, chunk_y: int) -> WorldChunk:
        """
        Generate a single chunk independently.
        Useful for on-demand chunk generation.
        
        Args:
            chunk_x: Chunk X coordinate
            chunk_y: Chunk Y coordinate
        
        Returns:
            Generated WorldChunk
        """
        print(f"Generating chunk ({chunk_x}, {chunk_y})...")
        
        chunk = self.world_state.get_or_create_chunk(chunk_x, chunk_y)
        
        # Execute each pass for this chunk
        for pass_name in GENERATION_PASSES:
            if pass_name not in self.pass_registry:
                continue
            
            pass_module = self.pass_registry[pass_name]
            
            # Check if pass supports chunk-level generation
            if hasattr(pass_module, 'execute_chunk'):
                pass_module.execute_chunk(chunk, self.world_state, self.params)
            else:
                # Pass operates on whole world, skip for chunk generation
                pass
        
        chunk.generated_at = datetime.utcnow()
        
        return chunk


class ChunkGenerator:
    """
    Helper for generating chunks in parallel.
    Can be used for distributed generation.
    """
    
    def __init__(self, params: WorldGenerationParams, world_state: WorldState):
        self.params = params
        self.world_state = world_state
    
    def generate_chunk_batch(self, chunk_coords: list) -> Dict[tuple, WorldChunk]:
        """
        Generate multiple chunks.
        
        Args:
            chunk_coords: List of (chunk_x, chunk_y) tuples
        
        Returns:
            Dictionary mapping coordinates to generated chunks
        """
        chunks = {}
        
        for chunk_x, chunk_y in chunk_coords:
            chunk = self.world_state.get_or_create_chunk(chunk_x, chunk_y)
            # Execute generation passes...
            chunks[(chunk_x, chunk_y)] = chunk
        
        return chunks


def create_pipeline(params: WorldGenerationParams) -> GenerationPipeline:
    """
    Factory function to create a fully configured generation pipeline.
    
    Args:
        params: World generation parameters
    
    Returns:
        Configured GenerationPipeline ready to execute
    """
    pipeline = GenerationPipeline(params)
    
    # Import and register all passes
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
    
    return pipeline
