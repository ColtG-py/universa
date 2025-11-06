"""
World Builder - Layer Visualization for Simulations
Provides real-time visualization and animation for erosion simulation debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List, Tuple
import time


class SimulationVisualizer:
    """
    Real-time visualization for erosion simulation debugging.
    Shows elevation changes, discharge accumulation, and river formation.
    """
    
    def __init__(
        self,
        elevation_initial: np.ndarray,
        world_size: int,
        update_interval: int = 100,
        save_frames: bool = False,
        output_dir: str = "debug_visualization"
    ):
        """
        Initialize simulation visualizer.
        
        Args:
            elevation_initial: Initial elevation map
            world_size: Size of world grid
            update_interval: Update display every N iterations
            save_frames: Save frames to disk for movie creation
            output_dir: Directory for saved frames
        """
        self.elevation_initial = elevation_initial.copy()
        self.world_size = world_size
        self.update_interval = update_interval
        self.save_frames = save_frames
        self.output_dir = Path(output_dir)
        
        if self.save_frames:
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "frames").mkdir(exist_ok=True)
        
        # Visualization state
        self.frames_captured = []
        self.frame_count = 0
        self.last_update_time = time.time()
        self.figure_initialized = False
        
        # Stored data for updates
        self.current_elevation = None
        self.current_discharge = None
        
        print(f"  - Debug visualization initialized")
        print(f"    Update interval: {update_interval} iterations")
        if save_frames:
            print(f"    Saving frames to: {self.output_dir / 'frames'}")
        
        # Don't initialize figure yet - wait for first update
        self.fig = None
        self.axes = None
    
    def _setup_figure(self):
        """Setup matplotlib figure and subplots with safe backend"""
        try:
            # Use Agg backend for non-interactive or TkAgg for interactive
            import matplotlib
            current_backend = matplotlib.get_backend()
            
            # If we're in a non-interactive environment, use Agg
            if current_backend == 'agg':
                matplotlib.use('Agg')
            else:
                # Try to use a safe interactive backend
                try:
                    matplotlib.use('TkAgg')
                except:
                    matplotlib.use('Agg')
                    print(f"    Warning: Using non-interactive backend (Agg)")
            
            plt.ion()  # Interactive mode
            
            self.fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Create subplots
            self.ax_elevation = self.fig.add_subplot(gs[0, 0])
            self.ax_erosion = self.fig.add_subplot(gs[0, 1])
            self.ax_discharge = self.fig.add_subplot(gs[0, 2])
            self.ax_rivers = self.fig.add_subplot(gs[1, 0])
            self.ax_momentum = self.fig.add_subplot(gs[1, 1])
            self.ax_stats = self.fig.add_subplot(gs[1, 2])
            
            # Set titles
            self.ax_elevation.set_title('Current Elevation', fontweight='bold')
            self.ax_erosion.set_title('Erosion/Deposition', fontweight='bold')
            self.ax_discharge.set_title('Discharge Accumulation', fontweight='bold')
            self.ax_rivers.set_title('River Network', fontweight='bold')
            self.ax_momentum.set_title('Flow Momentum', fontweight='bold')
            self.ax_stats.set_title('Statistics', fontweight='bold')
            
            # Turn off axes
            for ax in [self.ax_elevation, self.ax_erosion, self.ax_discharge, 
                       self.ax_rivers, self.ax_momentum]:
                ax.axis('off')
            
            # Setup stats axis
            self.ax_stats.axis('off')
            self.stats_text = self.ax_stats.text(
                0.1, 0.5, '', 
                transform=self.ax_stats.transAxes,
                verticalalignment='center',
                fontfamily='monospace',
                fontsize=10
            )
            
            # Initialize image objects (will be updated)
            self.im_elevation = None
            self.im_erosion = None
            self.im_discharge = None
            self.im_rivers = None
            self.im_momentum = None
            
            plt.tight_layout()
            
            # Force initial draw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.figure_initialized = True
            print(f"    Matplotlib figure initialized successfully")
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not initialize matplotlib figure: {e}")
            print(f"    Debug visualization will be disabled")
            self.figure_initialized = False
            self.fig = None
    
    def update(self, iteration: int, total_iterations: int, stats: dict):
        """
        Update visualization with current simulation state.
        
        Called by the simulation at regular intervals.
        
        Args:
            iteration: Current iteration number
            total_iterations: Total number of iterations
            stats: Dictionary of simulation statistics
        """
        # Only update at specified interval
        if iteration % self.update_interval != 0:
            return
        
        # Initialize figure on first update (lazy initialization)
        if not self.figure_initialized:
            print(f"    [DEBUG VIZ] Initializing matplotlib figure...")
            self._setup_figure()
            if not self.figure_initialized:
                # Failed to initialize, disable visualization
                return
        
        try:
            # Get current time for FPS calculation
            current_time = time.time()
            dt = current_time - self.last_update_time
            fps = self.update_interval / dt if dt > 0 else 0
            self.last_update_time = current_time
            
            print(f"    [DEBUG VIZ] Updating display at iteration {iteration}/{total_iterations} ({fps:.1f} updates/s)")
            
            # Note: We can't access simulation data directly from the callback
            # The visualization will be updated in finalize() with actual data
            # For now, just update the stats text
            
            if self.stats_text:
                stats_str = f"""
Simulation Progress
{'='*30}

Iteration: {iteration}/{total_iterations}
Progress: {iteration/total_iterations*100:.1f}%

FPS: {fps:.1f} updates/s
Total Drops: {stats.get('total_drops', 0):,}

Max Discharge: {stats.get('max_discharge', 0):.6f}
"""
                self.stats_text.set_text(stats_str)
            
            # Update the display
            if self.fig:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # Allow matplotlib to process events
            
            self.frame_count += 1
            
            # Save frame if requested
            if self.save_frames and self.fig:
                frame_path = self.output_dir / "frames" / f"frame_{self.frame_count:06d}.png"
                self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')
                self.frames_captured.append(frame_path)
        
        except Exception as e:
            print(f"    [DEBUG VIZ] Warning: Update failed: {e}")
            # Don't crash the simulation if visualization fails
    
    def finalize(
        self,
        elevation_final: np.ndarray,
        discharge: np.ndarray,
        river_presence: np.ndarray
    ):
        """
        Finalize visualization and create summary plots.
        
        Args:
            elevation_final: Final eroded elevation
            discharge: Final discharge field
            river_presence: Boolean array of river locations
        """
        print(f"  - Finalizing debug visualization...")
        
        try:
            # Calculate erosion/deposition
            elevation_change = elevation_final - self.elevation_initial
            
            # Create final summary figure (always create this, independent of live viz)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Initial elevation
            im1 = axes[0, 0].imshow(self.elevation_initial, cmap='terrain', interpolation='bilinear')
            axes[0, 0].set_title('Initial Elevation', fontweight='bold')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            # 2. Final elevation
            im2 = axes[0, 1].imshow(elevation_final, cmap='terrain', interpolation='bilinear')
            axes[0, 1].set_title('Final Elevation (Eroded)', fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # 3. Erosion/deposition
            erosion_max = max(abs(elevation_change.min()), abs(elevation_change.max()))
            if erosion_max > 0:
                im3 = axes[0, 2].imshow(
                    elevation_change, 
                    cmap='RdBu_r', 
                    interpolation='bilinear',
                    vmin=-erosion_max,
                    vmax=erosion_max
                )
                axes[0, 2].set_title('Erosion (blue) / Deposition (red)', fontweight='bold')
                axes[0, 2].axis('off')
                plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # 4. Discharge (log scale)
            land_mask = elevation_final > 0
            discharge_masked = np.ma.masked_where(~land_mask, discharge)
            discharge_log = np.log10(discharge_masked + 1e-6)
            
            im4 = axes[1, 0].imshow(discharge_log, cmap='Blues', interpolation='bilinear')
            axes[1, 0].set_title('Discharge (log scale)', fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # 5. River network on terrain
            axes[1, 1].imshow(elevation_final, cmap='terrain', interpolation='bilinear', alpha=0.5)
            river_display = np.zeros((*river_presence.shape, 4))
            river_display[river_presence] = [0, 0.9, 1.0, 1.0]  # Cyan
            axes[1, 1].imshow(river_display, interpolation='nearest')
            axes[1, 1].set_title('River Network', fontweight='bold')
            axes[1, 1].axis('off')
            
            # 6. Statistics
            axes[1, 2].axis('off')
            
            # Calculate stats
            total_erosion = -elevation_change[elevation_change < 0].sum() if np.any(elevation_change < 0) else 0
            total_deposition = elevation_change[elevation_change > 0].sum() if np.any(elevation_change > 0) else 0
            river_cells = river_presence.sum()
            land_cells = land_mask.sum()
            
            stats_text = f"""
EROSION SIMULATION SUMMARY
{'=' * 40}

Terrain Changes:
  Total Erosion:     {total_erosion:,.1f} m³
  Total Deposition:  {total_deposition:,.1f} m³
  Max Erosion:       {-elevation_change.min():.3f} m
  Max Deposition:    {elevation_change.max():.3f} m

River Network:
  River Cells:       {river_cells:,}
  Land Coverage:     {river_cells / land_cells * 100 if land_cells > 0 else 0:.2f}%
  Max Discharge:     {discharge.max():.6f}

Discharge Statistics:
  Mean:              {discharge[land_mask].mean():.6f}
  Median:            {np.median(discharge[land_mask]):.6f}
  Std Dev:           {discharge[land_mask].std():.6f}

Frames Captured:     {self.frame_count}
"""
            
            axes[1, 2].text(
                0.1, 0.5, stats_text,
                transform=axes[1, 2].transAxes,
                verticalalignment='center',
                fontfamily='monospace',
                fontsize=9
            )
            
            plt.suptitle('Hydraulic Erosion Simulation Results', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save summary
            summary_path = self.output_dir / "erosion_summary.png"
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"    Saved summary: {summary_path}")
            
            # Create animation if frames were saved
            if self.save_frames and len(self.frames_captured) > 0:
                self._create_animation()
            
            # Close live visualization figure if it exists
            if self.fig:
                plt.close(self.fig)
            
            # Show summary briefly then close
            plt.show(block=False)
            plt.pause(2)  # Show for 2 seconds
            plt.close('all')
            
        except Exception as e:
            print(f"    ⚠ Warning: Could not finalize visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_animation(self):
        """Create MP4 animation from saved frames"""
        print(f"    Creating animation from {len(self.frames_captured)} frames...")
        
        try:
            import imageio
            
            animation_path = self.output_dir / "erosion_animation.mp4"
            
            with imageio.get_writer(animation_path, fps=10) as writer:
                for frame_path in self.frames_captured:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
            
            print(f"    Animation saved: {animation_path}")
            
        except ImportError:
            print(f"    ⚠ imageio not available - cannot create animation")
            print(f"      Frames saved in: {self.output_dir / 'frames'}")


class LayerAnimator:
    """
    Standalone animation tool for visualizing simulation data frame-by-frame.
    """
    
    def __init__(self, output_dir: str = "animations"):
        """
        Initialize layer animator.
        
        Args:
            output_dir: Directory for output animations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def animate_erosion_sequence(
        self,
        elevation_sequence: List[np.ndarray],
        discharge_sequence: List[np.ndarray],
        interval: int = 100,
        filename: str = "erosion_animation.gif"
    ):
        """
        Create animation from sequence of elevation and discharge maps.
        
        Args:
            elevation_sequence: List of elevation arrays at different times
            discharge_sequence: List of discharge arrays at different times
            interval: Milliseconds between frames
            filename: Output filename
        """
        print(f"Creating erosion animation with {len(elevation_sequence)} frames...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initialize plots
        im1 = ax1.imshow(elevation_sequence[0], cmap='terrain', interpolation='bilinear')
        im2 = ax2.imshow(discharge_sequence[0], cmap='Blues', interpolation='bilinear')
        
        ax1.set_title('Elevation', fontweight='bold')
        ax2.set_title('Discharge', fontweight='bold')
        ax1.axis('off')
        ax2.axis('off')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        frame_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12, fontweight='bold')
        
        def update_frame(frame_num):
            """Update animation frame"""
            im1.set_array(elevation_sequence[frame_num])
            im2.set_array(discharge_sequence[frame_num])
            frame_text.set_text(f'Frame {frame_num + 1}/{len(elevation_sequence)}')
            return im1, im2, frame_text
        
        anim = animation.FuncAnimation(
            fig,
            update_frame,
            frames=len(elevation_sequence),
            interval=interval,
            blit=True,
            repeat=True
        )
        
        # Save animation
        output_path = self.output_dir / filename
        anim.save(output_path, writer='pillow', fps=10)
        
        plt.close()
        
        print(f"✓ Animation saved: {output_path}")
    
    def create_comparison_visualization(
        self,
        elevation_before: np.ndarray,
        elevation_after: np.ndarray,
        discharge: np.ndarray,
        river_presence: np.ndarray,
        filename: str = "erosion_comparison.png"
    ):
        """
        Create before/after comparison visualization.
        
        Args:
            elevation_before: Initial elevation
            elevation_after: Final elevation
            discharge: Discharge field
            river_presence: River network boolean array
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        # Before
        im1 = axes[0, 0].imshow(elevation_before, cmap='terrain', interpolation='bilinear')
        axes[0, 0].set_title('Before Erosion', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # After
        im2 = axes[0, 1].imshow(elevation_after, cmap='terrain', interpolation='bilinear')
        axes[0, 1].set_title('After Erosion', fontweight='bold', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Difference
        diff = elevation_after - elevation_before
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im3 = axes[1, 0].imshow(
            diff, 
            cmap='RdBu_r', 
            interpolation='bilinear',
            vmin=-diff_max,
            vmax=diff_max
        )
        axes[1, 0].set_title('Elevation Change', fontweight='bold', fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Rivers
        axes[1, 1].imshow(elevation_after, cmap='terrain', interpolation='bilinear', alpha=0.5)
        river_display = np.zeros((*river_presence.shape, 4))
        river_display[river_presence] = [0, 0.9, 1.0, 1.0]
        axes[1, 1].imshow(river_display, interpolation='nearest')
        axes[1, 1].set_title('River Network', fontweight='bold', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.suptitle('Hydraulic Erosion Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison saved: {output_path}")