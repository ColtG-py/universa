# """
# World Builder - Visualizers Package
# Modular visualization system for each generation pass
# """

# from .base_visualizer import BaseVisualizer
# from .pass_01_planetary_viz import Pass01PlanetaryVisualizer
# from .pass_02_tectonics_viz import Pass02TectonicsVisualizer
# from .pass_03_topography_viz import Pass03TopographyVisualizer
# from .pass_04_geology_viz import Pass04GeologyVisualizer
# from .pass_05_atmosphere_viz import Pass05AtmosphereVisualizer
# from .pass_06_oceans_viz import Pass06OceansVisualizer
# from .pass_07_climate_viz import Pass07ClimateVisualizer
# from .pass_08_erosion_viz import Pass08ErosionVisualizer
# from .pass_09_groundwater_viz import Pass09GroundwaterVisualizer
# from .pass_10_rivers_viz import Pass10RiversVisualizer
# from .pass_11_soil_viz import Pass11SoilVisualizer
# from .pass_12_biomes_viz import Pass12BiomesVisualizer
# from .unified_visualizer import UnifiedVisualizer, create_visualization_summary

# __all__ = [
#     'BaseVisualizer',
#     'Pass01PlanetaryVisualizer',
#     'Pass02TectonicsVisualizer',
#     'Pass03TopographyVisualizer',
#     'Pass04GeologyVisualizer',
#     'Pass05AtmosphereVisualizer',
#     'Pass06OceansVisualizer',
#     'Pass07ClimateVisualizer',
#     'Pass08ErosionVisualizer',
#     'Pass09GroundwaterVisualizer',
#     'Pass10RiversVisualizer',
#     'Pass11SoilVisualizer',
#     'Pass12BiomesVisualizer',
#     'UnifiedVisualizer',
#     'create_visualization_summary',
# ]


"""
World Builder - Visualizers Package
Modular visualization system for each generation pass

Provides both matplotlib-based static visualizers and interactive Napari viewers.
Napari is recommended for interactive exploration during development.
"""

# Napari-based interactive visualizers (recommended)
from .base_napari_visualizer import BaseNapariVisualizer
from .unified_napari_visualizer import (
    UnifiedNapariVisualizer,
    view_world_interactive
)
NAPARI_AVAILABLE = True

# Legacy matplotlib visualizers can still be imported if needed
# from .unified_visualizer import UnifiedVisualizer, create_visualization_summary

__all__ = [
    # Napari visualizers (primary interface)
    'UnifiedNapariVisualizer',
    'view_world_interactive',
    'BaseNapariVisualizer',
    'NAPARI_AVAILABLE',
]