# World Builder - Complete File Hierarchy

This document shows the complete project structure, including where each file belongs and how the system is organized.

## Project Root Structure

```
world-builder/
├── README.md                           # Main project documentation
├── QUICK_START.md                      # Quick start guide
├── EXTENDING.md                        # Guide for extending the system
├── IMPLEMENTATION_SUMMARY.md           # What was built
├── FILE_HIERARCHY.md                   # This file
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore patterns
│
├── config.py                           # Global configuration and constants
├── main.py                             # FastAPI application entry point
├── demo.py                             # Demonstration/testing script
│
├── api/                                # REST API layer
│   ├── __init__.py
│   ├── routes.py                       # API endpoints
│   ├── models.py                       # Request/response models
│   ├── dependencies.py                 # FastAPI dependencies
│   └── middleware.py                   # Custom middleware
│
├── generation/                         # World generation engine
│   ├── __init__.py
│   ├── pipeline.py                     # Main generation orchestrator
│   │
│   ├── passes/                         # Generation passes (organized)
│   │   ├── __init__.py
│   │   │
│   │   ├── foundation/                 # Foundation passes (1-4)
│   │   │   ├── __init__.py
│   │   │   ├── pass_01_planetary.py
│   │   │   ├── pass_02_tectonics.py
│   │   │   ├── pass_03_topography.py
│   │   │   └── pass_04_geology.py
│   │   │
│   │   ├── climate/                    # Climate passes (5-7)
│   │   │   ├── __init__.py
│   │   │   ├── pass_05_atmosphere.py
│   │   │   ├── pass_06_oceans.py
│   │   │   └── pass_07_climate.py
│   │   │
│   │   ├── hydrology/                  # Water passes (8-10)
│   │   │   ├── __init__.py
│   │   │   ├── pass_08_erosion.py
│   │   │   ├── pass_09_groundwater.py
│   │   │   └── pass_10_rivers.py
│   │   │
│   │   └── detail/                     # Detail passes (11-14)
│   │       ├── __init__.py
│   │       ├── pass_11_soil.py
│   │       ├── pass_12_microclimate.py
│   │       ├── pass_13_features.py
│   │       └── pass_14_polish.py
│   │
│   └── validators.py                   # Parameter validation
│
├── models/                             # Data models and structures
│   ├── __init__.py
│   ├── world.py                        # World state and chunk models
│   ├── enums.py                        # Enumerations (rock types, etc.)
│   └── schemas.py                      # Pydantic schemas
│
├── storage/                            # Data persistence layer
│   ├── __init__.py
│   ├── supabase_client.py             # Supabase connection
│   ├── metadata.py                     # PostgreSQL operations
│   ├── chunks.py                       # Chunk serialization/loading
│   └── cache.py                        # Caching layer
│
├── utils/                              # Utility functions
│   ├── __init__.py
│   ├── noise.py                        # Noise generation
│   ├── spatial.py                      # Spatial calculations
│   ├── graph.py                        # Graph algorithms (rivers)
│   └── logging.py                      # Logging configuration
│
├── agents/                             # AI agents (future)
│   ├── __init__.py
│   ├── base_agent.py                   # Base agent class
│   ├── lore_historian.py               # Lore generation agent
│   ├── skill_architect.py              # Skill management agent
│   ├── world_simulator.py              # World simulation agent
│   └── memory_manager.py               # Memory management agent
│
├── skills/                             # Agent skills system (future)
│   ├── __init__.py
│   ├── skill_loader.py                 # Dynamic skill loading
│   ├── skill_validator.py              # Skill validation
│   │
│   └── definitions/                    # Skill definitions
│       ├── world_query.json
│       ├── lore_generation.json
│       └── simulation.json
│
├── memory/                             # Memory and state management (future)
│   ├── __init__.py
│   ├── conversation.py                 # Conversation memory
│   ├── world_state.py                  # World state tracking
│   └── cache.py                        # Memory caching
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest configuration
│   │
│   ├── generation/                     # Generation tests
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   ├── test_pass_01_planetary.py
│   │   ├── test_pass_02_tectonics.py
│   │   └── ...
│   │
│   ├── models/                         # Model tests
│   │   ├── __init__.py
│   │   ├── test_world.py
│   │   └── test_chunks.py
│   │
│   ├── utils/                          # Utility tests
│   │   ├── __init__.py
│   │   ├── test_noise.py
│   │   └── test_spatial.py
│   │
│   └── integration/                    # Integration tests
│       ├── __init__.py
│       └── test_full_generation.py
│
├── docs/                               # Additional documentation
│   ├── api/                            # API documentation
│   │   ├── endpoints.md
│   │   └── examples.md
│   │
│   ├── architecture/                   # Architecture docs
│   │   ├── overview.md
│   │   ├── data_flow.md
│   │   └── scaling.md
│   │
│   └── guides/                         # User guides
│       ├── getting_started.md
│       ├── customization.md
│       └── deployment.md
│
├── scripts/                            # Utility scripts
│   ├── setup_db.py                     # Database setup
│   ├── migrate.py                      # Database migrations
│   ├── export_world.py                 # World export utilities
│   └── benchmark.py                    # Performance benchmarking
│
├── examples/                           # Example implementations
│   ├── basic_generation.py
│   ├── custom_world.py
│   ├── chunk_on_demand.py
│   └── visualization.py
│
└── deployment/                         # Deployment configurations
    ├── docker/
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── .dockerignore
    │
    ├── kubernetes/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── configmap.yaml
    │
    └── terraform/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

## Current Implementation Status

### ✅ Implemented (Phase 1 - Core Generation)

```
world-builder/
├── config.py                           ✅ Complete
├── demo.py                             ✅ Complete
├── requirements.txt                    ✅ Complete
├── README.md                           ✅ Complete
├── QUICK_START.md                      ✅ Complete
├── EXTENDING.md                        ✅ Complete
├── IMPLEMENTATION_SUMMARY.md           ✅ Complete
│
├── generation/
│   ├── __init__.py                     ✅ Complete
│   ├── pipeline.py                     ✅ Complete
│   ├── pass_01_planetary.py            ✅ Complete
│   ├── pass_02_tectonics.py            ✅ Complete
│   ├── pass_03_topography.py           ✅ Complete
│   ├── pass_04_geology.py              ✅ Complete
│   ├── pass_05_atmosphere.py           ✅ Complete
│   ├── pass_06_oceans.py               ✅ Complete
│   ├── pass_07_climate.py              ✅ Complete
│   ├── pass_08_erosion.py              ✅ Complete
│   ├── pass_09_groundwater.py          ✅ Complete
│   ├── pass_10_rivers.py               ✅ Complete
│   ├── pass_11_soil.py                 ✅ Complete
│   ├── pass_12_microclimate.py         ✅ Complete
│   ├── pass_13_features.py             ✅ Complete
│   └── pass_14_polish.py               ✅ Complete
│
├── models/
│   ├── __init__.py                     ✅ Complete
│   └── world.py                        ✅ Complete
│
└── utils/
    ├── __init__.py                     ✅ Complete
    ├── noise.py                        ✅ Complete
    └── spatial.py                      ✅ Complete
```

### 📋 Planned (Phase 2 - API & Storage)

```
world-builder/
├── main.py                             📋 Planned
│
├── api/
│   ├── __init__.py                     📋 Planned
│   ├── routes.py                       📋 Planned
│   ├── models.py                       📋 Planned
│   ├── dependencies.py                 📋 Planned
│   └── middleware.py                   📋 Planned
│
└── storage/
    ├── __init__.py                     📋 Planned
    ├── supabase_client.py             📋 Planned
    ├── metadata.py                     📋 Planned
    ├── chunks.py                       📋 Planned
    └── cache.py                        📋 Planned
```

### 🔮 Future (Phase 3 - Agents & Skills)

```
world-builder/
├── agents/                             🔮 Future
├── skills/                             🔮 Future
├── memory/                             🔮 Future
└── tests/                              🔮 Future
```

## File Placement Guide

### Where Do I Put New Generation Passes?

```
generation/
├── pass_01_planetary.py               # Keep at root for now
├── pass_02_tectonics.py
├── ...
└── pass_14_polish.py

# Optional: Organize later
generation/passes/
├── foundation/pass_01_planetary.py    # Planetary, tectonics, topography, geology
├── climate/pass_05_atmosphere.py      # Atmosphere, oceans, climate
├── hydrology/pass_08_erosion.py       # Erosion, groundwater, rivers
└── detail/pass_11_soil.py             # Soil, microclimate, features, polish
```

### Where Do I Put New Models?

```
models/
├── world.py                           # Core world structures
├── enums.py                           # All enumerations (RockType, etc.)
└── schemas.py                         # Pydantic API schemas
```

### Where Do I Put New Utilities?

```
utils/
├── noise.py                           # Noise generation functions
├── spatial.py                         # Spatial calculations
├── graph.py                           # Graph algorithms (future)
└── [your_utility].py                  # New utilities here
```

### Where Do I Put API Endpoints?

```
api/
├── routes.py                          # All API endpoints
├── models.py                          # Request/response models
└── dependencies.py                    # Shared dependencies
```

### Where Do I Put Tests?

```
tests/
├── generation/
│   └── test_[pass_name].py           # One test file per pass
├── models/
│   └── test_[model_name].py          # One test file per model
└── utils/
    └── test_[util_name].py           # One test file per utility
```

## Moving from Current to Organized Structure

If you want to reorganize the generation passes into subdirectories:

```bash
# Create subdirectories
mkdir -p generation/passes/{foundation,climate,hydrology,detail}

# Move foundation passes
mv generation/pass_01_planetary.py generation/passes/foundation/
mv generation/pass_02_tectonics.py generation/passes/foundation/
mv generation/pass_03_topography.py generation/passes/foundation/
mv generation/pass_04_geology.py generation/passes/foundation/

# Move climate passes
mv generation/pass_05_atmosphere.py generation/passes/climate/
mv generation/pass_06_oceans.py generation/passes/climate/
mv generation/pass_07_climate.py generation/passes/climate/

# Move hydrology passes
mv generation/pass_08_erosion.py generation/passes/hydrology/
mv generation/pass_09_groundwater.py generation/passes/hydrology/
mv generation/pass_10_rivers.py generation/passes/hydrology/

# Move detail passes
mv generation/pass_11_soil.py generation/passes/detail/
mv generation/pass_12_microclimate.py generation/passes/detail/
mv generation/pass_13_features.py generation/passes/detail/
mv generation/pass_14_polish.py generation/passes/detail/

# Update imports in pipeline.py
# Change: from generation import pass_01_planetary
# To:     from generation.passes.foundation import pass_01_planetary
```

## Configuration Files

### .env.example
```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Generation Defaults
DEFAULT_WORLD_SIZE=1024
DEFAULT_NUM_PLATES=12
DEFAULT_SEED=42
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Generated Data
output/
exports/
*.npy
*.h5

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db
```

## Docker Structure

```
deployment/docker/
├── Dockerfile
├── docker-compose.yml
└── .dockerignore

# Dockerfile location: deployment/docker/Dockerfile
# Run from project root: docker build -f deployment/docker/Dockerfile .
```

## Quick Reference

### Add a New Pass
1. Create file: `generation/pass_XX_name.py`
2. Implement `execute(world_state, params)` function
3. Register in `generation/pipeline.py`
4. Add to `GENERATION_PASSES` in `config.py`
5. Add weight to `PASS_WEIGHTS` in `config.py`

### Add a New Model
1. Create or edit: `models/[name].py`
2. Export in `models/__init__.py`
3. Import where needed

### Add a New Utility
1. Create file: `utils/[name].py`
2. Export in `utils/__init__.py`
3. Import in passes that need it

### Add API Endpoints
1. Define endpoint in `api/routes.py`
2. Define schemas in `api/models.py`
3. Register router in `main.py`

## Directory Navigation

```bash
# From project root
cd generation/              # Generation engine
cd models/                  # Data models
cd utils/                   # Utilities
cd api/                     # API layer
cd tests/                   # Tests
cd docs/                    # Documentation
cd examples/                # Examples
cd scripts/                 # Scripts
```

## Integration Points

### Where Generation Connects to Other Systems

```
generation/pipeline.py
    ↓
    Generates → models/world.py (WorldState)
    ↓
    Saved by → storage/chunks.py
    ↓
    Queried via → api/routes.py
    ↓
    Used by → agents/world_simulator.py
```

This hierarchy is designed to scale from the current implementation to a full-featured world building system!