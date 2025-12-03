"""
World Management API Router
Endpoints for creating, loading, and managing worlds.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from uuid import UUID
import logging

from api.models.requests import (
    CreateWorldRequest,
    WorldQueryParams
)
from api.models.responses import (
    WorldResponse,
    WorldListResponse,
    WorldChunksResponse,
    GenerationProgressResponse
)
from api.services.world_service import WorldService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instance (will be dependency injected in production)
world_service = WorldService()


@router.post("", response_model=WorldResponse)
async def create_world(
    request: CreateWorldRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new world with the specified parameters.

    The world generation runs in the background. Poll the status
    endpoint to check progress.
    """
    try:
        world_id = await world_service.create_world(request)
        logger.info(f"Started world generation: {world_id}")

        return WorldResponse(
            world_id=world_id,
            name=request.name,
            status="generating",
            message="World generation started"
        )
    except Exception as e:
        logger.error(f"Failed to create world: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=WorldListResponse)
async def list_worlds(
    limit: int = 20,
    offset: int = 0
):
    """List all available worlds."""
    try:
        worlds = await world_service.list_worlds(limit=limit, offset=offset)
        return WorldListResponse(worlds=worlds, total=len(worlds))
    except Exception as e:
        logger.error(f"Failed to list worlds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}", response_model=WorldResponse)
async def get_world(world_id: UUID):
    """Get details of a specific world."""
    try:
        world = await world_service.get_world(str(world_id))
        if not world:
            raise HTTPException(status_code=404, detail="World not found")
        return world
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get world {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/status", response_model=GenerationProgressResponse)
async def get_generation_status(world_id: UUID):
    """Get the generation progress of a world."""
    try:
        status = await world_service.get_generation_status(str(world_id))
        if not status:
            raise HTTPException(status_code=404, detail="World not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/chunks", response_model=WorldChunksResponse)
async def get_world_chunks(
    world_id: UUID,
    x_min: int = -50,
    x_max: int = 50,
    y_min: int = -50,
    y_max: int = 50
):
    """
    Get world chunk data for the specified viewport.

    Returns tile data optimized for frontend rendering.
    """
    try:
        chunks = await world_service.get_chunks_in_range(
            world_id=str(world_id),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )
        return WorldChunksResponse(
            world_id=str(world_id),
            chunks=chunks,
            bounds={"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        )
    except Exception as e:
        logger.error(f"Failed to get chunks for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/settlements")
async def get_settlements(world_id: UUID):
    """Get all settlements in a world."""
    try:
        settlements = await world_service.get_settlements(str(world_id))
        return {"settlements": settlements}
    except Exception as e:
        logger.error(f"Failed to get settlements for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/factions")
async def get_factions(world_id: UUID):
    """Get all factions in a world."""
    try:
        factions = await world_service.get_factions(str(world_id))
        return {"factions": factions}
    except Exception as e:
        logger.error(f"Failed to get factions for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{world_id}")
async def delete_world(world_id: UUID):
    """Delete a world and all associated data."""
    try:
        success = await world_service.delete_world(str(world_id))
        if not success:
            raise HTTPException(status_code=404, detail="World not found")
        return {"status": "deleted", "world_id": str(world_id)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete world {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/chunk/{chunk_x}/{chunk_y}")
async def get_chunk(world_id: UUID, chunk_x: int, chunk_y: int):
    """
    Get detailed data for a specific chunk.

    Returns all layer data for the 256x256 tile chunk.
    """
    try:
        chunk = await world_service.get_chunk(
            world_id=str(world_id),
            chunk_x=chunk_x,
            chunk_y=chunk_y
        )
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk ({chunk_x}, {chunk_y}) for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/tile/{x}/{y}")
async def get_tile(world_id: UUID, x: int, y: int):
    """
    Get data for a specific tile coordinate.

    Returns elevation, biome, temperature, etc. for the exact location.
    """
    try:
        tile = await world_service.get_tile_data(
            world_id=str(world_id),
            x=x,
            y=y
        )
        if not tile:
            raise HTTPException(status_code=404, detail="Tile not found")
        return tile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tile ({x}, {y}) for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/roads")
async def get_roads(world_id: UUID):
    """Get all roads in a world."""
    try:
        roads = await world_service.get_roads(str(world_id))
        return {"roads": roads, "total": len(roads)}
    except Exception as e:
        logger.error(f"Failed to get roads for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{world_id}/ley-lines")
async def get_ley_lines(world_id: UUID):
    """Get all ley lines (magical conduits) in a world."""
    try:
        ley_lines = await world_service.get_ley_lines(str(world_id))
        return {"ley_lines": ley_lines, "total": len(ley_lines)}
    except Exception as e:
        logger.error(f"Failed to get ley lines for {world_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
