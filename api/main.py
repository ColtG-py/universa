"""
Universa API - FastAPI Application
Main entry point for the backend API server.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.config import get_settings
from api.routers import worlds, game, agents, player, party, debug, dialogue
from api.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection manager for WebSockets
ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Universa API...")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Supabase URL: {settings.supabase_url}")

    # Wire up service dependencies
    from api.routers.game import game_service
    from api.routers.debug import debug_service
    from api.routers import dialogue as dialogue_router
    from api.routers.agents import agent_service

    # Connect debug service to game service
    debug_service.set_game_service(game_service)

    # Connect dialogue service to game service
    dialogue_router.set_game_service(game_service)

    # Connect agent service to game service
    agent_service.set_game_service(game_service)

    logger.info("Service dependencies wired up")

    yield

    # Shutdown
    logger.info("Shutting down Universa API...")
    await ws_manager.disconnect_all()


# Create FastAPI app
app = FastAPI(
    title="Universa API",
    description="Backend API for the Universa AI-powered world simulation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(worlds.router, prefix="/api/worlds", tags=["worlds"])
app.include_router(game.router, prefix="/api/game", tags=["game"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(player.router, prefix="/api/player", tags=["player"])
app.include_router(party.router, prefix="/api/party", tags=["party"])
app.include_router(debug.router, prefix="/api/debug", tags=["debug"])
app.include_router(dialogue.router, prefix="/api/dialogue", tags=["dialogue"])


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "ok",
        "service": "Universa API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws/game/{session_id}")
async def game_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time game updates.

    Clients connect here to receive:
    - Tick updates (agent movements, actions)
    - Dialogue messages
    - World events
    - Debug information (if enabled)
    """
    await ws_manager.connect(websocket, session_id)
    try:
        while True:
            # Receive messages from client (player actions)
            data = await websocket.receive_json()
            await ws_manager.handle_client_message(session_id, data)
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
        logger.info(f"Client disconnected from session {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_mode
    )
