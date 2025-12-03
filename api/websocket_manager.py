"""
WebSocket Manager
Handles real-time connections for game sessions.
"""

from fastapi import WebSocket
from typing import Dict, Set, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    session_id: str
    player_id: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.utcnow)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time game updates.

    Supports:
    - Multiple connections per session
    - Subscription-based message filtering
    - Broadcast to session or specific connections
    - Connection lifecycle management
    """

    def __init__(self):
        # session_id -> list of connections
        self._connections: Dict[str, list[ConnectionInfo]] = {}
        # websocket -> connection info for quick lookup
        self._ws_to_info: Dict[WebSocket, ConnectionInfo] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str, player_id: Optional[str] = None) -> ConnectionInfo:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        info = ConnectionInfo(
            websocket=websocket,
            session_id=session_id,
            player_id=player_id,
            subscriptions={"tick", "chat", "event"}  # Default subscriptions
        )

        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = []
            self._connections[session_id].append(info)
            self._ws_to_info[websocket] = info

        logger.info(f"WebSocket connected: session={session_id}, player={player_id}")

        # Send connection confirmation
        await self._send(websocket, {
            "type": "connected",
            "session_id": session_id,
            "player_id": player_id,
            "subscriptions": list(info.subscriptions)
        })

        return info

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        async with self._lock:
            info = self._ws_to_info.pop(websocket, None)
            if info:
                session_conns = self._connections.get(info.session_id, [])
                self._connections[info.session_id] = [
                    c for c in session_conns if c.websocket != websocket
                ]
                # Clean up empty session entries
                if not self._connections[info.session_id]:
                    del self._connections[info.session_id]

                logger.info(f"WebSocket disconnected: session={info.session_id}, player={info.player_id}")

    async def subscribe(self, websocket: WebSocket, channels: list[str]):
        """Subscribe connection to specific channels."""
        info = self._ws_to_info.get(websocket)
        if info:
            info.subscriptions.update(channels)
            await self._send(websocket, {
                "type": "subscribed",
                "channels": channels,
                "subscriptions": list(info.subscriptions)
            })

    async def unsubscribe(self, websocket: WebSocket, channels: list[str]):
        """Unsubscribe connection from specific channels."""
        info = self._ws_to_info.get(websocket)
        if info:
            info.subscriptions.difference_update(channels)
            await self._send(websocket, {
                "type": "unsubscribed",
                "channels": channels,
                "subscriptions": list(info.subscriptions)
            })

    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        channel: Optional[str] = None
    ):
        """
        Broadcast message to all connections in a session.

        If channel is specified, only send to connections subscribed to that channel.
        """
        connections = self._connections.get(session_id, [])
        if not connections:
            return

        # Add metadata
        message["timestamp"] = datetime.utcnow().isoformat()
        message["session_id"] = session_id

        tasks = []
        for info in connections:
            # Check subscription if channel specified
            if channel and channel not in info.subscriptions:
                continue
            tasks.append(self._send(info.websocket, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_player(
        self,
        session_id: str,
        player_id: str,
        message: Dict[str, Any]
    ):
        """Send message to a specific player's connections."""
        connections = self._connections.get(session_id, [])

        message["timestamp"] = datetime.utcnow().isoformat()
        message["session_id"] = session_id

        tasks = []
        for info in connections:
            if info.player_id == player_id:
                tasks.append(self._send(info.websocket, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a single WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            # Connection likely dead, trigger cleanup
            await self.disconnect(websocket)

    # ==========================================================================
    # Message Type Helpers
    # ==========================================================================

    async def broadcast_tick(
        self,
        session_id: str,
        tick_number: int,
        game_time: str,
        events: list[Dict[str, Any]] = None
    ):
        """Broadcast tick update to session."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "tick",
                "tick_number": tick_number,
                "game_time": game_time,
                "events": events or []
            },
            channel="tick"
        )

    async def broadcast_agent_update(
        self,
        session_id: str,
        agent_id: str,
        changes: Dict[str, Any]
    ):
        """Broadcast agent state change."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "agent_update",
                "agent_id": agent_id,
                "changes": changes
            },
            channel="agent"
        )

    async def broadcast_chat(
        self,
        session_id: str,
        channel: str,
        sender_id: str,
        sender_name: str,
        message: str
    ):
        """Broadcast chat message."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "chat",
                "channel": channel,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "message": message
            },
            channel="chat"
        )

    async def broadcast_event(
        self,
        session_id: str,
        event_type: str,
        description: str,
        location: Optional[Dict[str, int]] = None,
        involved_agents: list[str] = None
    ):
        """Broadcast world event."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "event",
                "event_type": event_type,
                "description": description,
                "location": location,
                "involved_agents": involved_agents or []
            },
            channel="event"
        )

    async def send_debug_update(
        self,
        session_id: str,
        player_id: str,
        agent_id: str,
        debug_data: Dict[str, Any]
    ):
        """Send debug information to a player watching an agent."""
        await self.send_to_player(
            session_id,
            player_id,
            {
                "type": "debug",
                "agent_id": agent_id,
                "data": debug_data
            }
        )

    async def broadcast_dialogue(
        self,
        session_id: str,
        conversation_id: str,
        speaker_id: str,
        speaker_name: str,
        message: str,
        emotion: Optional[str] = None,
        is_player: bool = False
    ):
        """Broadcast a dialogue message."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "dialogue",
                "conversation_id": conversation_id,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "message": message,
                "emotion": emotion,
                "is_player": is_player
            },
            channel="chat"
        )

    async def broadcast_dialogue_started(
        self,
        session_id: str,
        conversation_id: str,
        participants: list[Dict[str, str]]
    ):
        """Broadcast that a dialogue has started."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "dialogue_started",
                "conversation_id": conversation_id,
                "participants": participants
            },
            channel="chat"
        )

    async def broadcast_dialogue_ended(
        self,
        session_id: str,
        conversation_id: str,
        reason: str = "ended"
    ):
        """Broadcast that a dialogue has ended."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "dialogue_ended",
                "conversation_id": conversation_id,
                "reason": reason
            },
            channel="chat"
        )

    async def broadcast_agent_action(
        self,
        session_id: str,
        agent_id: str,
        agent_name: str,
        action: str,
        location: Optional[Dict[str, int]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Broadcast an agent's action."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "agent_action",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "action": action,
                "location": location,
                "details": details or {}
            },
            channel="agent"
        )

    async def broadcast_tier_update(
        self,
        session_id: str,
        agent_id: str,
        old_tier: str,
        new_tier: str,
        reason: str
    ):
        """Broadcast agent tier change (for debug mode)."""
        await self.broadcast_to_session(
            session_id,
            {
                "type": "tier_update",
                "agent_id": agent_id,
                "old_tier": old_tier,
                "new_tier": new_tier,
                "reason": reason
            },
            channel="debug"
        )

    # ==========================================================================
    # Message Handling
    # ==========================================================================

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """
        Handle incoming WebSocket message.

        Message types:
        - subscribe: Subscribe to channels
        - unsubscribe: Unsubscribe from channels
        - ping: Respond with pong
        """
        msg_type = data.get("type")

        if msg_type == "subscribe":
            channels = data.get("channels", [])
            await self.subscribe(websocket, channels)

        elif msg_type == "unsubscribe":
            channels = data.get("channels", [])
            await self.unsubscribe(websocket, channels)

        elif msg_type == "ping":
            await self._send(websocket, {"type": "pong"})

        else:
            logger.warning(f"Unknown WebSocket message type: {msg_type}")

    # ==========================================================================
    # World Generation Progress
    # ==========================================================================

    async def broadcast_generation_progress(
        self,
        world_id: str,
        status: str,
        progress_percent: float,
        current_pass: str,
        pass_number: int,
        total_passes: int
    ):
        """Broadcast world generation progress to all connections watching this world."""
        # For now, broadcast to all connections
        # TODO: Track which connections are watching which worlds
        message = {
            "type": "generation_progress",
            "world_id": world_id,
            "status": status,
            "progress_percent": progress_percent,
            "current_pass": current_pass,
            "pass_number": pass_number,
            "total_passes": total_passes,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Broadcast to all sessions
        for session_id in list(self._connections.keys()):
            await self.broadcast_to_session(session_id, message, channel="generation")

    # ==========================================================================
    # Utilities
    # ==========================================================================

    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if session_id:
            return len(self._connections.get(session_id, []))
        return sum(len(conns) for conns in self._connections.values())

    def get_session_ids(self) -> list[str]:
        """Get all session IDs with active connections."""
        return list(self._connections.keys())

    def get_connection_info(self, websocket: WebSocket) -> Optional[ConnectionInfo]:
        """Get connection info for a WebSocket."""
        return self._ws_to_info.get(websocket)

    async def disconnect_all(self):
        """Disconnect all WebSocket connections (for shutdown)."""
        async with self._lock:
            for websocket in list(self._ws_to_info.keys()):
                try:
                    await websocket.close()
                except Exception:
                    pass
            self._connections.clear()
            self._ws_to_info.clear()
        logger.info("All WebSocket connections closed")

    async def handle_client_message(self, session_id: str, data: Dict[str, Any]):
        """
        Handle incoming message from a client.

        Routes messages to appropriate handlers based on type.
        """
        msg_type = data.get("type")

        if msg_type == "subscribe":
            # Subscribe to specific channels
            channels = data.get("channels", [])
            for info in self._connections.get(session_id, []):
                info.subscriptions.update(channels)
            logger.debug(f"Session {session_id} subscribed to {channels}")

        elif msg_type == "unsubscribe":
            channels = data.get("channels", [])
            for info in self._connections.get(session_id, []):
                info.subscriptions.difference_update(channels)
            logger.debug(f"Session {session_id} unsubscribed from {channels}")

        elif msg_type == "ping":
            await self.broadcast_to_session(session_id, {"type": "pong"})

        elif msg_type == "player_action":
            # Player action - will be handled by game service
            logger.debug(f"Player action in session {session_id}: {data}")
            # TODO: Route to game service

        elif msg_type == "chat":
            # Chat message - broadcast to session
            await self.broadcast_chat(
                session_id,
                channel=data.get("channel", "general"),
                sender_id=data.get("sender_id", "unknown"),
                sender_name=data.get("sender_name", "Unknown"),
                message=data.get("message", "")
            )
        else:
            logger.warning(f"Unknown message type: {msg_type}")


# Global instance
ws_manager = WebSocketManager()
