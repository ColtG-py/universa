/**
 * Game WebSocket Hook
 * Real-time updates from the game server.
 */

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useGameStore } from '@/stores/gameStore';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

interface WSMessage {
  type: string;
  timestamp?: string;
  session_id?: string;
  [key: string]: unknown;
}

interface TickUpdate {
  type: 'tick';
  tick_number: number;
  game_time: string;
  events: Array<{
    id: string;
    type: string;
    description: string;
    x: number;
    y: number;
  }>;
}

interface AgentUpdateMessage {
  type: 'agent_update';
  agent_id: string;
  changes: Record<string, unknown>;
}

interface DialogueMessage {
  type: 'dialogue' | 'dialogue_started' | 'dialogue_ended';
  conversation_id: string;
  speaker_id?: string;
  speaker_name?: string;
  message?: string;
  emotion?: string;
  is_player?: boolean;
  participants?: Array<{ id: string; name: string; type: string }>;
  reason?: string;
}

interface ChatMessage {
  type: 'chat';
  channel: string;
  sender_id: string;
  sender_name: string;
  message: string;
}

interface EventMessage {
  type: 'event';
  event_type: string;
  description: string;
  location?: { x: number; y: number };
  involved_agents?: string[];
}

interface TierUpdateMessage {
  type: 'tier_update';
  agent_id: string;
  old_tier: string;
  new_tier: string;
  reason: string;
}

interface GenerationProgressMessage {
  type: 'generation_progress';
  world_id: string;
  status: string;
  progress_percent: number;
  current_pass: string;
  pass_number: number;
  total_passes: number;
}

export function useGameWebSocket(sessionId: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [tickCount, setTickCount] = useState(0);
  const [gameTime, setGameTime] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const updateAgent = useGameStore(state => state.updateAgent);
  const addMessage = useGameStore(state => state.addMessage);
  const addEvent = useGameStore(state => state.addEvent);

  const handleMessage = useCallback((data: WSMessage) => {
    switch (data.type) {
      case 'tick': {
        const tickData = data as unknown as TickUpdate;
        setTickCount(tickData.tick_number);
        setGameTime(tickData.game_time);

        // Process events from tick
        for (const event of tickData.events || []) {
          addEvent({
            id: event.id,
            type: event.type as 'combat' | 'dialogue' | 'discovery' | 'trade' | 'movement',
            x: event.x,
            y: event.y,
            description: event.description,
            participants: [],
            timestamp: Date.now(),
          });
        }
        break;
      }

      case 'agent_update': {
        const agentData = data as unknown as AgentUpdateMessage;
        const changes = agentData.changes;
        updateAgent({
          id: agentData.agent_id,
          name: (changes.name as string) || 'Unknown',
          x: (changes.x as number) || 0,
          y: (changes.y as number) || 0,
          currentAction: (changes.current_action as string) || null,
          status: (changes.status as 'idle' | 'active' | 'sleeping' | 'dead') || 'idle',
          sprite: 'default',
          direction: 'down',
          traits: {
            occupation: 'Unknown',
            personality: [],
            goals: [],
          },
        });
        break;
      }

      case 'dialogue': {
        const dialogueData = data as unknown as DialogueMessage;
        addMessage({
          id: `dialogue-${Date.now()}`,
          senderId: dialogueData.speaker_id || 'unknown',
          senderName: dialogueData.speaker_name || 'Unknown',
          senderType: dialogueData.is_player ? 'player' : 'agent',
          content: dialogueData.message || '',
          timestamp: Date.now(),
          channel: 'whisper',
        });
        break;
      }

      case 'dialogue_started': {
        const startData = data as unknown as DialogueMessage;
        const names = (startData.participants || []).map(p => p.name).join(' and ');
        addMessage({
          id: `dialogue-start-${Date.now()}`,
          senderId: 'system',
          senderName: 'System',
          senderType: 'system',
          content: `Conversation started between ${names}`,
          timestamp: Date.now(),
          channel: 'narration',
        });
        break;
      }

      case 'dialogue_ended': {
        const endData = data as unknown as DialogueMessage;
        addMessage({
          id: `dialogue-end-${Date.now()}`,
          senderId: 'system',
          senderName: 'System',
          senderType: 'system',
          content: `Conversation ended: ${endData.reason || 'completed'}`,
          timestamp: Date.now(),
          channel: 'narration',
        });
        break;
      }

      case 'chat': {
        const chatData = data as unknown as ChatMessage;
        addMessage({
          id: `chat-${Date.now()}`,
          senderId: chatData.sender_id,
          senderName: chatData.sender_name,
          senderType: 'player',
          content: chatData.message,
          timestamp: Date.now(),
          channel: chatData.channel as 'global' | 'party' | 'whisper' | 'narration',
        });
        break;
      }

      case 'event': {
        const eventData = data as unknown as EventMessage;
        addEvent({
          id: `event-${Date.now()}`,
          type: eventData.event_type as 'combat' | 'dialogue' | 'discovery' | 'trade' | 'movement',
          x: eventData.location?.x || 0,
          y: eventData.location?.y || 0,
          description: eventData.description,
          participants: eventData.involved_agents || [],
          timestamp: Date.now(),
        });
        addMessage({
          id: `event-msg-${Date.now()}`,
          senderId: 'system',
          senderName: 'World',
          senderType: 'dm',
          content: eventData.description,
          timestamp: Date.now(),
          channel: 'narration',
        });
        break;
      }

      case 'tier_update': {
        const tierData = data as unknown as TierUpdateMessage;
        console.log(`Agent ${tierData.agent_id} tier changed: ${tierData.old_tier} -> ${tierData.new_tier} (${tierData.reason})`);
        break;
      }

      case 'connected': {
        console.log('WebSocket connected to session');
        break;
      }

      case 'pong': {
        // Heartbeat response
        break;
      }

      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }, [updateAgent, addMessage, addEvent]);

  const connect = useCallback(() => {
    if (!sessionId) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(`${WS_URL}/ws/game/${sessionId}`);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttempts.current = 0;

        // Subscribe to relevant channels
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: ['tick', 'agent', 'chat', 'event', 'debug'],
        }));
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);

        // Attempt reconnection if not a clean close
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeout.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('Connection error');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WSMessage;
          handleMessage(data);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setConnectionError('Failed to connect');
    }
  }, [sessionId, handleMessage]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  const sendChat = useCallback((channel: string, senderId: string, senderName: string, content: string) => {
    sendMessage({
      type: 'chat',
      channel,
      sender_id: senderId,
      sender_name: senderName,
      message: content,
    });
  }, [sendMessage]);

  const ping = useCallback(() => {
    sendMessage({ type: 'ping' });
  }, [sendMessage]);

  // Connect when sessionId is available
  useEffect(() => {
    if (sessionId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  // Heartbeat
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      ping();
    }, 30000);

    return () => clearInterval(interval);
  }, [isConnected, ping]);

  return {
    isConnected,
    connectionError,
    tickCount,
    gameTime,
    sendMessage,
    sendChat,
    connect,
    disconnect,
  };
}

// =============================================================================
// useGenerationProgress - WebSocket for world generation progress
// =============================================================================

export function useGenerationProgress(worldId: string | null) {
  const [progress, setProgress] = useState<GenerationProgressMessage | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!worldId) return;

    try {
      const ws = new WebSocket(`${WS_URL}/ws/generation/${worldId}`);

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as GenerationProgressMessage;
          if (data.type === 'generation_progress') {
            setProgress(data);
          }
        } catch (err) {
          console.error('Failed to parse progress message:', err);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to connect to generation WebSocket:', err);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [worldId]);

  return {
    progress,
    isConnected,
  };
}
