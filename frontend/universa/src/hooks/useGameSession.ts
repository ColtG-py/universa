/**
 * Game Session Hooks
 * React hooks for managing game sessions and simulation.
 */

'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  api,
  GameSession,
  PlayerConfig,
  PartyConfig,
  SessionSettings,
  TickResult,
  Agent,
} from '@/lib/api';
import { useGameStore } from '@/stores/gameStore';

// =============================================================================
// useGameSession - Create and manage a game session
// =============================================================================

export function useGameSession() {
  const [session, setSession] = useState<GameSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAutoTicking, setIsAutoTicking] = useState(false);

  const storeSetSession = useGameStore(state => state.setSession);
  const storeSetLocalPlayer = useGameStore(state => state.setLocalPlayer);

  const createSession = useCallback(async (
    worldId: string,
    player: PlayerConfig,
    party?: PartyConfig,
    settings?: SessionSettings
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const newSession = await api.createSession(worldId, player, party, settings);
      setSession(newSession);

      // Update store
      storeSetSession({
        id: newSession.session_id,
        worldId: newSession.world_id,
        name: `Session ${newSession.session_id.slice(0, 8)}`,
        status: 'active',
        players: [],
        dmAgentId: null,
        currentTurn: null,
        combatState: null,
      });

      if (newSession.player) {
        storeSetLocalPlayer({
          id: newSession.player.id,
          name: newSession.player.name,
          x: newSession.player.x,
          y: newSession.player.y,
          isAgent: false,
          agentId: null,
          stats: {
            health: newSession.player.stats?.health || 100,
            maxHealth: newSession.player.stats?.max_health || 100,
            mana: newSession.player.stats?.mana || 50,
            maxMana: newSession.player.stats?.max_mana || 50,
            strength: 10,
            dexterity: 10,
            constitution: 10,
            intelligence: 10,
            wisdom: 10,
            charisma: 10,
          },
          inventory: [],
        });
      }

      return newSession;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create session';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [storeSetSession, storeSetLocalPlayer]);

  const loadSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const loadedSession = await api.getSession(sessionId);
      setSession(loadedSession);
      return loadedSession;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load session';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startAutoTick = useCallback(async () => {
    if (!session) return;
    try {
      await api.startAutoTick(session.session_id);
      setIsAutoTicking(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start auto-tick';
      setError(message);
    }
  }, [session]);

  const pause = useCallback(async () => {
    if (!session) return;
    try {
      await api.pauseSession(session.session_id);
      setIsAutoTicking(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to pause session';
      setError(message);
    }
  }, [session]);

  const resume = useCallback(async () => {
    if (!session) return;
    try {
      await api.resumeSession(session.session_id);
      setIsAutoTicking(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to resume session';
      setError(message);
    }
  }, [session]);

  const stop = useCallback(async () => {
    if (!session) return;
    try {
      await api.stopSession(session.session_id);
      setIsAutoTicking(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to stop session';
      setError(message);
    }
  }, [session]);

  return {
    session,
    isLoading,
    error,
    isAutoTicking,
    createSession,
    loadSession,
    startAutoTick,
    pause,
    resume,
    stop,
  };
}

// =============================================================================
// useGameTick - Manual tick control
// =============================================================================

export function useGameTick(sessionId: string | null) {
  const [lastTick, setLastTick] = useState<TickResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tickCount, setTickCount] = useState(0);
  const [gameTime, setGameTime] = useState<string>('');

  const updateAgent = useGameStore(state => state.updateAgent);
  const addEvent = useGameStore(state => state.addEvent);
  const addMessage = useGameStore(state => state.addMessage);

  const tick = useCallback(async (numTicks = 1) => {
    if (!sessionId) return null;

    setIsProcessing(true);
    setError(null);
    try {
      const result = await api.tick(sessionId, numTicks);
      setLastTick(result);
      setTickCount(result.tick_number);
      setGameTime(result.game_time);

      // Process agent updates (with defensive check for missing array)
      const agentUpdates = result.agent_updates || [];
      for (const update of agentUpdates) {
        const agentData = update.changes as Record<string, unknown>;
        updateAgent({
          id: update.agent_id,
          name: (agentData.name as string) || 'Unknown',
          x: (agentData.x as number) || 0,
          y: (agentData.y as number) || 0,
          currentAction: (agentData.current_action as string) || null,
          status: (agentData.status as 'idle' | 'active' | 'sleeping' | 'dead') || 'idle',
          sprite: 'default',
          direction: 'down',
          traits: {
            occupation: (agentData.role as string) || 'Unknown',
            personality: [],
            goals: [],
          },
        });
      }

      // Process events (with defensive check for missing array)
      const events = result.events || [];
      for (const event of events) {
        addEvent({
          id: event.id || `event-${Date.now()}`,
          type: event.type as 'combat' | 'dialogue' | 'discovery' | 'trade' | 'movement',
          x: event.x || 0,
          y: event.y || 0,
          description: event.description || '',
          participants: [],
          timestamp: Date.now(),
        });

        // Also add as message for visibility
        addMessage({
          id: event.id || `event-${Date.now()}`,
          senderId: 'system',
          senderName: 'World',
          senderType: 'system',
          content: event.description || '',
          timestamp: Date.now(),
          channel: 'narration',
        });
      }

      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to process tick';
      setError(message);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, [sessionId, updateAgent, addEvent, addMessage]);

  // Fetch current time on mount
  useEffect(() => {
    if (sessionId) {
      api.getGameTime(sessionId).then(data => {
        setGameTime(data.game_time);
        setTickCount(data.tick);
      }).catch(() => {
        // Ignore errors on initial load
      });
    }
  }, [sessionId]);

  return {
    tick,
    lastTick,
    isProcessing,
    error,
    tickCount,
    gameTime,
  };
}

// =============================================================================
// useNearbyAgents - Load agents near the player
// =============================================================================

export function useNearbyAgents(sessionId: string | null, playerX: number, playerY: number) {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastPosition = useRef<string>('');

  const updateAgent = useGameStore(state => state.updateAgent);

  const loadNearbyAgents = useCallback(async (radius = 15) => {
    if (!sessionId) return [];

    // Debounce based on position
    const posKey = `${playerX},${playerY}`;
    if (posKey === lastPosition.current) return agents;
    lastPosition.current = posKey;

    setIsLoading(true);
    setError(null);
    try {
      const response = await api.getNearbyAgents(sessionId, playerX, playerY, radius);
      setAgents(response.agents);

      // Update store with agents
      for (const agent of response.agents) {
        updateAgent({
          id: agent.id,
          name: agent.name,
          x: agent.x,
          y: agent.y,
          currentAction: agent.current_action,
          status: (agent.status as 'idle' | 'active' | 'sleeping' | 'dead') || 'idle',
          sprite: 'default',
          direction: 'down',
          traits: {
            occupation: agent.traits?.occupation || 'Unknown',
            personality: agent.traits?.personality || [],
            goals: agent.traits?.goals || [],
          },
        });
      }

      return response.agents;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load agents';
      setError(message);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, playerX, playerY, agents, updateAgent]);

  // Note: Removed auto-load on position change to prevent clearing party members
  // Party members are loaded via tick updates; call refresh() manually when needed

  return {
    agents,
    isLoading,
    error,
    refresh: loadNearbyAgents,
  };
}

// =============================================================================
// usePlayerMovement - Handle player movement
// =============================================================================

export function usePlayerMovement(sessionId: string | null, playerId: string | null) {
  const [isMoving, setIsMoving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateLocalPlayerPosition = useGameStore(state => state.updateLocalPlayerPosition);

  const moveTo = useCallback(async (x: number, y: number) => {
    if (!sessionId || !playerId) return false;

    setIsMoving(true);
    setError(null);
    try {
      const result = await api.movePlayer(sessionId, playerId, x, y);
      if (result.success) {
        updateLocalPlayerPosition(result.new_position.x, result.new_position.y);
        // Also update player position for tier recalculation
        await api.updatePlayerPosition(sessionId, result.new_position.x, result.new_position.y);
      }
      return result.success;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to move';
      setError(message);
      return false;
    } finally {
      setIsMoving(false);
    }
  }, [sessionId, playerId, updateLocalPlayerPosition]);

  return {
    moveTo,
    isMoving,
    error,
  };
}
