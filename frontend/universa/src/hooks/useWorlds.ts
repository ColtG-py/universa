/**
 * World Management Hooks
 * React hooks for creating, loading, and managing worlds.
 */

'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { api, World, CreateWorldParams, GenerationProgress, Settlement, ChunksResponse, SessionListItem } from '@/lib/api';

// =============================================================================
// useWorldList - List all available worlds
// =============================================================================

export function useWorldList() {
  const [worlds, setWorlds] = useState<World[]>([]);
  const [sessions, setSessions] = useState<Map<string, SessionListItem[]>>(new Map());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorlds = useCallback(async (limit = 20, offset = 0) => {
    setIsLoading(true);
    setError(null);
    try {
      // Fetch worlds and active sessions in parallel
      const [worldsResponse, sessionsResponse] = await Promise.all([
        api.listWorlds(limit, offset),
        api.listSessions(undefined, 'active', 100),
      ]);

      setWorlds(worldsResponse.worlds);

      // Group sessions by world_id
      const sessionsByWorld = new Map<string, SessionListItem[]>();
      for (const session of sessionsResponse.sessions) {
        const existing = sessionsByWorld.get(session.world_id) || [];
        existing.push(session);
        sessionsByWorld.set(session.world_id, existing);
      }
      setSessions(sessionsByWorld);

      return worldsResponse;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch worlds';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const deleteWorld = useCallback(async (worldId: string) => {
    try {
      await api.deleteWorld(worldId);
      setWorlds(prev => prev.filter(w => w.world_id !== worldId));
      // Also remove sessions for this world
      setSessions(prev => {
        const newMap = new Map(prev);
        newMap.delete(worldId);
        return newMap;
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete world';
      setError(message);
      throw err;
    }
  }, []);

  // Fetch on mount
  useEffect(() => {
    fetchWorlds();
  }, [fetchWorlds]);

  return {
    worlds,
    sessions,
    isLoading,
    error,
    refresh: fetchWorlds,
    deleteWorld,
  };
}

// =============================================================================
// useWorldCreation - Create a new world with progress tracking
// =============================================================================

export function useWorldCreation() {
  const [isCreating, setIsCreating] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [world, setWorld] = useState<World | null>(null);
  const pollInterval = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollInterval.current) {
      clearInterval(pollInterval.current);
      pollInterval.current = null;
    }
  }, []);

  const createWorld = useCallback(async (params: CreateWorldParams) => {
    setIsCreating(true);
    setError(null);
    setProgress(null);
    setWorld(null);

    try {
      // Create the world (starts generation in background)
      const newWorld = await api.createWorld(params);
      setWorld(newWorld);

      // Start polling for progress
      pollInterval.current = setInterval(async () => {
        try {
          const status = await api.getGenerationStatus(newWorld.world_id);
          setProgress(status);

          // Check if generation is complete
          if (status.status === 'ready' || status.status === 'complete') {
            stopPolling();
            setIsCreating(false);
            // Fetch final world data
            const finalWorld = await api.getWorld(newWorld.world_id);
            setWorld(finalWorld);
          } else if (status.status === 'failed') {
            stopPolling();
            setIsCreating(false);
            setError('World generation failed');
          }
        } catch (err) {
          // Ignore polling errors, will retry
          console.warn('Progress poll error:', err);
        }
      }, 1000);

      return newWorld;
    } catch (err) {
      setIsCreating(false);
      const message = err instanceof Error ? err.message : 'Failed to create world';
      setError(message);
      throw err;
    }
  }, [stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    createWorld,
    isCreating,
    progress,
    world,
    error,
    reset: () => {
      stopPolling();
      setIsCreating(false);
      setProgress(null);
      setWorld(null);
      setError(null);
    },
  };
}

// =============================================================================
// useWorld - Load a specific world and its data
// =============================================================================

export function useWorld(worldId: string | null) {
  const [world, setWorld] = useState<World | null>(null);
  const [settlements, setSettlements] = useState<Settlement[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadWorld = useCallback(async (id: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const [worldData, settlementsData] = await Promise.all([
        api.getWorld(id),
        api.getSettlements(id),
      ]);
      setWorld(worldData);
      setSettlements(settlementsData.settlements);
      return worldData;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load world';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load on mount if worldId provided
  useEffect(() => {
    if (worldId) {
      loadWorld(worldId);
    }
  }, [worldId, loadWorld]);

  return {
    world,
    settlements,
    isLoading,
    error,
    reload: loadWorld,
  };
}

// =============================================================================
// useWorldChunks - Load world chunks for viewport
// =============================================================================

export function useWorldChunks(worldId: string | null) {
  const [chunks, setChunks] = useState<ChunksResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastBounds = useRef<string>('');

  const loadChunks = useCallback(async (
    bounds: { xMin: number; xMax: number; yMin: number; yMax: number }
  ) => {
    if (!worldId) return null;

    // Debounce based on bounds
    const boundsKey = `${bounds.xMin},${bounds.xMax},${bounds.yMin},${bounds.yMax}`;
    if (boundsKey === lastBounds.current) return chunks;
    lastBounds.current = boundsKey;

    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getWorldChunks(worldId, bounds);
      setChunks(data);
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load chunks';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [worldId, chunks]);

  return {
    chunks,
    isLoading,
    error,
    loadChunks,
  };
}
