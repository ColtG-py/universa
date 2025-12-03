/**
 * Game State Store
 * Zustand store for managing game state with Supabase realtime sync.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  GameAgent,
  PlayerCharacter,
  GameSession,
  ChatMessage,
  WorldEvent,
  ViewportState,
  InputMode,
  TileData,
} from '@/types/game';

interface GameState {
  // World data
  worldId: string | null;
  worldName: string | null;
  tiles: Map<string, TileData>;

  // Agents
  agents: Map<string, GameAgent>;

  // Session
  session: GameSession | null;
  localPlayer: PlayerCharacter | null;

  // UI state
  viewport: ViewportState;
  inputMode: InputMode;
  selectedAgentId: string | null;
  hoveredTile: { x: number; y: number } | null;

  // Chat
  messages: ChatMessage[];

  // Events
  recentEvents: WorldEvent[];

  // Loading states
  isLoading: boolean;
  error: string | null;

  // Actions
  setWorld: (worldId: string, worldName: string) => void;
  setTiles: (tiles: TileData[]) => void;
  updateAgent: (agent: GameAgent) => void;
  removeAgent: (agentId: string) => void;
  setSession: (session: GameSession) => void;
  setLocalPlayer: (player: PlayerCharacter) => void;
  updateLocalPlayerPosition: (x: number, y: number) => void;
  setViewport: (viewport: Partial<ViewportState>) => void;
  setInputMode: (mode: InputMode) => void;
  selectAgent: (agentId: string | null) => void;
  setHoveredTile: (tile: { x: number; y: number } | null) => void;
  addMessage: (message: ChatMessage) => void;
  addEvent: (event: WorldEvent) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialViewport: ViewportState = {
  x: 0,
  y: 0,
  zoom: 1,
  width: 800,
  height: 600,
};

export const useGameStore = create<GameState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    worldId: null,
    worldName: null,
    tiles: new Map(),
    agents: new Map(),
    session: null,
    localPlayer: null,
    viewport: initialViewport,
    inputMode: 'explore',
    selectedAgentId: null,
    hoveredTile: null,
    messages: [],
    recentEvents: [],
    isLoading: false,
    error: null,

    // Actions
    setWorld: (worldId, worldName) => {
      set({ worldId, worldName, tiles: new Map(), agents: new Map() });
    },

    setTiles: (tiles) => {
      const tileMap = new Map<string, TileData>();
      for (const tile of tiles) {
        tileMap.set(`${tile.x},${tile.y}`, tile);
      }
      set({ tiles: tileMap });
    },

    updateAgent: (agent) => {
      set((state) => {
        const newAgents = new Map(state.agents);
        newAgents.set(agent.id, agent);
        return { agents: newAgents };
      });
    },

    removeAgent: (agentId) => {
      set((state) => {
        const newAgents = new Map(state.agents);
        newAgents.delete(agentId);
        return { agents: newAgents };
      });
    },

    setSession: (session) => {
      set({ session });
    },

    setLocalPlayer: (player) => {
      set({ localPlayer: player });
    },

    updateLocalPlayerPosition: (x, y) => {
      set((state) => {
        if (!state.localPlayer) return state;
        return {
          localPlayer: { ...state.localPlayer, x, y },
        };
      });
    },

    setViewport: (viewport) => {
      set((state) => ({
        viewport: { ...state.viewport, ...viewport },
      }));
    },

    setInputMode: (mode) => {
      set({ inputMode: mode });
    },

    selectAgent: (agentId) => {
      set({ selectedAgentId: agentId });
    },

    setHoveredTile: (tile) => {
      set({ hoveredTile: tile });
    },

    addMessage: (message) => {
      set((state) => ({
        messages: [...state.messages.slice(-99), message], // Keep last 100
      }));
    },

    addEvent: (event) => {
      set((state) => ({
        recentEvents: [...state.recentEvents.slice(-19), event], // Keep last 20
      }));
    },

    setLoading: (isLoading) => {
      set({ isLoading });
    },

    setError: (error) => {
      set({ error });
    },

    reset: () => {
      set({
        worldId: null,
        worldName: null,
        tiles: new Map(),
        agents: new Map(),
        session: null,
        localPlayer: null,
        viewport: initialViewport,
        inputMode: 'explore',
        selectedAgentId: null,
        hoveredTile: null,
        messages: [],
        recentEvents: [],
        isLoading: false,
        error: null,
      });
    },
  }))
);

// Selectors for optimized re-renders
export const selectAgentById = (id: string) => (state: GameState) =>
  state.agents.get(id);

export const selectVisibleAgents = (state: GameState) => {
  const { viewport, agents } = state;
  const visible: GameAgent[] = [];

  // Calculate visible tile range based on viewport
  const tileSize = 32; // pixels
  const startX = Math.floor(viewport.x / tileSize);
  const startY = Math.floor(viewport.y / tileSize);
  const endX = Math.ceil((viewport.x + viewport.width / viewport.zoom) / tileSize);
  const endY = Math.ceil((viewport.y + viewport.height / viewport.zoom) / tileSize);

  for (const agent of agents.values()) {
    if (
      agent.x >= startX &&
      agent.x <= endX &&
      agent.y >= startY &&
      agent.y <= endY
    ) {
      visible.push(agent);
    }
  }

  return visible;
};

export const selectTileAt = (x: number, y: number) => (state: GameState) =>
  state.tiles.get(`${x},${y}`);
