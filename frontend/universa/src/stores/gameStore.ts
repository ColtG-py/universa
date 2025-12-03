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
  GameTime,
  CurrentLocation,
  PartyMember,
  UIPanel,
  JournalEntry,
  Quest,
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

  // HUD state
  gameTime: GameTime;
  currentLocation: CurrentLocation | null;
  openPanel: UIPanel;
  partySlots: [PartyMember | null, PartyMember | null, PartyMember | null, PartyMember | null];

  // Journal & Quests
  journalEntries: JournalEntry[];
  quests: Quest[];

  // Chat
  messages: ChatMessage[];

  // Events
  recentEvents: WorldEvent[];

  // Loading states
  isLoading: boolean;
  error: string | null;

  // Chunk tracking for efficient loading
  loadedChunkRegions: Set<string>;

  // Actions
  setWorld: (worldId: string, worldName: string) => void;
  setTiles: (tiles: TileData[]) => void;
  addTiles: (tiles: TileData[]) => void;
  markChunkLoaded: (chunkKey: string) => void;
  isChunkLoaded: (chunkKey: string) => boolean;
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

  // HUD actions
  setGameTime: (time: Partial<GameTime>) => void;
  advanceTime: (minutes: number) => void;
  setCurrentLocation: (location: CurrentLocation | null) => void;
  setOpenPanel: (panel: UIPanel) => void;
  togglePanel: (panel: NonNullable<UIPanel>) => void;
  setPartySlot: (slot: 0 | 1 | 2 | 3, member: PartyMember | null) => void;

  // Journal & Quest actions
  addJournalEntry: (entry: JournalEntry) => void;
  addQuest: (quest: Quest) => void;
  updateQuest: (questId: string, updates: Partial<Quest>) => void;
  completeQuestObjective: (questId: string, objectiveId: string) => void;
}

const initialViewport: ViewportState = {
  x: 0,
  y: 0,
  zoom: 1,
  width: 800,
  height: 600,
};

const initialGameTime: GameTime = {
  hour: 8,
  minute: 0,
  day: 1,
};

const initialPartySlots: [PartyMember | null, PartyMember | null, PartyMember | null, PartyMember | null] = [null, null, null, null];

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

    // HUD state
    gameTime: initialGameTime,
    currentLocation: null,
    openPanel: null,
    partySlots: initialPartySlots,

    // Journal & Quests
    journalEntries: [],
    quests: [],

    messages: [],
    recentEvents: [],
    isLoading: false,
    error: null,
    loadedChunkRegions: new Set(),

    // Actions
    setWorld: (worldId, worldName) => {
      set({ worldId, worldName, tiles: new Map(), agents: new Map() });
    },

    setTiles: (tiles) => {
      const tileMap = new Map<string, TileData>();
      for (const tile of tiles) {
        tileMap.set(`${tile.x},${tile.y}`, tile);
      }
      set({ tiles: tileMap, loadedChunkRegions: new Set() });
    },

    addTiles: (tiles) => {
      set((state) => {
        const newTileMap = new Map(state.tiles);
        for (const tile of tiles) {
          newTileMap.set(`${tile.x},${tile.y}`, tile);
        }
        return { tiles: newTileMap };
      });
    },

    markChunkLoaded: (chunkKey) => {
      set((state) => {
        const newRegions = new Set(state.loadedChunkRegions);
        newRegions.add(chunkKey);
        return { loadedChunkRegions: newRegions };
      });
    },

    isChunkLoaded: (chunkKey) => {
      return get().loadedChunkRegions.has(chunkKey);
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
        gameTime: initialGameTime,
        currentLocation: null,
        openPanel: null,
        partySlots: initialPartySlots,
        journalEntries: [],
        quests: [],
        messages: [],
        recentEvents: [],
        isLoading: false,
        error: null,
      });
    },

    // HUD actions
    setGameTime: (time) => {
      set((state) => ({
        gameTime: { ...state.gameTime, ...time },
      }));
    },

    advanceTime: (minutes) => {
      set((state) => {
        let { hour, minute, day } = state.gameTime;
        minute += minutes;

        while (minute >= 60) {
          minute -= 60;
          hour += 1;
        }
        while (minute < 0) {
          minute += 60;
          hour -= 1;
        }
        while (hour >= 24) {
          hour -= 24;
          day += 1;
        }
        while (hour < 0) {
          hour += 24;
          day -= 1;
        }

        return { gameTime: { hour, minute, day } };
      });
    },

    setCurrentLocation: (location) => {
      set({ currentLocation: location });
    },

    setOpenPanel: (panel) => {
      set({ openPanel: panel });
    },

    togglePanel: (panel) => {
      set((state) => ({
        openPanel: state.openPanel === panel ? null : panel,
      }));
    },

    setPartySlot: (slot, member) => {
      set((state) => {
        const newSlots = [...state.partySlots] as typeof state.partySlots;
        newSlots[slot] = member;
        return { partySlots: newSlots };
      });
    },

    // Journal & Quest actions
    addJournalEntry: (entry) => {
      set((state) => ({
        journalEntries: [entry, ...state.journalEntries],
      }));
    },

    addQuest: (quest) => {
      set((state) => ({
        quests: [...state.quests, quest],
      }));
    },

    updateQuest: (questId, updates) => {
      set((state) => ({
        quests: state.quests.map((q) =>
          q.id === questId ? { ...q, ...updates } : q
        ),
      }));
    },

    completeQuestObjective: (questId, objectiveId) => {
      set((state) => ({
        quests: state.quests.map((q) => {
          if (q.id !== questId) return q;
          const objectives = q.objectives.map((obj) =>
            obj.id === objectiveId ? { ...obj, isCompleted: true } : obj
          );
          const allComplete = objectives.every((obj) => obj.isCompleted);
          return {
            ...q,
            objectives,
            status: allComplete ? 'completed' : q.status,
          };
        }),
      }));
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

export const selectIsNight = (state: GameState) => {
  const { hour } = state.gameTime;
  return hour < 6 || hour >= 20;
};

export const selectActiveQuests = (state: GameState) =>
  state.quests.filter((q) => q.status === 'active');

export const selectCompletedQuests = (state: GameState) =>
  state.quests.filter((q) => q.status === 'completed');
