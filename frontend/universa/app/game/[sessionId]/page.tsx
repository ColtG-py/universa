'use client';

import { use, useEffect, useState, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { useGameStore } from '@/stores/gameStore';
import { useGameSession, useGameTick, useNearbyAgents } from '@/hooks/useGameSession';
import { useGameWebSocket } from '@/hooks/useGameWebSocket';
import { useWorldChunks } from '@/hooks/useWorlds';
import { useDialogue } from '@/hooks/useDialogue';
import { api } from '@/lib/api';
import CharacterSheet from '@/components/ui/CharacterSheet';
import AgentInfoPanel from '@/components/ui/AgentInfoPanel';
import TileInfoPanel from '@/components/ui/TileInfoPanel';
import LocationDisplay from '@/components/ui/LocationDisplay';
import TimeIndicator from '@/components/ui/TimeIndicator';
import MenuButtons from '@/components/ui/MenuButtons';
import PartyListBar from '@/components/ui/PartyListBar';
import Minimap from '@/components/ui/Minimap';
import DMPlaceholder from '@/components/ui/DMPlaceholder';
import SettingsPanel from '@/components/ui/SettingsPanel';
import JournalPanel from '@/components/ui/JournalPanel';
import QuestsPanel from '@/components/ui/QuestsPanel';
import InventoryPanel from '@/components/ui/InventoryPanel';
import type { ChatMessage } from '@/types/game';

const TILE_SIZE = 32;
const MOVEMENT_SPEED = 1; // tiles per keypress

// Dynamic import for GameCanvas
const GameCanvas = dynamic(() => import('@/components/game/GameCanvas'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-gray-900">
      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-amber-400"></div>
    </div>
  ),
});

export default function GameSessionPage({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = use(params);
  const router = useRouter();

  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [sessionData, setSessionData] = useState<{
    worldId: string;
    status: string;
    currentTick: number;
    gameTime: string;
  } | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // Store state
  const localPlayer = useGameStore(state => state.localPlayer);
  const inputMode = useGameStore(state => state.inputMode);
  const selectedAgentId = useGameStore(state => state.selectedAgentId);
  const agents = useGameStore(state => state.agents);
  const addMessage = useGameStore(state => state.addMessage);
  const setTiles = useGameStore(state => state.setTiles);
  const addTiles = useGameStore(state => state.addTiles);
  const markChunkLoaded = useGameStore(state => state.markChunkLoaded);
  const isChunkLoaded = useGameStore(state => state.isChunkLoaded);
  const setWorld = useGameStore(state => state.setWorld);
  const setInputMode = useGameStore(state => state.setInputMode);
  const selectAgent = useGameStore(state => state.selectAgent);
  const setLocalPlayer = useGameStore(state => state.setLocalPlayer);
  const setViewport = useGameStore(state => state.setViewport);

  // HUD state
  const openPanel = useGameStore(state => state.openPanel);
  const setOpenPanel = useGameStore(state => state.setOpenPanel);
  const togglePanel = useGameStore(state => state.togglePanel);
  const setGameTime = useGameStore(state => state.setGameTime);
  const setCurrentLocation = useGameStore(state => state.setCurrentLocation);

  // Hooks
  const { loadSession, isLoading: sessionLoading, error: sessionError, pause, resume, isAutoTicking } = useGameSession();
  const { tick, isProcessing, tickCount, gameTime } = useGameTick(sessionId);
  const { isConnected, sendChat } = useGameWebSocket(sessionId);
  const { loadChunks } = useWorldChunks(sessionData?.worldId || null);
  const updateLocalPlayerPosition = useGameStore(state => state.updateLocalPlayerPosition);
  const viewport = useGameStore(state => state.viewport);
  const { refresh: refreshAgents } = useNearbyAgents(
    sessionId,
    localPlayer?.x || 0,
    localPlayer?.y || 0
  );

  // Dialogue hook
  const {
    conversation,
    isLoading: dialogueLoading,
    isWaitingForResponse,
    startConversation,
    sendMessage: sendDialogueMessage,
    getAgentResponse,
    endConversation,
    isActive: isInDialogue,
  } = useDialogue();

  // Track if already initialized to prevent re-running
  const hasInitialized = useRef(false);

  // Initialize session (run once only)
  useEffect(() => {
    // Only run once
    if (hasInitialized.current) return;
    hasInitialized.current = true;

    const init = async () => {
      try {
        // Type for API response
        interface SessionAPIResponse {
          session_id: string;
          world_id: string;
          status: string;
          current_tick: number;
          game_time: string;
          player: {
            id: string;
            name: string;
            x: number;
            y: number;
            health: number;
            max_health: number;
            stamina: number;
            max_stamina: number;
            stats: Record<string, number>;
          };
          party_size: number;
        }

        const session = await api.getSession(sessionId) as unknown as SessionAPIResponse;
        console.log('Session loaded:', session);

        setSessionData({
          worldId: session.world_id,
          status: session.status,
          currentTick: session.current_tick || 0,
          gameTime: session.game_time || '00:00',
        });

        // Set world in store (but don't clear agents - use direct set instead)
        useGameStore.setState({
          worldId: session.world_id,
          worldName: `World ${session.world_id.slice(0, 8)}`,
        });

        // Set local player in store
        if (session.player) {
          const playerStats = session.player.stats || {};
          setLocalPlayer({
            id: session.player.id,
            name: session.player.name,
            x: session.player.x,
            y: session.player.y,
            isAgent: false,
            agentId: null,
            stats: {
              health: session.player.health ?? playerStats.health ?? 100,
              maxHealth: session.player.max_health ?? playerStats.maxHealth ?? 100,
              mana: playerStats.mana ?? 50,
              maxMana: playerStats.maxMana ?? 50,
              strength: playerStats.strength ?? 10,
              dexterity: playerStats.dexterity ?? 10,
              constitution: playerStats.constitution ?? 10,
              intelligence: playerStats.intelligence ?? 10,
              wisdom: playerStats.wisdom ?? 10,
              charisma: playerStats.charisma ?? 10,
            },
            inventory: [],
          });

          // Center viewport on player position
          // Player at (px, py) should be in center of screen
          // So viewport top-left = (px * TILE_SIZE - width/2, py * TILE_SIZE - height/2)
          const TILE_SIZE = 32;
          setViewport({
            x: session.player.x * TILE_SIZE - canvasSize.width / 2,
            y: session.player.y * TILE_SIZE - canvasSize.height / 2,
          });

          // Load initial chunks around player using chunk regions
          const CHUNK_SIZE = 30;
          const playerChunkX = Math.floor(session.player.x / CHUNK_SIZE);
          const playerChunkY = Math.floor(session.player.y / CHUNK_SIZE);

          // Mark initial 3x3 chunk regions as loaded
          const markChunkLoaded = useGameStore.getState().markChunkLoaded;
          for (let cx = playerChunkX - 1; cx <= playerChunkX + 1; cx++) {
            for (let cy = playerChunkY - 1; cy <= playerChunkY + 1; cy++) {
              markChunkLoaded(`${cx},${cy}`);
            }
          }

          const bounds = {
            xMin: (playerChunkX - 1) * CHUNK_SIZE,
            xMax: (playerChunkX + 2) * CHUNK_SIZE - 1,
            yMin: (playerChunkY - 1) * CHUNK_SIZE,
            yMax: (playerChunkY + 2) * CHUNK_SIZE - 1,
          };
          console.log('Loading initial chunks with bounds:', bounds);
          const chunksData = await api.getWorldChunks(session.world_id, bounds);
          console.log('Chunks data received:', chunksData);

          if (chunksData.chunks && chunksData.chunks.length > 0) {
            // Transform API tile data to frontend format
            const tiles = chunksData.chunks.map(chunk => ({
              x: chunk.x,
              y: chunk.y,
              elevation: chunk.elevation || 0,
              biomeType: chunk.biome_type || 'temperate_grassland',
              temperatureC: chunk.temperature_c || 15,
              hasRoad: chunk.has_road || false,
              hasRiver: chunk.has_river || false,
              settlementId: chunk.settlement_id,
              settlementType: chunk.settlement_type,
              factionName: chunk.faction_name,
              resourceType: chunk.resource_type,
            }));
            console.log(`Setting ${tiles.length} initial tiles (${bounds.xMax - bounds.xMin + 1}x${bounds.yMax - bounds.yMin + 1} area)`);
            setTiles(tiles);
          } else {
            console.warn('No chunks data received');
          }

          // Load nearby agents (initial load only)
          try {
            const response = await api.getNearbyAgents(sessionId, session.player.x, session.player.y, 15);
            const updateAgent = useGameStore.getState().updateAgent;
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
          } catch (err) {
            console.error('Failed to load nearby agents:', err);
          }
        }

        setIsInitializing(false);
      } catch (err) {
        console.error('Failed to initialize session:', err);
        setIsInitializing(false);
      }
    };

    init();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Handle window resize - fullscreen canvas
  useEffect(() => {
    const updateSize = () => {
      setCanvasSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Handle chat messages
  const handleSendMessage = useCallback(
    (content: string, channel: ChatMessage['channel']) => {
      if (!localPlayer) return;

      // If in dialogue mode, send to dialogue service
      if (inputMode === 'dialogue' && isInDialogue && conversation) {
        const agent = conversation.participants.find(p => p.type === 'agent');
        if (agent) {
          sendDialogueMessage(content, localPlayer.id, localPlayer.name)
            .then(() => getAgentResponse(agent.id, agent.name))
            .catch(console.error);
        }
        return;
      }

      // Regular chat message
      const message: ChatMessage = {
        id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        senderId: localPlayer.id,
        senderName: localPlayer.name,
        senderType: 'player',
        content,
        timestamp: Date.now(),
        channel,
      };

      addMessage(message);

      // Broadcast via WebSocket
      if (isConnected) {
        sendChat(channel, localPlayer.id, localPlayer.name, content);
      }
    },
    [localPlayer, addMessage, isConnected, sendChat, inputMode, isInDialogue, conversation, sendDialogueMessage, getAgentResponse]
  );

  // Handle agent interaction (talk)
  const handleTalkToAgent = useCallback(async () => {
    if (!localPlayer || !selectedAgentId || !sessionId) return;

    const agent = agents.get(selectedAgentId);
    if (!agent) return;

    try {
      await startConversation({
        sessionId,
        playerId: localPlayer.id,
        playerName: localPlayer.name,
        agentId: agent.id,
        agentName: agent.name,
        playerX: localPlayer.x,
        playerY: localPlayer.y,
      });
    } catch (err) {
      console.error('Failed to start conversation:', err);
    }
  }, [localPlayer, selectedAgentId, sessionId, agents, startConversation]);

  // Handle end dialogue
  const handleEndDialogue = useCallback(async () => {
    try {
      await endConversation();
      selectAgent(null);
    } catch (err) {
      console.error('Failed to end conversation:', err);
    }
  }, [endConversation, selectAgent]);

  // Handle player movement
  const handleMove = useCallback(async (dx: number, dy: number) => {
    if (!localPlayer || !sessionId || !sessionData?.worldId) return;

    const newX = localPlayer.x + dx * MOVEMENT_SPEED;
    const newY = localPlayer.y + dy * MOVEMENT_SPEED;

    // Update local player position immediately for responsive feel
    updateLocalPlayerPosition(newX, newY);

    // Update viewport to follow player
    setViewport({
      x: newX * TILE_SIZE - canvasSize.width / 2,
      y: newY * TILE_SIZE - canvasSize.height / 2,
    });

    // Sync with server (fire and forget for now)
    try {
      await api.movePlayer(sessionId, localPlayer.id, newX, newY);
      await api.updatePlayerPosition(sessionId, newX, newY);
    } catch (err) {
      console.error('Failed to sync player position:', err);
    }

    // Check if we need to load new chunks based on chunk regions
    // Chunks are 30x30 tiles, we load when entering a new chunk region
    const CHUNK_SIZE = 30;
    const currentChunkX = Math.floor(newX / CHUNK_SIZE);
    const currentChunkY = Math.floor(newY / CHUNK_SIZE);

    // Check surrounding chunk regions (3x3 around current position)
    const chunksToLoad: { x: number; y: number }[] = [];
    for (let cx = currentChunkX - 1; cx <= currentChunkX + 1; cx++) {
      for (let cy = currentChunkY - 1; cy <= currentChunkY + 1; cy++) {
        const chunkKey = `${cx},${cy}`;
        if (!isChunkLoaded(chunkKey)) {
          chunksToLoad.push({ x: cx, y: cy });
          markChunkLoaded(chunkKey);
        }
      }
    }

    // Load any unloaded chunks
    if (chunksToLoad.length > 0) {
      // Calculate bounds to cover all chunks we need to load
      const minChunkX = Math.min(...chunksToLoad.map(c => c.x));
      const maxChunkX = Math.max(...chunksToLoad.map(c => c.x));
      const minChunkY = Math.min(...chunksToLoad.map(c => c.y));
      const maxChunkY = Math.max(...chunksToLoad.map(c => c.y));

      const bounds = {
        xMin: minChunkX * CHUNK_SIZE,
        xMax: (maxChunkX + 1) * CHUNK_SIZE - 1,
        yMin: minChunkY * CHUNK_SIZE,
        yMax: (maxChunkY + 1) * CHUNK_SIZE - 1,
      };

      console.log(`Loading ${chunksToLoad.length} new chunk regions:`, chunksToLoad.map(c => `${c.x},${c.y}`).join(', '));

      try {
        const chunksData = await api.getWorldChunks(sessionData.worldId, bounds);
        if (chunksData.chunks && chunksData.chunks.length > 0) {
          const tiles = chunksData.chunks.map(chunk => ({
            x: chunk.x,
            y: chunk.y,
            elevation: chunk.elevation || 0,
            biomeType: chunk.biome_type || 'temperate_grassland',
            temperatureC: chunk.temperature_c || 15,
            hasRoad: chunk.has_road || false,
            hasRiver: chunk.has_river || false,
            settlementId: chunk.settlement_id,
            settlementType: chunk.settlement_type,
            factionName: chunk.faction_name,
            resourceType: chunk.resource_type,
          }));
          console.log(`Adding ${tiles.length} new tiles`);
          addTiles(tiles);
        }
      } catch (err) {
        console.error('Failed to load chunks:', err);
        // Unmark chunks on failure so they can be retried
        chunksToLoad.forEach(c => {
          // Note: We'd need to add a removeChunkLoaded action to properly handle this
        });
      }
    }

    // Note: Don't call refreshAgents() on every move - it clears party members
    // Party members are updated via tick, nearby world agents can be loaded on demand
  }, [localPlayer, sessionId, sessionData, updateLocalPlayerPosition, setViewport, canvasSize, isChunkLoaded, markChunkLoaded, addTiles]);

  // Handle party member talk
  const handleTalkToPartyMember = useCallback(async (memberId: string) => {
    if (!localPlayer || !sessionId) return;

    const agent = agents.get(memberId);
    if (!agent) return;

    try {
      await startConversation({
        sessionId,
        playerId: localPlayer.id,
        playerName: localPlayer.name,
        agentId: agent.id,
        agentName: agent.name,
        playerX: localPlayer.x,
        playerY: localPlayer.y,
      });
    } catch (err) {
      console.error('Failed to start conversation with party member:', err);
    }
  }, [localPlayer, sessionId, agents, startConversation]);

  // Handle party member selection
  const handleSelectPartyMember = useCallback((memberId: string) => {
    selectAgent(memberId);
  }, [selectAgent]);

  // Close panel helper
  const closePanel = useCallback(() => setOpenPanel(null), [setOpenPanel]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      ) {
        return;
      }

      switch (e.key.toLowerCase()) {
        // Movement - WASD
        case 'w':
          e.preventDefault();
          handleMove(0, -1);
          break;
        case 'a':
          e.preventDefault();
          handleMove(-1, 0);
          break;
        case 's':
          e.preventDefault();
          handleMove(0, 1);
          break;
        case 'd':
          e.preventDefault();
          handleMove(1, 0);
          break;
        // Movement - Arrow keys
        case 'arrowup':
          e.preventDefault();
          handleMove(0, -1);
          break;
        case 'arrowdown':
          e.preventDefault();
          handleMove(0, 1);
          break;
        case 'arrowleft':
          e.preventDefault();
          handleMove(-1, 0);
          break;
        case 'arrowright':
          e.preventDefault();
          handleMove(1, 0);
          break;
        // UI shortcuts
        case 'c':
          togglePanel('character');
          break;
        case 'i':
          togglePanel('inventory');
          break;
        case 'j':
          togglePanel('journal');
          break;
        case 'q':
          togglePanel('quests');
          break;
        case 'escape':
          if (openPanel) {
            closePanel();
          } else if (isInDialogue) {
            handleEndDialogue();
          } else {
            togglePanel('settings');
          }
          break;
        case ' ':
          e.preventDefault();
          tick();
          break;
        case 't':
          if (selectedAgentId && !isInDialogue) {
            handleTalkToAgent();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tick, selectedAgentId, isInDialogue, handleTalkToAgent, handleEndDialogue, handleMove, togglePanel, openPanel, closePanel]);

  // Loading state
  if (isInitializing) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-amber-400 mx-auto mb-4"></div>
          <div className="text-gray-400">Loading game session...</div>
        </div>
      </div>
    );
  }

  // Error state
  if (sessionError) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">😞</div>
          <h2 className="text-2xl font-semibold mb-2">Session Error</h2>
          <p className="text-gray-400 mb-6">{sessionError}</p>
          <Link
            href="/worlds"
            className="inline-block px-6 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
          >
            Back to Worlds
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-screen bg-gray-950 overflow-hidden">
      {/* Fullscreen game canvas */}
      <GameCanvas width={canvasSize.width} height={canvasSize.height} />

      {/* HUD Overlay Layer */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top bar */}
        <div className="absolute top-4 left-4 pointer-events-auto">
          <LocationDisplay />
        </div>
        <div className="absolute top-4 right-4 pointer-events-auto">
          <TimeIndicator />
        </div>
        <div className="absolute top-4 left-1/2 -translate-x-1/2">
          <DMPlaceholder />
        </div>

        {/* Left menu */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-auto">
          <MenuButtons />
        </div>

        {/* Bottom bar */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-auto">
          <PartyListBar />
        </div>
        <div className="absolute bottom-4 right-4 pointer-events-auto">
          <Minimap />
        </div>

        {/* Dialogue/interaction controls - contextual */}
        {(selectedAgentId || isInDialogue) && (
          <div className="absolute bottom-24 left-1/2 -translate-x-1/2 pointer-events-auto">
            {selectedAgentId && !isInDialogue && (
              <button
                onClick={handleTalkToAgent}
                disabled={dialogueLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 shadow-lg"
              >
                Talk (T)
              </button>
            )}
            {isInDialogue && (
              <div className="flex items-center gap-3 bg-gray-900/90 backdrop-blur-sm border border-gray-700 rounded-lg px-4 py-2">
                <span className="text-sm text-gray-400">
                  {isWaitingForResponse ? 'Agent is thinking...' : 'In conversation'}
                </span>
                <button
                  onClick={handleEndDialogue}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
                >
                  End (Esc)
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Contextual overlays */}
      <AgentInfoPanel />
      <TileInfoPanel />

      {/* Modal panels */}
      <SettingsPanel isOpen={openPanel === 'settings'} onClose={closePanel} />
      <JournalPanel isOpen={openPanel === 'journal'} onClose={closePanel} />
      <QuestsPanel isOpen={openPanel === 'quests'} onClose={closePanel} />
      <InventoryPanel isOpen={openPanel === 'inventory'} onClose={closePanel} />
      <CharacterSheet isOpen={openPanel === 'character'} onClose={closePanel} />
    </div>
  );
}
