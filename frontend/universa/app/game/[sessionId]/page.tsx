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
import ChatPanel from '@/components/ui/ChatPanel';
import CharacterSheet from '@/components/ui/CharacterSheet';
import AgentInfoPanel from '@/components/ui/AgentInfoPanel';
import TileInfoPanel from '@/components/ui/TileInfoPanel';
import PartyPanel from '@/components/ui/PartyPanel';
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

  const [isCharacterSheetOpen, setCharacterSheetOpen] = useState(false);
  const [isDebugOpen, setDebugOpen] = useState(false);
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
  const setWorld = useGameStore(state => state.setWorld);
  const setInputMode = useGameStore(state => state.setInputMode);
  const selectAgent = useGameStore(state => state.selectAgent);
  const setLocalPlayer = useGameStore(state => state.setLocalPlayer);
  const setViewport = useGameStore(state => state.setViewport);

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
          setLocalPlayer({
            id: session.player.id,
            name: session.player.name,
            x: session.player.x,
            y: session.player.y,
            health: session.player.health,
            maxHealth: session.player.max_health,
            stamina: session.player.stamina,
            maxStamina: session.player.max_stamina,
            stats: session.player.stats || {},
          });

          // Center viewport on player position
          // Player at (px, py) should be in center of screen
          // So viewport top-left = (px * TILE_SIZE - width/2, py * TILE_SIZE - height/2)
          const TILE_SIZE = 32;
          setViewport({
            x: session.player.x * TILE_SIZE - canvasSize.width / 2,
            y: session.player.y * TILE_SIZE - canvasSize.height / 2,
          });

          // Load initial chunks around player
          const bounds = {
            xMin: session.player.x - 30,
            xMax: session.player.x + 30,
            yMin: session.player.y - 30,
            yMax: session.player.y + 30,
          };
          console.log('Loading chunks with bounds:', bounds);
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
            console.log(`Setting ${tiles.length} tiles`);
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

  // Handle window resize
  useEffect(() => {
    const updateSize = () => {
      const chatPanelWidth = 320;
      const padding = 32;
      setCanvasSize({
        width: Math.max(600, window.innerWidth - chatPanelWidth - padding),
        height: Math.max(400, window.innerHeight - 100),
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

    // Load new chunks if player moved significantly
    const chunkThreshold = 10; // tiles
    const needsNewChunks =
      Math.abs(newX - (viewport.x / TILE_SIZE + canvasSize.width / (2 * TILE_SIZE))) > chunkThreshold ||
      Math.abs(newY - (viewport.y / TILE_SIZE + canvasSize.height / (2 * TILE_SIZE))) > chunkThreshold;

    if (needsNewChunks) {
      const bounds = {
        xMin: newX - 30,
        xMax: newX + 30,
        yMin: newY - 30,
        yMax: newY + 30,
      };
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
          setTiles(tiles);
        }
      } catch (err) {
        console.error('Failed to load chunks:', err);
      }
    }

    // Note: Don't call refreshAgents() on every move - it clears party members
    // Party members are updated via tick, nearby world agents can be loaded on demand
  }, [localPlayer, sessionId, sessionData, updateLocalPlayerPosition, setViewport, canvasSize, viewport, setTiles]);

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
          if (e.shiftKey) {
            // Shift+D for debug
            setDebugOpen(prev => !prev);
          } else {
            // D for move right
            e.preventDefault();
            handleMove(1, 0);
          }
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
          setCharacterSheetOpen(prev => !prev);
          break;
        case 'escape':
          setCharacterSheetOpen(false);
          setDebugOpen(false);
          if (isInDialogue) {
            handleEndDialogue();
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
  }, [tick, selectedAgentId, isInDialogue, handleTalkToAgent, handleEndDialogue, handleMove]);

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
    <div className="flex h-screen bg-gray-950">
      {/* Game canvas area */}
      <div className="flex-1 relative">
        <GameCanvas width={canvasSize.width} height={canvasSize.height} />

        {/* Overlay UI */}
        <AgentInfoPanel />
        <TileInfoPanel />

        {/* Top bar */}
        <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
          {/* Left side - Status */}
          <div className="flex items-center gap-3 pointer-events-auto">
            <div className="bg-gray-900/90 border border-gray-700 rounded px-3 py-1">
              <span className="text-xs text-gray-400">Move: </span>
              <span className="text-sm text-white">WASD</span>
            </div>
            <div className="bg-gray-900/90 border border-gray-700 rounded px-3 py-1">
              <span className="text-xs text-gray-400">Mode: </span>
              <span className="text-sm text-amber-400 capitalize">{inputMode}</span>
            </div>
            <div className="bg-gray-900/90 border border-gray-700 rounded px-3 py-1">
              <span className="text-xs text-gray-400">Tick: </span>
              <span className="text-sm text-white">{tickCount || sessionData?.currentTick || 0}</span>
            </div>
            <div className="bg-gray-900/90 border border-gray-700 rounded px-3 py-1">
              <span className="text-xs text-gray-400">Time: </span>
              <span className="text-sm text-white">{gameTime || sessionData?.gameTime || '00:00'}</span>
            </div>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} title={isConnected ? 'Connected' : 'Disconnected'} />
          </div>

          {/* Right side - Controls */}
          <div className="flex items-center gap-2 pointer-events-auto">
            <button
              onClick={() => tick()}
              disabled={isProcessing}
              className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded text-sm disabled:opacity-50 transition-colors"
              title="Advance tick (Space)"
            >
              {isProcessing ? '...' : '▶ Tick'}
            </button>
            <Link
              href="/worlds"
              className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded text-sm transition-colors"
            >
              Exit
            </Link>
          </div>
        </div>

        {/* Bottom bar - Quick actions */}
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none">
          {/* Agent interaction */}
          {selectedAgentId && !isInDialogue && (
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                onClick={handleTalkToAgent}
                disabled={dialogueLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors disabled:opacity-50"
              >
                💬 Talk (T)
              </button>
            </div>
          )}

          {/* Dialogue controls */}
          {isInDialogue && (
            <div className="flex items-center gap-2 pointer-events-auto">
              <span className="text-sm text-gray-400">
                {isWaitingForResponse ? 'Agent is thinking...' : 'In conversation'}
              </span>
              <button
                onClick={handleEndDialogue}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
              >
                End Conversation (Esc)
              </button>
            </div>
          )}

          <div className="flex-1" />

          {/* Right side actions */}
          <div className="flex gap-2 pointer-events-auto">
            <button
              onClick={() => setDebugOpen(true)}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm hover:bg-gray-700 transition-colors"
              title="Debug Panel (D)"
            >
              🔧 Debug
            </button>
            <button
              onClick={() => setCharacterSheetOpen(true)}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm hover:bg-gray-700 transition-colors"
              title="Character Sheet (C)"
            >
              📋 Character
            </button>
          </div>
        </div>
      </div>

      {/* Right sidebar - Party & Chat */}
      <div className="w-80 border-l border-gray-800 flex flex-col">
        {/* Party panel */}
        <div className="p-2 border-b border-gray-800">
          <PartyPanel
            onTalkTo={handleTalkToPartyMember}
            onSelectMember={handleSelectPartyMember}
          />
        </div>

        {/* Chat panel */}
        <div className="flex-1 overflow-hidden">
          <ChatPanel
            onSendMessage={handleSendMessage}
            isDialogueMode={inputMode === 'dialogue'}
          />
        </div>
      </div>

      {/* Character sheet modal */}
      <CharacterSheet
        isOpen={isCharacterSheetOpen}
        onClose={() => setCharacterSheetOpen(false)}
      />

      {/* Debug panel modal */}
      {isDebugOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h2 className="text-lg font-semibold">Debug Information</h2>
              <button
                onClick={() => setDebugOpen(false)}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[calc(80vh-60px)]">
              <div className="space-y-4 text-sm">
                <div>
                  <h3 className="font-medium text-amber-400 mb-2">Session</h3>
                  <pre className="bg-gray-800 p-3 rounded overflow-x-auto">
                    {JSON.stringify({
                      sessionId,
                      worldId: sessionData?.worldId,
                      status: sessionData?.status,
                      tick: tickCount || sessionData?.currentTick || 0,
                      gameTime: gameTime || sessionData?.gameTime || '00:00',
                      wsConnected: isConnected,
                    }, null, 2)}
                  </pre>
                </div>
                <div>
                  <h3 className="font-medium text-amber-400 mb-2">Player</h3>
                  <pre className="bg-gray-800 p-3 rounded overflow-x-auto">
                    {JSON.stringify(localPlayer, null, 2)}
                  </pre>
                </div>
                <div>
                  <h3 className="font-medium text-amber-400 mb-2">Agents ({agents.size})</h3>
                  <pre className="bg-gray-800 p-3 rounded overflow-x-auto">
                    {JSON.stringify(Array.from(agents.values()).slice(0, 5), null, 2)}
                  </pre>
                </div>
                {conversation && (
                  <div>
                    <h3 className="font-medium text-amber-400 mb-2">Active Conversation</h3>
                    <pre className="bg-gray-800 p-3 rounded overflow-x-auto">
                      {JSON.stringify({
                        id: conversation.conversation_id,
                        state: conversation.state,
                        participants: conversation.participants,
                        turnCount: conversation.turns.length,
                      }, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
