'use client';

/**
 * GameView Component
 * Main game view combining canvas, chat, and UI panels.
 */

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useGameStore } from '@/stores/gameStore';
import ChatPanel from '@/components/ui/ChatPanel';
import CharacterSheet from '@/components/ui/CharacterSheet';
import AgentInfoPanel from '@/components/ui/AgentInfoPanel';
import TileInfoPanel from '@/components/ui/TileInfoPanel';
import type { ChatMessage } from '@/types/game';

// Dynamic import for GameCanvas to avoid SSR issues with PixiJS
const GameCanvas = dynamic(() => import('./GameCanvas'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-gray-900">
      <div className="text-white">Loading game...</div>
    </div>
  ),
});

interface GameViewProps {
  worldId: string;
}

export default function GameView({ worldId }: GameViewProps) {
  const [isCharacterSheetOpen, setCharacterSheetOpen] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  const localPlayer = useGameStore((state) => state.localPlayer);
  const addMessage = useGameStore((state) => state.addMessage);
  const inputMode = useGameStore((state) => state.inputMode);

  // Handle window resize
  useEffect(() => {
    const updateSize = () => {
      // Account for chat panel width (320px) and some padding
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

      // TODO: Broadcast to Supabase realtime
    },
    [localPlayer, addMessage]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in input
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA'
      ) {
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'c':
          setCharacterSheetOpen((prev) => !prev);
          break;
        case 'escape':
          setCharacterSheetOpen(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="flex h-screen bg-gray-950">
      {/* Game canvas area */}
      <div className="flex-1 relative">
        <GameCanvas width={canvasSize.width} height={canvasSize.height} />

        {/* Overlay UI */}
        <AgentInfoPanel />
        <TileInfoPanel />

        {/* Mode indicator */}
        <div className="absolute top-4 left-4 bg-gray-900/80 border border-gray-700 rounded px-3 py-1">
          <span className="text-xs text-gray-400">Mode: </span>
          <span className="text-sm text-amber-400 capitalize">{inputMode}</span>
        </div>

        {/* Quick actions bar */}
        <div className="absolute bottom-4 right-4 flex gap-2">
          <button
            onClick={() => setCharacterSheetOpen(true)}
            className="px-3 py-2 bg-gray-800 border border-gray-600 rounded text-white text-sm hover:bg-gray-700 transition-colors"
            title="Character Sheet (C)"
          >
            📋 Character
          </button>
        </div>
      </div>

      {/* Chat panel */}
      <div className="w-80 border-l border-gray-800">
        <ChatPanel onSendMessage={handleSendMessage} />
      </div>

      {/* Character sheet modal */}
      <CharacterSheet
        isOpen={isCharacterSheetOpen}
        onClose={() => setCharacterSheetOpen(false)}
      />
    </div>
  );
}
