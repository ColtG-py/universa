'use client';

/**
 * GameView Component
 * Main game view with fullscreen canvas and HUD overlay.
 */

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useGameStore } from '@/stores/gameStore';

// HUD Components
import LocationDisplay from '@/components/ui/LocationDisplay';
import TimeIndicator from '@/components/ui/TimeIndicator';
import MenuButtons from '@/components/ui/MenuButtons';
import PartyListBar from '@/components/ui/PartyListBar';
import Minimap from '@/components/ui/Minimap';
import DMPlaceholder from '@/components/ui/DMPlaceholder';

// Panel Components
import SettingsPanel from '@/components/ui/SettingsPanel';
import JournalPanel from '@/components/ui/JournalPanel';
import QuestsPanel from '@/components/ui/QuestsPanel';
import InventoryPanel from '@/components/ui/InventoryPanel';
import CharacterSheet from '@/components/ui/CharacterSheet';

// Overlay Components
import AgentInfoPanel from '@/components/ui/AgentInfoPanel';
import TileInfoPanel from '@/components/ui/TileInfoPanel';

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
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });

  const openPanel = useGameStore((state) => state.openPanel);
  const setOpenPanel = useGameStore((state) => state.setOpenPanel);
  const togglePanel = useGameStore((state) => state.togglePanel);

  // Handle window resize - now fullscreen
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

  // Close panel helper
  const closePanel = () => setOpenPanel(null);

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
          } else {
            togglePanel('settings');
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [openPanel, togglePanel]);

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
