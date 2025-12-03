'use client';

/**
 * MenuButtons Component
 * Vertical stack of menu buttons on the left side of the HUD.
 */

import { useGameStore } from '@/stores/gameStore';
import type { UIPanel } from '@/types/game';

interface MenuButton {
  id: NonNullable<UIPanel>;
  label: string;
  icon: string;
  shortcut: string;
}

const menuButtons: MenuButton[] = [
  { id: 'settings', label: 'Settings', icon: '', shortcut: 'ESC' },
  { id: 'journal', label: 'Journal', icon: '', shortcut: 'J' },
  { id: 'quests', label: 'Quests', icon: '', shortcut: 'Q' },
  { id: 'inventory', label: 'Inventory', icon: '', shortcut: 'I' },
];

export default function MenuButtons() {
  const openPanel = useGameStore((state) => state.openPanel);
  const togglePanel = useGameStore((state) => state.togglePanel);

  return (
    <div className="flex flex-col gap-2">
      {menuButtons.map((button) => {
        const isActive = openPanel === button.id;
        return (
          <button
            key={button.id}
            onClick={() => togglePanel(button.id)}
            className={`
              w-12 h-12 rounded-lg flex items-center justify-center
              transition-all duration-150 group relative
              ${isActive
                ? 'bg-amber-600 border-2 border-amber-400 shadow-lg shadow-amber-600/30'
                : 'bg-gray-900/80 backdrop-blur-sm border border-gray-700 hover:bg-gray-800 hover:border-gray-600'
              }
            `}
            title={`${button.label} (${button.shortcut})`}
          >
            <span className="text-xl">{button.icon}</span>

            {/* Tooltip */}
            <div className="absolute left-full ml-2 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm text-white whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
              {button.label}
              <span className="ml-2 text-gray-500 text-xs">{button.shortcut}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
