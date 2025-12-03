'use client';

/**
 * LocationDisplay Component
 * Shows the current location name in the top-left of the HUD.
 */

import { useGameStore } from '@/stores/gameStore';

const locationIcons: Record<string, string> = {
  settlement: '',
  wilderness: '',
  dungeon: '',
  road: '',
};

export default function LocationDisplay() {
  const currentLocation = useGameStore((state) => state.currentLocation);

  if (!currentLocation) {
    return null;
  }

  const icon = locationIcons[currentLocation.type] || '';

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-700 rounded-lg px-4 py-2 shadow-lg">
      <div className="flex items-center gap-2">
        <span className="text-lg">{icon}</span>
        <span className="text-amber-400 font-semibold">{currentLocation.name}</span>
      </div>
    </div>
  );
}
