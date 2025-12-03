'use client';

/**
 * CharacterSheet Component
 * Displays player character stats, inventory, and abilities.
 */

import { useGameStore } from '@/stores/gameStore';
import type { PlayerCharacter, InventoryItem } from '@/types/game';

interface CharacterSheetProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CharacterSheet({ isOpen, onClose }: CharacterSheetProps) {
  const localPlayer = useGameStore((state) => state.localPlayer);

  if (!isOpen || !localPlayer) return null;

  const { stats, inventory, name } = localPlayer;

  const statLabels: Record<keyof typeof stats, string> = {
    health: 'HP',
    maxHealth: 'Max HP',
    mana: 'MP',
    maxMana: 'Max MP',
    strength: 'STR',
    dexterity: 'DEX',
    constitution: 'CON',
    intelligence: 'INT',
    wisdom: 'WIS',
    charisma: 'CHA',
  };

  const getStatModifier = (value: number): string => {
    const mod = Math.floor((value - 10) / 2);
    return mod >= 0 ? `+${mod}` : `${mod}`;
  };

  const getItemTypeIcon = (type: InventoryItem['type']): string => {
    switch (type) {
      case 'weapon':
        return '⚔️';
      case 'armor':
        return '🛡️';
      case 'consumable':
        return '🧪';
      case 'quest':
        return '📜';
      case 'misc':
        return '📦';
      default:
        return '•';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">{name}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(90vh-60px)]">
          {/* Vitals */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-amber-400 mb-3">Vitals</h3>
            <div className="grid grid-cols-2 gap-4">
              {/* Health bar */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Health</span>
                  <span className="text-white">
                    {stats.health} / {stats.maxHealth}
                  </span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 transition-all"
                    style={{ width: `${(stats.health / stats.maxHealth) * 100}%` }}
                  />
                </div>
              </div>

              {/* Mana bar */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Mana</span>
                  <span className="text-white">
                    {stats.mana} / {stats.maxMana}
                  </span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${(stats.mana / stats.maxMana) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </section>

          {/* Ability Scores */}
          <section className="mb-6">
            <h3 className="text-lg font-semibold text-amber-400 mb-3">
              Ability Scores
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {(
                [
                  'strength',
                  'dexterity',
                  'constitution',
                  'intelligence',
                  'wisdom',
                  'charisma',
                ] as const
              ).map((stat) => (
                <div
                  key={stat}
                  className="bg-gray-800 rounded-lg p-3 text-center"
                >
                  <div className="text-xs text-gray-400 uppercase">
                    {statLabels[stat]}
                  </div>
                  <div className="text-2xl font-bold text-white">{stats[stat]}</div>
                  <div className="text-sm text-amber-400">
                    {getStatModifier(stats[stat])}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Inventory */}
          <section>
            <h3 className="text-lg font-semibold text-amber-400 mb-3">Inventory</h3>
            {inventory.length === 0 ? (
              <p className="text-gray-500 italic">No items</p>
            ) : (
              <div className="space-y-2">
                {inventory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center gap-3 p-2 bg-gray-800 rounded hover:bg-gray-750 cursor-pointer transition-colors"
                  >
                    <span className="text-xl">{getItemTypeIcon(item.type)}</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">{item.name}</span>
                        {item.quantity > 1 && (
                          <span className="text-xs text-gray-400">
                            x{item.quantity}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500">{item.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
