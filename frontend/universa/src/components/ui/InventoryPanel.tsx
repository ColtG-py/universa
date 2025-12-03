'use client';

/**
 * InventoryPanel Component
 * Modal panel for managing inventory items.
 */

import { useState, useMemo } from 'react';
import { useGameStore } from '@/stores/gameStore';
import type { InventoryItem } from '@/types/game';

interface InventoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

type ItemCategory = InventoryItem['type'] | 'all';

const categoryLabels: Record<ItemCategory, string> = {
  all: 'All Items',
  weapon: 'Weapons',
  armor: 'Armor',
  consumable: 'Consumables',
  quest: 'Quest Items',
  misc: 'Miscellaneous',
};

const categoryIcons: Record<ItemCategory, string> = {
  all: '',
  weapon: '',
  armor: '',
  consumable: '',
  quest: '',
  misc: '',
};

export default function InventoryPanel({ isOpen, onClose }: InventoryPanelProps) {
  const localPlayer = useGameStore((state) => state.localPlayer);
  const [selectedCategory, setSelectedCategory] = useState<ItemCategory>('all');
  const [selectedItem, setSelectedItem] = useState<InventoryItem | null>(null);

  const inventory = localPlayer?.inventory ?? [];

  const filteredItems = useMemo(() => {
    if (selectedCategory === 'all') return inventory;
    return inventory.filter((item) => item.type === selectedCategory);
  }, [inventory, selectedCategory]);

  const categoryCounts = useMemo(() => {
    const counts: Record<ItemCategory, number> = {
      all: inventory.length,
      weapon: 0,
      armor: 0,
      consumable: 0,
      quest: 0,
      misc: 0,
    };
    inventory.forEach((item) => {
      counts[item.type]++;
    });
    return counts;
  }, [inventory]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span></span> Inventory
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            x
          </button>
        </div>

        {/* Category tabs */}
        <div className="flex border-b border-gray-700 overflow-x-auto">
          {(Object.keys(categoryLabels) as ItemCategory[]).map((category) => (
            <button
              key={category}
              onClick={() => {
                setSelectedCategory(category);
                setSelectedItem(null);
              }}
              className={`px-3 py-2 text-sm font-medium transition-colors whitespace-nowrap ${
                selectedCategory === category
                  ? 'text-white bg-gray-800 border-b-2 border-amber-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <span className="mr-1">{categoryIcons[category]}</span>
              {categoryLabels[category]}
              <span className="ml-1 text-xs text-gray-500">({categoryCounts[category]})</span>
            </button>
          ))}
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Item grid */}
          <div className="flex-1 overflow-y-auto p-4">
            {filteredItems.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <p className="text-3xl mb-2"></p>
                  <p className="text-sm">No items in this category</p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-2">
                {filteredItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setSelectedItem(item)}
                    className={`
                      p-3 rounded-lg text-center transition-all
                      ${selectedItem?.id === item.id
                        ? 'bg-amber-900/40 border-2 border-amber-500'
                        : 'bg-gray-800 border border-gray-700 hover:bg-gray-700 hover:border-gray-600'
                      }
                    `}
                  >
                    <div className="text-2xl mb-1">{categoryIcons[item.type]}</div>
                    <div className="text-xs text-white truncate">{item.name}</div>
                    {item.quantity > 1 && (
                      <div className="text-xs text-gray-500">x{item.quantity}</div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Item detail sidebar */}
          <div className="w-64 border-l border-gray-700 p-4 bg-gray-800/50">
            {selectedItem ? (
              <div>
                <div className="text-center mb-4">
                  <div className="text-4xl mb-2">{categoryIcons[selectedItem.type]}</div>
                  <h3 className="text-lg font-bold text-amber-400">{selectedItem.name}</h3>
                  <div className="text-xs text-gray-500 uppercase mt-1">
                    {selectedItem.type}
                  </div>
                </div>

                <p className="text-sm text-gray-300 mb-4 leading-relaxed">
                  {selectedItem.description}
                </p>

                {selectedItem.quantity > 1 && (
                  <div className="text-sm text-gray-400 mb-4">
                    Quantity: <span className="text-white">{selectedItem.quantity}</span>
                  </div>
                )}

                {/* Action buttons */}
                <div className="space-y-2">
                  {selectedItem.type === 'consumable' && (
                    <button className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition-colors">
                      Use Item
                    </button>
                  )}
                  {(selectedItem.type === 'weapon' || selectedItem.type === 'armor') && (
                    <button className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors">
                      Equip
                    </button>
                  )}
                  {selectedItem.type !== 'quest' && (
                    <button className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors">
                      Drop
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <p className="text-2xl mb-2"></p>
                  <p className="text-xs">Select an item to view details</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex justify-between items-center">
          <div className="text-sm text-gray-500">
            {inventory.length} items total
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
