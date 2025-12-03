'use client';

/**
 * JournalPanel Component
 * Modal panel for viewing journal entries.
 */

import { useState, useMemo } from 'react';
import { useGameStore } from '@/stores/gameStore';
import type { JournalEntry } from '@/types/game';

interface JournalPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

type JournalCategory = JournalEntry['category'] | 'all';

const categoryLabels: Record<JournalCategory, string> = {
  all: 'All Entries',
  story: 'Story',
  notes: 'Notes',
  discoveries: 'Discoveries',
};

const categoryIcons: Record<JournalCategory, string> = {
  all: '',
  story: '',
  notes: '',
  discoveries: '',
};

export default function JournalPanel({ isOpen, onClose }: JournalPanelProps) {
  const journalEntries = useGameStore((state) => state.journalEntries);
  const gameTime = useGameStore((state) => state.gameTime);
  const [selectedCategory, setSelectedCategory] = useState<JournalCategory>('all');
  const [selectedEntry, setSelectedEntry] = useState<JournalEntry | null>(null);

  const filteredEntries = useMemo(() => {
    if (selectedCategory === 'all') return journalEntries;
    return journalEntries.filter((entry) => entry.category === selectedCategory);
  }, [journalEntries, selectedCategory]);

  if (!isOpen) return null;

  const formatDate = (day: number, timestamp: number) => {
    const time = new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
    return `Day ${day}, ${time}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span></span> Journal
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            x
          </button>
        </div>

        {/* Category tabs */}
        <div className="flex border-b border-gray-700">
          {(Object.keys(categoryLabels) as JournalCategory[]).map((category) => (
            <button
              key={category}
              onClick={() => {
                setSelectedCategory(category);
                setSelectedEntry(null);
              }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                selectedCategory === category
                  ? 'text-white bg-gray-800 border-b-2 border-amber-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <span className="mr-1">{categoryIcons[category]}</span>
              {categoryLabels[category]}
            </button>
          ))}
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Entry list */}
          <div className="w-1/3 border-r border-gray-700 overflow-y-auto">
            {filteredEntries.length === 0 ? (
              <div className="p-4 text-center text-gray-500">
                <p className="text-2xl mb-2"></p>
                <p className="text-sm">No entries yet</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-800">
                {filteredEntries.map((entry) => (
                  <button
                    key={entry.id}
                    onClick={() => setSelectedEntry(entry)}
                    className={`w-full p-3 text-left transition-colors ${
                      selectedEntry?.id === entry.id
                        ? 'bg-amber-900/30 border-l-2 border-amber-500'
                        : 'hover:bg-gray-800'
                    }`}
                  >
                    <div className="font-medium text-white text-sm truncate">
                      {entry.title}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {formatDate(entry.day, entry.timestamp)}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Entry detail */}
          <div className="flex-1 overflow-y-auto p-4">
            {selectedEntry ? (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-lg">{categoryIcons[selectedEntry.category]}</span>
                  <span className="text-xs text-gray-500 uppercase">
                    {selectedEntry.category}
                  </span>
                </div>
                <h3 className="text-xl font-bold text-amber-400 mb-2">
                  {selectedEntry.title}
                </h3>
                <div className="text-xs text-gray-500 mb-4">
                  {formatDate(selectedEntry.day, selectedEntry.timestamp)}
                </div>
                <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {selectedEntry.content}
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <p className="text-3xl mb-2"></p>
                  <p className="text-sm">Select an entry to read</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex justify-between items-center">
          <div className="text-sm text-gray-500">
            Day {gameTime.day} - {filteredEntries.length} entries
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
