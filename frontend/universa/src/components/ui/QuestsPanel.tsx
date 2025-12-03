'use client';

/**
 * QuestsPanel Component
 * Modal panel for viewing active and completed quests.
 */

import { useState, useMemo } from 'react';
import { useGameStore, selectActiveQuests, selectCompletedQuests } from '@/stores/gameStore';
import type { Quest } from '@/types/game';

interface QuestsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

type QuestTab = 'active' | 'completed';

export default function QuestsPanel({ isOpen, onClose }: QuestsPanelProps) {
  const activeQuests = useGameStore(selectActiveQuests);
  const completedQuests = useGameStore(selectCompletedQuests);
  const [activeTab, setActiveTab] = useState<QuestTab>('active');
  const [selectedQuest, setSelectedQuest] = useState<Quest | null>(null);

  const displayedQuests = activeTab === 'active' ? activeQuests : completedQuests;

  if (!isOpen) return null;

  const getProgressPercentage = (quest: Quest): number => {
    if (quest.objectives.length === 0) return 0;
    const completed = quest.objectives.filter((o) => o.isCompleted).length;
    return Math.round((completed / quest.objectives.length) * 100);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span></span> Quests
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            x
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => {
              setActiveTab('active');
              setSelectedQuest(null);
            }}
            className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'active'
                ? 'text-white bg-gray-800 border-b-2 border-amber-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            Active ({activeQuests.length})
          </button>
          <button
            onClick={() => {
              setActiveTab('completed');
              setSelectedQuest(null);
            }}
            className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'completed'
                ? 'text-white bg-gray-800 border-b-2 border-green-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            Completed ({completedQuests.length})
          </button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Quest list */}
          <div className="w-2/5 border-r border-gray-700 overflow-y-auto">
            {displayedQuests.length === 0 ? (
              <div className="p-4 text-center text-gray-500">
                <p className="text-2xl mb-2">{activeTab === 'active' ? '' : ''}</p>
                <p className="text-sm">
                  {activeTab === 'active' ? 'No active quests' : 'No completed quests'}
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-800">
                {displayedQuests.map((quest) => {
                  const progress = getProgressPercentage(quest);
                  return (
                    <button
                      key={quest.id}
                      onClick={() => setSelectedQuest(quest)}
                      className={`w-full p-3 text-left transition-colors ${
                        selectedQuest?.id === quest.id
                          ? 'bg-amber-900/30 border-l-2 border-amber-500'
                          : 'hover:bg-gray-800'
                      }`}
                    >
                      <div className="font-medium text-white text-sm truncate">
                        {quest.title}
                      </div>
                      {quest.giverName && (
                        <div className="text-xs text-gray-500 mt-0.5">
                          From: {quest.giverName}
                        </div>
                      )}
                      {activeTab === 'active' && (
                        <div className="mt-2">
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-500">Progress</span>
                            <span className="text-amber-400">{progress}%</span>
                          </div>
                          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-500 transition-all"
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Quest detail */}
          <div className="flex-1 overflow-y-auto p-4">
            {selectedQuest ? (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {selectedQuest.status === 'completed' ? (
                    <span className="text-green-500 text-lg"></span>
                  ) : (
                    <span className="text-amber-500 text-lg"></span>
                  )}
                  <span
                    className={`text-xs uppercase ${
                      selectedQuest.status === 'completed'
                        ? 'text-green-500'
                        : 'text-amber-500'
                    }`}
                  >
                    {selectedQuest.status}
                  </span>
                </div>

                <h3 className="text-xl font-bold text-amber-400 mb-2">
                  {selectedQuest.title}
                </h3>

                {selectedQuest.giverName && (
                  <div className="text-sm text-gray-500 mb-3">
                    Quest Giver: <span className="text-gray-300">{selectedQuest.giverName}</span>
                  </div>
                )}

                <p className="text-gray-300 mb-4 leading-relaxed">
                  {selectedQuest.description}
                </p>

                {/* Objectives */}
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-gray-400 uppercase mb-2">
                    Objectives
                  </h4>
                  <div className="space-y-2">
                    {selectedQuest.objectives.map((objective) => (
                      <div
                        key={objective.id}
                        className={`flex items-start gap-2 p-2 rounded ${
                          objective.isCompleted
                            ? 'bg-green-900/20'
                            : 'bg-gray-800'
                        }`}
                      >
                        <span
                          className={`mt-0.5 ${
                            objective.isCompleted
                              ? 'text-green-500'
                              : 'text-gray-600'
                          }`}
                        >
                          {objective.isCompleted ? '' : ''}
                        </span>
                        <div className="flex-1">
                          <span
                            className={`text-sm ${
                              objective.isCompleted
                                ? 'text-green-400 line-through'
                                : 'text-gray-200'
                            }`}
                          >
                            {objective.description}
                          </span>
                          {objective.target && (
                            <div className="text-xs text-gray-500 mt-0.5">
                              {objective.current ?? 0} / {objective.target}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Reward */}
                {selectedQuest.reward && (
                  <div className="bg-amber-900/20 border border-amber-800 rounded p-3">
                    <h4 className="text-sm font-semibold text-amber-400 mb-1">
                      Reward
                    </h4>
                    <p className="text-sm text-gray-300">{selectedQuest.reward}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <p className="text-3xl mb-2"></p>
                  <p className="text-sm">Select a quest to view details</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex justify-end">
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
