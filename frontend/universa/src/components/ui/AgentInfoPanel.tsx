'use client';

/**
 * AgentInfoPanel Component
 * Shows information about a selected agent.
 */

import { useGameStore, selectAgentById } from '@/stores/gameStore';

export default function AgentInfoPanel() {
  const selectedAgentId = useGameStore((state) => state.selectedAgentId);
  const selectAgent = useGameStore((state) => state.selectAgent);

  const agent = useGameStore((state) =>
    selectedAgentId ? state.agents.get(selectedAgentId) : null
  );

  if (!agent) return null;

  const statusColors: Record<string, string> = {
    idle: 'bg-blue-500',
    active: 'bg-green-500',
    sleeping: 'bg-purple-500',
    dead: 'bg-gray-500',
  };

  return (
    <div className="absolute top-4 right-4 w-72 bg-gray-900/95 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${statusColors[agent.status] || 'bg-gray-500'}`}
          />
          <h3 className="font-semibold text-white">{agent.name}</h3>
        </div>
        <button
          onClick={() => selectAgent(null)}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div className="p-3 space-y-3">
        {/* Status */}
        <div>
          <div className="text-xs text-gray-400 uppercase mb-1">Status</div>
          <div className="text-sm text-white capitalize">{agent.status}</div>
        </div>

        {/* Current Action */}
        {agent.currentAction && (
          <div>
            <div className="text-xs text-gray-400 uppercase mb-1">Current Action</div>
            <div className="text-sm text-green-400">{agent.currentAction}</div>
          </div>
        )}

        {/* Location */}
        <div>
          <div className="text-xs text-gray-400 uppercase mb-1">Location</div>
          <div className="text-sm text-white">
            ({agent.x}, {agent.y})
          </div>
        </div>

        {/* Traits */}
        {agent.traits && (
          <>
            {agent.traits.occupation && (
              <div>
                <div className="text-xs text-gray-400 uppercase mb-1">Occupation</div>
                <div className="text-sm text-amber-400">{agent.traits.occupation}</div>
              </div>
            )}

            {agent.traits.personality && agent.traits.personality.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 uppercase mb-1">Personality</div>
                <div className="flex flex-wrap gap-1">
                  {agent.traits.personality.map((trait, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded"
                    >
                      {trait}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {agent.traits.goals && agent.traits.goals.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 uppercase mb-1">Goals</div>
                <ul className="text-sm text-gray-300 space-y-1">
                  {agent.traits.goals.map((goal, i) => (
                    <li key={i} className="flex items-start gap-1">
                      <span className="text-amber-500">•</span>
                      <span>{goal}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {/* Actions */}
        <div className="pt-2 border-t border-gray-700">
          <button className="w-full px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors">
            Talk to {agent.name}
          </button>
        </div>
      </div>
    </div>
  );
}
