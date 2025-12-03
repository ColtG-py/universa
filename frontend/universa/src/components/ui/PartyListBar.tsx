'use client';

/**
 * PartyListBar Component
 * Horizontal bar of party member portraits at the bottom of the HUD.
 */

import { useMemo } from 'react';
import { useGameStore } from '@/stores/gameStore';

export default function PartyListBar() {
  const partySlots = useGameStore((state) => state.partySlots);
  const agents = useGameStore((state) => state.agents);
  const selectedAgentId = useGameStore((state) => state.selectedAgentId);
  const selectAgent = useGameStore((state) => state.selectAgent);

  // Resolve party members from agent data
  const partyMembers = useMemo(() => {
    return partySlots.map((slot) => {
      if (!slot) return null;
      const agent = agents.get(slot.agentId);
      if (!agent) return null;
      return {
        ...slot,
        agent,
      };
    });
  }, [partySlots, agents]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500';
      case 'idle':
        return 'bg-blue-500';
      case 'sleeping':
        return 'bg-purple-500';
      case 'dead':
        return 'bg-red-900';
      default:
        return 'bg-gray-500';
    }
  };

  const getRoleIcon = (occupation: string) => {
    const role = occupation.toLowerCase();
    if (role.includes('warrior') || role.includes('fighter')) return '';
    if (role.includes('mage') || role.includes('wizard')) return '';
    if (role.includes('healer') || role.includes('cleric')) return '';
    if (role.includes('rogue') || role.includes('thief')) return '';
    if (role.includes('ranger') || role.includes('archer')) return '';
    return '';
  };

  return (
    <div className="flex gap-3 bg-gray-900/80 backdrop-blur-sm border border-gray-700 rounded-xl px-4 py-3">
      {partyMembers.map((member, index) => {
        const slotNumber = index + 1;
        const isSelected = member?.agent && selectedAgentId === member.agent.id;

        if (!member || !member.agent) {
          // Empty slot
          return (
            <div
              key={index}
              className="w-12 h-12 rounded-full border-2 border-dashed border-gray-700 flex items-center justify-center"
              title={`Party slot ${slotNumber} (empty)`}
            >
              <span className="text-gray-600 text-xs">{slotNumber}</span>
            </div>
          );
        }

        const { agent } = member;
        const roleIcon = getRoleIcon(agent.traits?.occupation || '');

        return (
          <button
            key={agent.id}
            onClick={() => selectAgent(isSelected ? null : agent.id)}
            className={`
              relative w-12 h-12 rounded-full transition-all duration-150 group
              ${isSelected
                ? 'ring-2 ring-amber-400 ring-offset-2 ring-offset-gray-900'
                : 'hover:ring-2 hover:ring-gray-500 hover:ring-offset-1 hover:ring-offset-gray-900'
              }
            `}
            title={`${agent.name} - ${agent.traits?.occupation || 'Companion'}`}
          >
            {/* Portrait background */}
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center text-2xl">
              {roleIcon}
            </div>

            {/* Status indicator */}
            <div
              className={`absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full border-2 border-gray-900 ${getStatusColor(agent.status)}`}
              title={agent.status}
            />

            {/* Slot number badge */}
            <div className="absolute -top-1 -left-1 w-4 h-4 rounded-full bg-gray-800 border border-gray-600 flex items-center justify-center">
              <span className="text-[10px] text-gray-400">{slotNumber}</span>
            </div>

            {/* Tooltip */}
            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-sm text-white whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
              <div className="font-medium">{agent.name}</div>
              <div className="text-xs text-gray-400">{agent.traits?.occupation}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
