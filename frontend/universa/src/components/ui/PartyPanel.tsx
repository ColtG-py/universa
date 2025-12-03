'use client';

/**
 * PartyPanel Component
 * Shows party members and allows interaction with them.
 */

import { useGameStore } from '@/stores/gameStore';
import { useMemo } from 'react';

interface PartyMember {
  id: string;
  name: string;
  role?: string;
  x: number;
  y: number;
  health?: number;
  energy?: number;
  status: string;
  currentAction?: string | null;
}

interface PartyPanelProps {
  onTalkTo?: (memberId: string) => void;
  onSelectMember?: (memberId: string) => void;
}

export default function PartyPanel({ onTalkTo, onSelectMember }: PartyPanelProps) {
  const agents = useGameStore(state => state.agents);
  const localPlayer = useGameStore(state => state.localPlayer);
  const selectedAgentId = useGameStore(state => state.selectedAgentId);
  const selectAgent = useGameStore(state => state.selectAgent);

  // Filter agents to only show party members (those with role/occupation containing party-related keywords)
  const partyMembers = useMemo(() => {
    const members: PartyMember[] = [];
    agents.forEach((agent) => {
      // Check if agent is a party member based on traits or proximity
      const isPartyMember =
        agent.traits?.occupation?.toLowerCase().includes('party') ||
        agent.traits?.occupation?.toLowerCase().includes('warrior') ||
        agent.traits?.occupation?.toLowerCase().includes('mage') ||
        agent.traits?.occupation?.toLowerCase().includes('healer') ||
        agent.traits?.occupation?.toLowerCase().includes('rogue') ||
        agent.traits?.occupation?.toLowerCase().includes('ranger') ||
        agent.traits?.occupation?.toLowerCase().includes('companion');

      if (isPartyMember) {
        members.push({
          id: agent.id,
          name: agent.name,
          role: agent.traits?.occupation || 'Companion',
          x: agent.x,
          y: agent.y,
          status: agent.status,
          currentAction: agent.currentAction,
        });
      }
    });
    return members;
  }, [agents]);

  const handleSelect = (memberId: string) => {
    selectAgent(memberId);
    onSelectMember?.(memberId);
  };

  const handleTalk = (memberId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onTalkTo?.(memberId);
  };

  const getRoleIcon = (role: string) => {
    const r = role.toLowerCase();
    if (r.includes('warrior')) return '⚔️';
    if (r.includes('mage')) return '🔮';
    if (r.includes('healer')) return '💚';
    if (r.includes('rogue')) return '🗡️';
    if (r.includes('ranger')) return '🏹';
    return '👤';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'idle': return 'bg-blue-500';
      case 'sleeping': return 'bg-purple-500';
      case 'dead': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="bg-gray-900/95 border border-gray-700 rounded-lg p-3">
      <h3 className="text-sm font-semibold text-amber-400 mb-2 flex items-center gap-2">
        <span>👥</span>
        Party ({partyMembers.length})
      </h3>

      {partyMembers.length === 0 ? (
        <p className="text-gray-500 text-xs">No party members nearby</p>
      ) : (
        <div className="space-y-2">
          {partyMembers.map((member) => (
            <div
              key={member.id}
              onClick={() => handleSelect(member.id)}
              className={`
                p-2 rounded cursor-pointer transition-colors
                ${selectedAgentId === member.id
                  ? 'bg-amber-900/50 border border-amber-600'
                  : 'bg-gray-800 hover:bg-gray-700 border border-transparent'}
              `}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{getRoleIcon(member.role || '')}</span>
                  <div>
                    <div className="text-sm font-medium text-white">{member.name}</div>
                    <div className="text-xs text-gray-400">{member.role}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(member.status)}`}
                       title={member.status} />
                  <button
                    onClick={(e) => handleTalk(member.id, e)}
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors"
                    title="Talk to this party member"
                  >
                    💬
                  </button>
                </div>
              </div>
              {member.currentAction && (
                <div className="mt-1 text-xs text-gray-500 italic">
                  {member.currentAction}
                </div>
              )}
              <div className="mt-1 text-xs text-gray-500">
                Position: ({member.x}, {member.y})
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Player info */}
      {localPlayer && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-400 mb-1">You</div>
          <div className="flex items-center gap-2">
            <span className="text-lg">🎮</span>
            <div>
              <div className="text-sm font-medium text-amber-400">{localPlayer.name}</div>
              <div className="text-xs text-gray-400">
                ({localPlayer.x}, {localPlayer.y})
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
