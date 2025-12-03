'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { useWorld } from '@/hooks/useWorlds';
import { useGameSession } from '@/hooks/useGameSession';
import type { Settlement } from '@/lib/api';

const PARTY_ROLES = [
  { value: 'warrior', label: 'Warrior', desc: 'Strong melee fighter' },
  { value: 'mage', label: 'Mage', desc: 'Arcane spellcaster' },
  { value: 'healer', label: 'Healer', desc: 'Support and healing' },
  { value: 'rogue', label: 'Rogue', desc: 'Stealth and agility' },
  { value: 'companion', label: 'Companion', desc: 'General helper' },
];

function GameSetupContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const worldId = searchParams.get('worldId');

  const { world, settlements, isLoading: worldLoading, error: worldError } = useWorld(worldId);
  const { createSession, isLoading: sessionLoading, error: sessionError } = useGameSession();

  // Player config
  const [playerName, setPlayerName] = useState('');
  const [spawnSettlement, setSpawnSettlement] = useState<Settlement | null>(null);

  // Party config
  const [partySize, setPartySize] = useState(2);
  const [partyRoles, setPartyRoles] = useState<string[]>(['warrior', 'healer']);

  // Game settings
  const [debugMode, setDebugMode] = useState(false);

  // Set default spawn location when settlements load
  useEffect(() => {
    if (settlements.length > 0 && !spawnSettlement) {
      // Prefer a village or town as starting location
      const preferred = settlements.find(s =>
        s.settlement_type === 'village' || s.settlement_type === 'town'
      ) || settlements[0];
      setSpawnSettlement(preferred);
    }
  }, [settlements, spawnSettlement]);

  const handlePartySizeChange = (size: number) => {
    setPartySize(size);
    // Adjust roles array
    if (size > partyRoles.length) {
      const newRoles = [...partyRoles];
      while (newRoles.length < size) {
        newRoles.push('companion');
      }
      setPartyRoles(newRoles);
    } else {
      setPartyRoles(partyRoles.slice(0, size));
    }
  };

  const handleRoleChange = (index: number, role: string) => {
    const newRoles = [...partyRoles];
    newRoles[index] = role;
    setPartyRoles(newRoles);
  };

  const handleStartGame = async () => {
    if (!worldId || !playerName.trim()) return;

    try {
      const session = await createSession(
        worldId,
        {
          name: playerName.trim(),
          // Default spawn to (32, 32) - center of available world data
          // Existing world chunks only cover tiles 0-63
          spawn_x: spawnSettlement?.x || 32,
          spawn_y: spawnSettlement?.y || 32,
        },
        {
          size: partySize,
          roles: partyRoles,
        },
        {
          debug_mode: debugMode,
        }
      );

      router.push(`/game/${session.session_id}`);
    } catch {
      // Error is handled in the hook
    }
  };

  if (!worldId) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">🤔</div>
          <h2 className="text-2xl font-semibold mb-2">No World Selected</h2>
          <p className="text-gray-400 mb-6">Please select a world first.</p>
          <Link
            href="/worlds"
            className="inline-block px-6 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
          >
            Browse Worlds
          </Link>
        </div>
      </div>
    );
  }

  if (worldLoading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-amber-400"></div>
        <span className="ml-3 text-gray-400">Loading world...</span>
      </div>
    );
  }

  if (worldError || !world) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">😞</div>
          <h2 className="text-2xl font-semibold mb-2">World Not Found</h2>
          <p className="text-gray-400 mb-6">{worldError || 'This world does not exist.'}</p>
          <Link
            href="/worlds"
            className="inline-block px-6 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
          >
            Browse Worlds
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link href={`/worlds/${worldId}`} className="text-gray-400 hover:text-white transition-colors">
            ← Back
          </Link>
          <h1 className="text-xl font-semibold">Game Setup</h1>
          <span className="text-gray-500">•</span>
          <span className="text-amber-400">{world.name}</span>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Player Config */}
          <div className="space-y-6">
            {/* Player Name */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Your Character</h2>
              <div className="space-y-4">
                <div>
                  <label htmlFor="playerName" className="block text-sm font-medium mb-2">
                    Character Name <span className="text-red-400">*</span>
                  </label>
                  <input
                    id="playerName"
                    type="text"
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value)}
                    placeholder="Enter your character's name"
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:border-amber-500 focus:outline-none transition-colors"
                    required
                  />
                </div>

                {/* Spawn Location */}
                <div>
                  <label className="block text-sm font-medium mb-2">Starting Location</label>
                  {settlements.length > 0 ? (
                    <select
                      value={spawnSettlement?.id || ''}
                      onChange={(e) => {
                        const s = settlements.find(s => s.id === e.target.value);
                        setSpawnSettlement(s || null);
                      }}
                      className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:border-amber-500 focus:outline-none transition-colors"
                    >
                      {settlements.map((settlement) => (
                        <option key={settlement.id} value={settlement.id}>
                          {settlement.name} ({settlement.settlement_type})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-gray-400">
                      Random wilderness spawn
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Party Configuration */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Party</h2>

              {/* Party Size */}
              <div className="mb-4">
                <label className="flex justify-between text-sm font-medium mb-2">
                  <span>Party Size</span>
                  <span className="text-gray-400">{partySize} companions</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="5"
                  value={partySize}
                  onChange={(e) => handlePartySizeChange(parseInt(e.target.value, 10))}
                  className="w-full accent-amber-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Solo</span>
                  <span>Full Party</span>
                </div>
              </div>

              {/* Party Roles */}
              {partySize > 0 && (
                <div className="space-y-3">
                  <label className="block text-sm font-medium">Companion Roles</label>
                  {Array.from({ length: partySize }, (_, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-gray-400 w-8">#{i + 1}</span>
                      <select
                        value={partyRoles[i] || 'companion'}
                        onChange={(e) => handleRoleChange(i, e.target.value)}
                        className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:border-amber-500 focus:outline-none transition-colors"
                      >
                        {PARTY_ROLES.map((role) => (
                          <option key={role.value} value={role.value}>
                            {role.label} - {role.desc}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Settings & Summary */}
          <div className="space-y-6">
            {/* Game Settings */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Settings</h2>
              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors">
                  <input
                    type="checkbox"
                    checked={debugMode}
                    onChange={(e) => setDebugMode(e.target.checked)}
                    className="w-4 h-4 accent-amber-500"
                  />
                  <div>
                    <div className="font-medium">Debug Mode</div>
                    <div className="text-sm text-gray-400">
                      View agent thinking, memories, and simulation stats
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Summary */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Summary</h2>
              <dl className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <dt className="text-gray-400">World</dt>
                  <dd className="font-medium">{world.name}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Character</dt>
                  <dd className="font-medium">{playerName || '(not set)'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Starting Location</dt>
                  <dd className="font-medium">{spawnSettlement?.name || 'Wilderness'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Party Size</dt>
                  <dd className="font-medium">{partySize} companions</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-400">Debug Mode</dt>
                  <dd className="font-medium">{debugMode ? 'Enabled' : 'Disabled'}</dd>
                </div>
              </dl>
            </div>

            {/* Errors */}
            {sessionError && (
              <div className="p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400">
                {sessionError}
              </div>
            )}

            {/* Start Button */}
            <button
              onClick={handleStartGame}
              disabled={!playerName.trim() || sessionLoading}
              className="w-full px-6 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-400 rounded-lg font-semibold text-lg transition-colors flex items-center justify-center gap-2"
            >
              {sessionLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                  Starting...
                </>
              ) : (
                <>
                  🎮 Start Adventure
                </>
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function GameSetupPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-amber-400"></div>
      </div>
    }>
      <GameSetupContent />
    </Suspense>
  );
}
