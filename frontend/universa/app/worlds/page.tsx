'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useWorldList } from '@/hooks/useWorlds';
import type { SessionListItem } from '@/lib/api';

export default function WorldsPage() {
  const { worlds, sessions, isLoading, error, refresh, deleteWorld } = useWorldList();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Get the most recent active session for a world
  const getActiveSession = (worldId: string): SessionListItem | null => {
    const worldSessions = sessions.get(worldId);
    if (!worldSessions || worldSessions.length === 0) return null;
    // Return most recent (first in the list, already sorted by created_at desc)
    return worldSessions[0];
  };

  const handleDelete = async (worldId: string, worldName: string) => {
    if (!confirm(`Are you sure you want to delete "${worldName}"? This cannot be undone.`)) {
      return;
    }
    setDeletingId(worldId);
    try {
      await deleteWorld(worldId);
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return 'Unknown';
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready':
        return <span className="px-2 py-1 bg-green-600/20 text-green-400 text-xs rounded-full">Ready</span>;
      case 'generating':
        return <span className="px-2 py-1 bg-amber-600/20 text-amber-400 text-xs rounded-full animate-pulse">Generating</span>;
      case 'failed':
        return <span className="px-2 py-1 bg-red-600/20 text-red-400 text-xs rounded-full">Failed</span>;
      default:
        return <span className="px-2 py-1 bg-gray-600/20 text-gray-400 text-xs rounded-full">{status}</span>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-2xl font-bold text-amber-400 hover:text-amber-300 transition-colors">
              Universa
            </Link>
            <span className="text-gray-600">/</span>
            <h1 className="text-xl font-semibold">Worlds</h1>
          </div>
          <Link
            href="/worlds/create"
            className="px-4 py-2 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            <span>+</span>
            Create World
          </Link>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400">
            {error}
            <button
              onClick={() => refresh()}
              className="ml-4 underline hover:no-underline"
            >
              Retry
            </button>
          </div>
        )}

        {/* Loading state */}
        {isLoading && worlds.length === 0 && (
          <div className="flex items-center justify-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-amber-400"></div>
            <span className="ml-3 text-gray-400">Loading worlds...</span>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && worlds.length === 0 && !error && (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">🌍</div>
            <h2 className="text-2xl font-semibold mb-2">No Worlds Yet</h2>
            <p className="text-gray-400 mb-6">Create your first world to start exploring!</p>
            <Link
              href="/worlds/create"
              className="inline-block px-6 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
            >
              Create Your First World
            </Link>
          </div>
        )}

        {/* World list */}
        {worlds.length > 0 && (
          <div className="grid gap-4">
            {worlds.map((world) => {
              const activeSession = getActiveSession(world.world_id);
              return (
                <div
                  key={world.world_id}
                  className="p-4 bg-gray-900 border border-gray-800 rounded-lg hover:border-gray-700 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-semibold">{world.name}</h3>
                        {getStatusBadge(world.status)}
                      </div>
                      <div className="text-sm text-gray-400 space-y-1">
                        <div>Seed: {world.seed || 'Random'}</div>
                        <div>Size: {world.width || '?'} x {world.height || '?'}</div>
                        <div>Created: {formatDate(world.created_at)}</div>
                      </div>

                      {/* Active Session Info */}
                      {activeSession && (
                        <div className="mt-3 p-2 bg-blue-900/20 border border-blue-800/50 rounded-lg">
                          <div className="flex items-center gap-2 text-sm">
                            <span className="text-blue-400 font-medium">Active Save:</span>
                            <span className="text-gray-300">{activeSession.player_name}</span>
                            <span className="text-gray-500">•</span>
                            <span className="text-gray-400">{activeSession.game_time}</span>
                            <span className="text-gray-500">•</span>
                            <span className="text-gray-400">Tick {activeSession.current_tick}</span>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      {world.status === 'ready' && (
                        <>
                          {activeSession ? (
                            <>
                              <Link
                                href={`/game/${activeSession.id}`}
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
                              >
                                Resume
                              </Link>
                              <Link
                                href={`/game/setup?worldId=${world.world_id}`}
                                className="px-4 py-2 bg-green-600/80 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors"
                              >
                                New Game
                              </Link>
                            </>
                          ) : (
                            <Link
                              href={`/game/setup?worldId=${world.world_id}`}
                              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors"
                            >
                              Play
                            </Link>
                          )}
                          <Link
                            href={`/worlds/${world.world_id}`}
                            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors"
                          >
                            View
                          </Link>
                        </>
                      )}
                      {world.status === 'generating' && (
                        <Link
                          href={`/worlds/${world.world_id}`}
                          className="px-4 py-2 bg-amber-600 hover:bg-amber-700 rounded-lg text-sm font-medium transition-colors"
                        >
                          View Progress
                        </Link>
                      )}
                      <button
                        onClick={() => handleDelete(world.world_id, world.name)}
                        disabled={deletingId === world.world_id}
                        className="px-3 py-2 bg-red-600/20 hover:bg-red-600/40 text-red-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                      >
                        {deletingId === world.world_id ? '...' : 'Delete'}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Refresh button */}
        {worlds.length > 0 && (
          <div className="mt-6 text-center">
            <button
              onClick={() => refresh()}
              disabled={isLoading}
              className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            >
              {isLoading ? 'Refreshing...' : 'Refresh List'}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
