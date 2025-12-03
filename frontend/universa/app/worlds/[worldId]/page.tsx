'use client';

import { use } from 'react';
import Link from 'next/link';
import { useWorld } from '@/hooks/useWorlds';

export default function WorldDetailPage({ params }: { params: Promise<{ worldId: string }> }) {
  const { worldId } = use(params);
  const { world, settlements, isLoading, error } = useWorld(worldId);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-amber-400"></div>
        <span className="ml-3 text-gray-400">Loading world...</span>
      </div>
    );
  }

  if (error || !world) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">😞</div>
          <h2 className="text-2xl font-semibold mb-2">World Not Found</h2>
          <p className="text-gray-400 mb-6">{error || 'This world does not exist or has been deleted.'}</p>
          <Link
            href="/worlds"
            className="inline-block px-6 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-medium transition-colors"
          >
            Back to Worlds
          </Link>
        </div>
      </div>
    );
  }

  const formatDate = (dateStr: string | undefined) => {
    if (!dateStr) return 'Unknown';
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/worlds" className="text-gray-400 hover:text-white transition-colors">
              ← Back
            </Link>
            <h1 className="text-xl font-semibold">{world.name}</h1>
            {world.status === 'ready' ? (
              <span className="px-2 py-1 bg-green-600/20 text-green-400 text-xs rounded-full">Ready</span>
            ) : world.status === 'generating' ? (
              <span className="px-2 py-1 bg-amber-600/20 text-amber-400 text-xs rounded-full animate-pulse">Generating</span>
            ) : (
              <span className="px-2 py-1 bg-red-600/20 text-red-400 text-xs rounded-full">Failed</span>
            )}
          </div>
          {world.status === 'ready' && (
            <Link
              href={`/game/setup?worldId=${world.world_id}`}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
            >
              Play This World
            </Link>
          )}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* World Info */}
          <div className="lg:col-span-2 space-y-6">
            {/* Stats Card */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">World Information</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-2xl font-bold text-amber-400">{world.width || '?'}</div>
                  <div className="text-sm text-gray-400">Width (tiles)</div>
                </div>
                <div className="p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-2xl font-bold text-amber-400">{world.height || '?'}</div>
                  <div className="text-sm text-gray-400">Height (tiles)</div>
                </div>
                <div className="p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-400">{settlements.length}</div>
                  <div className="text-sm text-gray-400">Settlements</div>
                </div>
                <div className="p-4 bg-gray-800/50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-400">{world.seed || 'Random'}</div>
                  <div className="text-sm text-gray-400">Seed</div>
                </div>
              </div>
              <div className="mt-4 text-sm text-gray-400">
                Created: {formatDate(world.created_at)}
              </div>
            </div>

            {/* Settlements */}
            {settlements.length > 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
                <h2 className="text-lg font-semibold mb-4">Settlements ({settlements.length})</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {settlements.slice(0, 10).map((settlement) => (
                    <div
                      key={settlement.id}
                      className="p-3 bg-gray-800/50 rounded-lg flex items-start justify-between"
                    >
                      <div>
                        <div className="font-medium">{settlement.name}</div>
                        <div className="text-sm text-gray-400">
                          {settlement.settlement_type} • Pop: {settlement.population}
                        </div>
                      </div>
                      <div className="text-xs text-gray-500">
                        ({settlement.x}, {settlement.y})
                      </div>
                    </div>
                  ))}
                </div>
                {settlements.length > 10 && (
                  <div className="mt-3 text-sm text-gray-400 text-center">
                    +{settlements.length - 10} more settlements
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Actions */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Actions</h2>
              <div className="space-y-3">
                {world.status === 'ready' && (
                  <>
                    <Link
                      href={`/game/setup?worldId=${world.world_id}`}
                      className="block w-full px-4 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium text-center transition-colors"
                    >
                      Start New Game
                    </Link>
                    <button
                      disabled
                      className="block w-full px-4 py-3 bg-gray-700 text-gray-400 rounded-lg font-medium text-center cursor-not-allowed"
                    >
                      Continue Game (Coming Soon)
                    </button>
                  </>
                )}
                <Link
                  href="/worlds"
                  className="block w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg font-medium text-center transition-colors"
                >
                  Back to Worlds
                </Link>
              </div>
            </div>

            {/* Quick Info */}
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Quick Info</h2>
              <dl className="space-y-3 text-sm">
                <div>
                  <dt className="text-gray-400">World ID</dt>
                  <dd className="font-mono text-xs break-all">{world.world_id}</dd>
                </div>
                <div>
                  <dt className="text-gray-400">Status</dt>
                  <dd className="capitalize">{world.status}</dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
