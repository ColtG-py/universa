'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { useGameStore } from '@/stores/gameStore';

export default function Home() {
  const reset = useGameStore((state) => state.reset);

  // Reset game state on home page
  useEffect(() => {
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center min-h-screen px-4">
        <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
          Universa
        </h1>
        <p className="text-xl text-gray-400 mb-8 text-center max-w-2xl">
          A living world powered by AI agents. Explore procedurally generated lands,
          interact with intelligent NPCs, and embark on dynamic adventures.
        </p>

        <div className="flex gap-4">
          <Link
            href="/game/demo"
            className="px-8 py-3 bg-amber-600 hover:bg-amber-700 rounded-lg font-semibold transition-colors"
          >
            Play Demo
          </Link>
          <Link
            href="/worlds"
            className="px-8 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg font-semibold transition-colors"
          >
            Browse Worlds
          </Link>
        </div>

        {/* Feature cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 max-w-4xl">
          <div className="p-6 bg-gray-900 border border-gray-800 rounded-lg">
            <div className="text-3xl mb-3">🌍</div>
            <h3 className="text-lg font-semibold mb-2">Procedural Worlds</h3>
            <p className="text-gray-400 text-sm">
              Explore vast, procedurally generated worlds with diverse biomes,
              settlements, and factions.
            </p>
          </div>

          <div className="p-6 bg-gray-900 border border-gray-800 rounded-lg">
            <div className="text-3xl mb-3">🤖</div>
            <h3 className="text-lg font-semibold mb-2">AI Agents</h3>
            <p className="text-gray-400 text-sm">
              Interact with intelligent NPCs that have memories, goals, and
              realistic behaviors.
            </p>
          </div>

          <div className="p-6 bg-gray-900 border border-gray-800 rounded-lg">
            <div className="text-3xl mb-3">🎲</div>
            <h3 className="text-lg font-semibold mb-2">AI Dungeon Master</h3>
            <p className="text-gray-400 text-sm">
              Experience dynamic storytelling with an AI DM that adapts to your
              choices.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
