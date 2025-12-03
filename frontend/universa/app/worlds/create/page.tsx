'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useWorldCreation } from '@/hooks/useWorlds';
import type { CreateWorldParams } from '@/lib/api';

const WORLD_SIZES = [
  { value: 'SMALL', label: 'Small', desc: '512x512 tiles', estimate: '~2 min' },
  { value: 'MEDIUM', label: 'Medium', desc: '1024x1024 tiles', estimate: '~5 min' },
  { value: 'LARGE', label: 'Large', desc: '2048x2048 tiles', estimate: '~15 min' },
  { value: 'HUGE', label: 'Huge', desc: '4096x4096 tiles', estimate: '~45 min' },
] as const;

const SETTLEMENT_DENSITY = [
  { value: 'low', label: 'Sparse', desc: 'Few settlements, vast wilderness' },
  { value: 'medium', label: 'Normal', desc: 'Balanced distribution' },
  { value: 'high', label: 'Dense', desc: 'Many towns and cities' },
] as const;

export default function CreateWorldPage() {
  const router = useRouter();
  const { createWorld, isCreating, progress, world, error, reset } = useWorldCreation();

  // Form state
  const [name, setName] = useState('');
  const [seed, setSeed] = useState('');
  const [size, setSize] = useState<'SMALL' | 'MEDIUM' | 'LARGE' | 'HUGE'>('MEDIUM');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [oceanPercentage, setOceanPercentage] = useState(40);
  const [numPlates, setNumPlates] = useState(12);
  const [enableMagic, setEnableMagic] = useState(true);
  const [enableCaves, setEnableCaves] = useState(true);
  const [settlementDensity, setSettlementDensity] = useState<'low' | 'medium' | 'high'>('medium');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    const params: CreateWorldParams = {
      name: name.trim(),
      size,
      settlement_density: settlementDensity,
      enable_magic: enableMagic,
      enable_caves: enableCaves,
      ocean_percentage: oceanPercentage / 100, // Convert from 0-100 to 0-1
      num_plates: numPlates,
    };

    if (seed.trim()) {
      params.seed = parseInt(seed, 10) || Math.floor(Math.random() * 1000000);
    }

    try {
      await createWorld(params);
    } catch {
      // Error is handled in the hook
    }
  };

  const handleContinue = () => {
    if (world) {
      router.push(`/game/setup?worldId=${world.world_id}`);
    }
  };

  // Generation in progress
  if (isCreating || progress) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="max-w-md w-full mx-4">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-bold mb-4 text-center">
              {progress?.status === 'ready' ? 'World Created!' : 'Generating World'}
            </h2>

            {/* Progress bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Pass {progress?.pass_number || 0} of {progress?.total_passes || 18}</span>
                <span>{Math.round(progress?.progress_percent || 0)}%</span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-amber-600 to-amber-400 transition-all duration-300"
                  style={{ width: `${progress?.progress_percent || 0}%` }}
                />
              </div>
            </div>

            {/* Current pass */}
            <div className="text-center mb-6">
              <div className="text-gray-400 text-sm">Current Pass</div>
              <div className="text-lg font-medium text-amber-400">
                {progress?.current_pass || 'Initializing...'}
              </div>
            </div>

            {/* Actions */}
            {progress?.status === 'ready' || world?.status === 'ready' ? (
              <div className="flex gap-3">
                <button
                  onClick={handleContinue}
                  className="flex-1 px-4 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
                >
                  Start Game
                </button>
                <Link
                  href={`/worlds/${world?.world_id}`}
                  className="px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
                >
                  View World
                </Link>
              </div>
            ) : (
              <p className="text-center text-gray-400 text-sm">
                This may take a few minutes. You can close this page and check back later.
              </p>
            )}

            {/* Error */}
            {error && (
              <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm">
                {error}
              </div>
            )}
          </div>

          <div className="mt-4 text-center">
            <Link href="/worlds" className="text-gray-400 hover:text-white transition-colors">
              Back to Worlds
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-2xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link href="/worlds" className="text-gray-400 hover:text-white transition-colors">
            ← Back
          </Link>
          <h1 className="text-xl font-semibold">Create New World</h1>
        </div>
      </header>

      {/* Form */}
      <main className="max-w-2xl mx-auto px-4 py-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* World Name */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium mb-2">
              World Name <span className="text-red-400">*</span>
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter a name for your world"
              className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg focus:border-amber-500 focus:outline-none transition-colors"
              required
            />
          </div>

          {/* Seed */}
          <div>
            <label htmlFor="seed" className="block text-sm font-medium mb-2">
              World Seed <span className="text-gray-500">(optional)</span>
            </label>
            <input
              id="seed"
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="Leave empty for random"
              className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg focus:border-amber-500 focus:outline-none transition-colors"
            />
            <p className="mt-1 text-sm text-gray-500">Use the same seed to recreate the same world</p>
          </div>

          {/* World Size */}
          <div>
            <label className="block text-sm font-medium mb-3">World Size</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {WORLD_SIZES.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setSize(option.value)}
                  className={`p-3 rounded-lg border text-left transition-colors ${
                    size === option.value
                      ? 'border-amber-500 bg-amber-500/10'
                      : 'border-gray-700 bg-gray-900 hover:border-gray-600'
                  }`}
                >
                  <div className="font-medium">{option.label}</div>
                  <div className="text-xs text-gray-400">{option.desc}</div>
                  <div className="text-xs text-gray-500 mt-1">{option.estimate}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            <span className={`transform transition-transform ${showAdvanced ? 'rotate-90' : ''}`}>▶</span>
            Advanced Options
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-6 p-4 bg-gray-900/50 border border-gray-800 rounded-lg">
              {/* Ocean Percentage */}
              <div>
                <label className="flex justify-between text-sm font-medium mb-2">
                  <span>Ocean Coverage</span>
                  <span className="text-gray-400">{oceanPercentage}%</span>
                </label>
                <input
                  type="range"
                  min="20"
                  max="80"
                  value={oceanPercentage}
                  onChange={(e) => setOceanPercentage(parseInt(e.target.value, 10))}
                  className="w-full accent-amber-500"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>More Land</span>
                  <span>More Water</span>
                </div>
              </div>

              {/* Tectonic Plates */}
              <div>
                <label className="flex justify-between text-sm font-medium mb-2">
                  <span>Tectonic Plates</span>
                  <span className="text-gray-400">{numPlates}</span>
                </label>
                <input
                  type="range"
                  min="4"
                  max="24"
                  value={numPlates}
                  onChange={(e) => setNumPlates(parseInt(e.target.value, 10))}
                  className="w-full accent-amber-500"
                />
                <p className="text-xs text-gray-500 mt-1">More plates = more varied terrain</p>
              </div>

              {/* Settlement Density */}
              <div>
                <label className="block text-sm font-medium mb-3">Settlement Density</label>
                <div className="grid grid-cols-3 gap-3">
                  {SETTLEMENT_DENSITY.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setSettlementDensity(option.value)}
                      className={`p-3 rounded-lg border text-center transition-colors ${
                        settlementDensity === option.value
                          ? 'border-amber-500 bg-amber-500/10'
                          : 'border-gray-700 bg-gray-800 hover:border-gray-600'
                      }`}
                    >
                      <div className="font-medium text-sm">{option.label}</div>
                      <div className="text-xs text-gray-400 mt-1">{option.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Feature Toggles */}
              <div className="flex gap-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableMagic}
                    onChange={(e) => setEnableMagic(e.target.checked)}
                    className="w-4 h-4 accent-amber-500"
                  />
                  <span>Enable Magic System</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableCaves}
                    onChange={(e) => setEnableCaves(e.target.checked)}
                    className="w-4 h-4 accent-amber-500"
                  />
                  <span>Generate Caves</span>
                </label>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400">
              {error}
            </div>
          )}

          {/* Submit */}
          <div className="flex gap-4">
            <button
              type="submit"
              disabled={!name.trim()}
              className="flex-1 px-6 py-3 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-700 disabled:text-gray-400 rounded-lg font-medium transition-colors"
            >
              Create World
            </button>
            <Link
              href="/worlds"
              className="px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg font-medium transition-colors"
            >
              Cancel
            </Link>
          </div>
        </form>
      </main>
    </div>
  );
}
