'use client';

/**
 * TileInfoPanel Component
 * Shows information about the currently hovered tile.
 */

import { useGameStore, selectTileAt } from '@/stores/gameStore';

export default function TileInfoPanel() {
  const hoveredTile = useGameStore((state) => state.hoveredTile);
  const tile = useGameStore((state) =>
    hoveredTile ? state.tiles.get(`${hoveredTile.x},${hoveredTile.y}`) : null
  );

  if (!hoveredTile) return null;

  const formatBiome = (biome: string): string => {
    return biome
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="absolute bottom-4 left-4 bg-gray-900/95 border border-gray-700 rounded-lg px-3 py-2 text-sm">
      <div className="flex items-center gap-4">
        {/* Coordinates */}
        <div>
          <span className="text-gray-400">Pos: </span>
          <span className="text-white font-mono">
            {hoveredTile.x}, {hoveredTile.y}
          </span>
        </div>

        {tile && (
          <>
            {/* Biome */}
            <div>
              <span className="text-gray-400">Biome: </span>
              <span className="text-green-400">{formatBiome(tile.biomeType)}</span>
            </div>

            {/* Elevation */}
            <div>
              <span className="text-gray-400">Elev: </span>
              <span className="text-blue-400">{tile.elevation.toFixed(0)}m</span>
            </div>

            {/* Temperature */}
            <div>
              <span className="text-gray-400">Temp: </span>
              <span className="text-amber-400">{tile.temperatureC.toFixed(1)}°C</span>
            </div>

            {/* Features */}
            {(tile.hasRoad || tile.hasRiver || tile.settlementId) && (
              <div className="flex gap-2">
                {tile.hasRoad && <span title="Road">🛤️</span>}
                {tile.hasRiver && <span title="River">🌊</span>}
                {tile.settlementId && (
                  <span title={`${tile.settlementType}: ${tile.factionName}`}>
                    🏘️
                  </span>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
