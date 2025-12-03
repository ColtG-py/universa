'use client';

/**
 * Minimap Component
 * Circular minimap showing surrounding area in the bottom-right of the HUD.
 * Click to open full map view.
 */

import { useState, useMemo } from 'react';
import { useGameStore } from '@/stores/gameStore';
import FullMapView from './FullMapView';

const MINIMAP_RADIUS = 10; // Tile radius to show
const MINIMAP_SIZE = 96; // Pixel size of minimap
const TILE_SIZE = MINIMAP_SIZE / (MINIMAP_RADIUS * 2 + 1);

// Simplified biome colors for minimap
const biomeColors: Record<string, string> = {
  // Water
  ocean_deep: '#0a2d4c',
  ocean: '#1a4a6e',
  ocean_shallow: '#2a6a8e',
  lake: '#3a8aae',
  river: '#4a90d9',
  // Land
  beach: '#e8d5a3',
  desert: '#edc9af',
  desert_dunes: '#d4a574',
  grassland: '#7cba5f',
  savanna: '#a8b55f',
  shrubland: '#8a9f5f',
  forest_temperate: '#228b22',
  forest_tropical: '#1a7a40',
  forest_boreal: '#2d5a3d',
  jungle: '#1a6a30',
  wetland: '#5a7a5a',
  swamp: '#4a5a3a',
  tundra: '#a8b8a8',
  snow: '#f0f0f0',
  ice: '#e0f0ff',
  mountain: '#6b6b6b',
  mountain_snow: '#a0a0a0',
  mountain_peak: '#ffffff',
  volcanic: '#4a3030',
  // Default
  default: '#5a6a5a',
};

export default function Minimap() {
  const tiles = useGameStore((state) => state.tiles);
  const localPlayer = useGameStore((state) => state.localPlayer);
  const partySlots = useGameStore((state) => state.partySlots);
  const agents = useGameStore((state) => state.agents);

  const [isFullMapOpen, setIsFullMapOpen] = useState(false);

  const playerPos = localPlayer ? { x: localPlayer.x, y: localPlayer.y } : { x: 0, y: 0 };

  // Get visible tiles for minimap
  const minimapTiles = useMemo(() => {
    const result: { x: number; y: number; color: string; hasRoad: boolean; hasRiver: boolean }[] = [];

    for (let dy = -MINIMAP_RADIUS; dy <= MINIMAP_RADIUS; dy++) {
      for (let dx = -MINIMAP_RADIUS; dx <= MINIMAP_RADIUS; dx++) {
        const worldX = playerPos.x + dx;
        const worldY = playerPos.y + dy;
        const tile = tiles.get(`${worldX},${worldY}`);

        if (tile) {
          result.push({
            x: dx + MINIMAP_RADIUS,
            y: dy + MINIMAP_RADIUS,
            color: biomeColors[tile.biomeType] || biomeColors.default,
            hasRoad: tile.hasRoad,
            hasRiver: tile.hasRiver,
          });
        } else {
          // Unexplored/unloaded area
          result.push({
            x: dx + MINIMAP_RADIUS,
            y: dy + MINIMAP_RADIUS,
            color: '#1a1a2e',
            hasRoad: false,
            hasRiver: false,
          });
        }
      }
    }

    return result;
  }, [tiles, playerPos.x, playerPos.y]);

  // Get party member positions relative to player
  const partyPositions = useMemo(() => {
    const positions: { x: number; y: number; id: string }[] = [];

    partySlots.forEach((slot) => {
      if (!slot) return;
      const agent = agents.get(slot.agentId);
      if (!agent) return;

      const relX = agent.x - playerPos.x;
      const relY = agent.y - playerPos.y;

      // Only show if within minimap radius
      if (Math.abs(relX) <= MINIMAP_RADIUS && Math.abs(relY) <= MINIMAP_RADIUS) {
        positions.push({
          x: relX + MINIMAP_RADIUS,
          y: relY + MINIMAP_RADIUS,
          id: agent.id,
        });
      }
    });

    return positions;
  }, [partySlots, agents, playerPos.x, playerPos.y]);

  return (
    <>
      <div
        className="relative rounded-full overflow-hidden shadow-lg border-2 border-gray-700 bg-gray-900 cursor-pointer hover:border-amber-500 transition-colors"
        style={{ width: MINIMAP_SIZE, height: MINIMAP_SIZE }}
        onClick={() => setIsFullMapOpen(true)}
        title="Click to open full map (M)"
      >
        {/* Tile layer */}
        <svg
          viewBox={`0 0 ${MINIMAP_RADIUS * 2 + 1} ${MINIMAP_RADIUS * 2 + 1}`}
          className="absolute inset-0 w-full h-full"
        >
          {minimapTiles.map((tile, i) => (
            <rect
              key={i}
              x={tile.x}
              y={tile.y}
              width={1}
              height={1}
              fill={tile.color}
            />
          ))}

          {/* Roads overlay */}
          {minimapTiles
            .filter((t) => t.hasRoad)
            .map((tile, i) => (
              <rect
                key={`road-${i}`}
                x={tile.x + 0.3}
                y={tile.y + 0.3}
                width={0.4}
                height={0.4}
                fill="#8b7355"
              />
            ))}

          {/* Party members */}
          {partyPositions.map((pos) => (
            <circle
              key={pos.id}
              cx={pos.x + 0.5}
              cy={pos.y + 0.5}
              r={0.4}
              fill="#60a5fa"
              stroke="#1e3a5f"
              strokeWidth={0.1}
            />
          ))}

          {/* Player indicator (always center) */}
          <circle
            cx={MINIMAP_RADIUS + 0.5}
            cy={MINIMAP_RADIUS + 0.5}
            r={0.5}
            fill="#fbbf24"
            stroke="#92400e"
            strokeWidth={0.15}
          />
        </svg>

        {/* North indicator */}
        <div className="absolute top-1 left-1/2 -translate-x-1/2">
          <span className="text-[10px] font-bold text-white drop-shadow-lg">N</span>
        </div>

        {/* Circular mask overlay (to soften edges) */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(circle, transparent 60%, rgba(17,24,39,0.8) 100%)',
          }}
        />
      </div>

      {/* Full Map Modal */}
      <FullMapView isOpen={isFullMapOpen} onClose={() => setIsFullMapOpen(false)} />
    </>
  );
}
