'use client';

/**
 * FullMapView Component
 * Modal overlay showing the full world map with zoom/pan capabilities.
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useGameStore } from '@/stores/gameStore';

interface FullMapViewProps {
  isOpen: boolean;
  onClose: () => void;
}

// Biome colors for the map
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
  temperate_grassland: '#7cba5f',
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

export default function FullMapView({ isOpen, onClose }: FullMapViewProps) {
  const tiles = useGameStore((state) => state.tiles);
  const localPlayer = useGameStore((state) => state.localPlayer);
  const agents = useGameStore((state) => state.agents);
  const partySlots = useGameStore((state) => state.partySlots);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate map bounds from loaded tiles
  const mapBounds = useMemo(() => {
    if (tiles.size === 0) return { minX: 0, maxX: 100, minY: 0, maxY: 100 };

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    tiles.forEach((_, key) => {
      const [x, y] = key.split(',').map(Number);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });

    return { minX, maxX, minY, maxY };
  }, [tiles]);

  const mapWidth = mapBounds.maxX - mapBounds.minX + 1;
  const mapHeight = mapBounds.maxY - mapBounds.minY + 1;

  // Center on player when opened
  useEffect(() => {
    if (isOpen && localPlayer) {
      setPan({
        x: -(localPlayer.x - mapBounds.minX - mapWidth / 2),
        y: -(localPlayer.y - mapBounds.minY - mapHeight / 2),
      });
      setZoom(1);
    }
  }, [isOpen, localPlayer, mapBounds, mapWidth, mapHeight]);

  // Handle mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.max(0.2, Math.min(5, z + delta)));
  }, []);

  // Handle mouse drag for panning
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x * zoom, y: e.clientY - pan.y * zoom });
  }, [pan, zoom]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: (e.clientX - dragStart.x) / zoom,
      y: (e.clientY - dragStart.y) / zoom,
    });
  }, [isDragging, dragStart, zoom]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Get party member positions
  const partyPositions = useMemo(() => {
    const positions: { x: number; y: number; id: string; name: string }[] = [];
    partySlots.forEach((slot) => {
      if (!slot) return;
      const agent = agents.get(slot.agentId);
      if (!agent) return;
      positions.push({ x: agent.x, y: agent.y, id: agent.id, name: agent.name });
    });
    return positions;
  }, [partySlots, agents]);

  if (!isOpen) return null;

  // Create a canvas-like rendering using divs for each tile
  // This is efficient for loaded tiles only
  const tileSize = 4 * zoom;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-[90vw] h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-gray-700">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            World Map
          </h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span>Zoom: {Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom((z) => Math.max(0.2, z - 0.2))}
                className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
              >
                -
              </button>
              <button
                onClick={() => setZoom((z) => Math.min(5, z + 0.2))}
                className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
              >
                +
              </button>
              <button
                onClick={() => {
                  setZoom(1);
                  if (localPlayer) {
                    setPan({
                      x: -(localPlayer.x - mapBounds.minX - mapWidth / 2),
                      y: -(localPlayer.y - mapBounds.minY - mapHeight / 2),
                    });
                  }
                }}
                className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
              >
                Center
              </button>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors text-xl px-2"
            >
              x
            </button>
          </div>
        </div>

        {/* Map container */}
        <div
          ref={containerRef}
          className="flex-1 overflow-hidden cursor-grab active:cursor-grabbing relative"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <svg
            viewBox={`${-pan.x} ${-pan.y} ${mapWidth} ${mapHeight}`}
            className="w-full h-full"
            style={{
              transform: `scale(${zoom})`,
              transformOrigin: 'center center',
            }}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Render tiles */}
            {Array.from(tiles.entries()).map(([key, tile]) => {
              const [x, y] = key.split(',').map(Number);
              const color = biomeColors[tile.biomeType] || biomeColors.default;
              return (
                <rect
                  key={key}
                  x={x - mapBounds.minX}
                  y={y - mapBounds.minY}
                  width={1}
                  height={1}
                  fill={color}
                />
              );
            })}

            {/* Roads overlay */}
            {Array.from(tiles.entries())
              .filter(([_, tile]) => tile.hasRoad)
              .map(([key, tile]) => {
                const [x, y] = key.split(',').map(Number);
                return (
                  <rect
                    key={`road-${key}`}
                    x={x - mapBounds.minX + 0.25}
                    y={y - mapBounds.minY + 0.25}
                    width={0.5}
                    height={0.5}
                    fill="#8b7355"
                  />
                );
              })}

            {/* Settlements */}
            {Array.from(tiles.entries())
              .filter(([_, tile]) => tile.settlementId)
              .map(([key, tile]) => {
                const [x, y] = key.split(',').map(Number);
                return (
                  <circle
                    key={`settlement-${key}`}
                    cx={x - mapBounds.minX + 0.5}
                    cy={y - mapBounds.minY + 0.5}
                    r={0.4}
                    fill="#dc2626"
                    stroke="#7f1d1d"
                    strokeWidth={0.1}
                  />
                );
              })}

            {/* Party members */}
            {partyPositions.map((pos) => (
              <g key={pos.id}>
                <circle
                  cx={pos.x - mapBounds.minX + 0.5}
                  cy={pos.y - mapBounds.minY + 0.5}
                  r={0.8}
                  fill="#60a5fa"
                  stroke="#1e3a5f"
                  strokeWidth={0.15}
                />
              </g>
            ))}

            {/* Player indicator */}
            {localPlayer && (
              <g>
                <circle
                  cx={localPlayer.x - mapBounds.minX + 0.5}
                  cy={localPlayer.y - mapBounds.minY + 0.5}
                  r={1}
                  fill="#fbbf24"
                  stroke="#92400e"
                  strokeWidth={0.2}
                />
                <circle
                  cx={localPlayer.x - mapBounds.minX + 0.5}
                  cy={localPlayer.y - mapBounds.minY + 0.5}
                  r={1.5}
                  fill="none"
                  stroke="#fbbf24"
                  strokeWidth={0.1}
                  opacity={0.5}
                />
              </g>
            )}
          </svg>

          {/* Info overlay */}
          <div className="absolute bottom-4 left-4 bg-gray-900/80 p-2 rounded text-xs text-gray-400">
            <div>Loaded: {tiles.size} tiles</div>
            <div>Map area: {mapWidth} x {mapHeight}</div>
            {localPlayer && <div>Player: ({localPlayer.x}, {localPlayer.y})</div>}
          </div>
        </div>

        {/* Legend */}
        <div className="p-3 border-t border-gray-700 flex items-center gap-6 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-amber-400 border border-amber-700"></div>
            <span className="text-gray-400">You</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-400 border border-blue-800"></div>
            <span className="text-gray-400">Party</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-600 border border-red-900"></div>
            <span className="text-gray-400">Settlement</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-amber-700 rounded"></div>
            <span className="text-gray-400">Road</span>
          </div>
          <div className="ml-auto text-gray-500">
            Scroll to zoom | Drag to pan | Press M or click minimap to close
          </div>
        </div>
      </div>
    </div>
  );
}
