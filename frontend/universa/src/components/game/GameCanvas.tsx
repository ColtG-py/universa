'use client';

/**
 * GameCanvas Component
 * Main PixiJS canvas for rendering the game world.
 * Uses @pixi/react for declarative rendering within React.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { Application, Container, Graphics, Sprite, Text, TextStyle } from 'pixi.js';
import { useGameStore, selectVisibleAgents } from '@/stores/gameStore';
import type { TileData, GameAgent } from '@/types/game';

// Tile size in pixels
const TILE_SIZE = 32;

// Biome colors (matching backend biome enum names)
const BIOME_COLORS: Record<string, number> = {
  // Ocean types
  ocean_trench: 0x0a2d4c,
  ocean_deep: 0x0e3d5c,
  ocean_shallow: 0x1a5276,
  ocean_shelf: 0x2874a6,
  coral_reef: 0x45b7d1,
  ocean: 0x1a5276,
  deep_ocean: 0x0e3d5c,

  // Cold biomes
  ice: 0xe8f4f8,
  tundra: 0xb0c4de,
  alpine: 0x9fa3a8,

  // Forests
  boreal_forest: 0x355e3b,
  temperate_forest: 0x228b22,
  temperate_rainforest: 0x1a7a40,
  tropical_rainforest: 0x0b6623,
  temperate_deciduous_forest: 0x2d6b2d,

  // Grasslands
  grassland: 0x7cba5f,
  temperate_grassland: 0x7cba5f,
  savanna: 0xc4a747,

  // Dry biomes
  desert: 0xedc9af,

  // Wetlands
  wetland: 0x4a7a5d,
  mangrove: 0x3d6b4d,
  swamp: 0x4a5d23,

  // Water features
  river: 0x4a90d9,
  lake: 0x3a7bc8,
  beach: 0xf5deb3,

  // Mountains
  mountain: 0x6b6b6b,
  snow_peak: 0xffffff,
};

interface GameCanvasProps {
  width?: number;
  height?: number;
}

export default function GameCanvas({ width = 800, height = 600 }: GameCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<Application | null>(null);
  const worldContainerRef = useRef<Container | null>(null);

  // Game state
  const tiles = useGameStore((state) => state.tiles);
  const agents = useGameStore(selectVisibleAgents);
  const viewport = useGameStore((state) => state.viewport);
  const localPlayer = useGameStore((state) => state.localPlayer);
  const selectedAgentId = useGameStore((state) => state.selectedAgentId);
  const setViewport = useGameStore((state) => state.setViewport);
  const selectAgent = useGameStore((state) => state.selectAgent);
  const setHoveredTile = useGameStore((state) => state.setHoveredTile);

  // Drag state for panning
  const [isDragging, setIsDragging] = useState(false);
  const [isPixiReady, setIsPixiReady] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const viewportStartRef = useRef({ x: 0, y: 0 });

  // Initialize PixiJS
  useEffect(() => {
    if (!canvasRef.current || appRef.current) return;

    // Track initialization to prevent double-init in StrictMode
    let isInitializing = false;
    let isCancelled = false;

    const initPixi = async () => {
      if (isInitializing || appRef.current) return;
      isInitializing = true;

      const app = new Application();
      await app.init({
        width,
        height,
        backgroundColor: 0x1a1a2e,
        antialias: true,
        resolution: window.devicePixelRatio || 1,
        autoDensity: true,
      });

      // Check if effect was cleaned up during async init
      if (isCancelled) {
        app.destroy(true, { children: true });
        return;
      }

      canvasRef.current?.appendChild(app.canvas as HTMLCanvasElement);
      appRef.current = app;

      // Create world container for camera/viewport control
      const worldContainer = new Container();
      worldContainer.sortableChildren = true;
      app.stage.addChild(worldContainer);
      worldContainerRef.current = worldContainer;

      // Set initial viewport size only - position will be set by the session page
      // based on player location
      setViewport({
        width,
        height,
      });

      // Signal that Pixi is ready - this triggers re-render for other effects
      setIsPixiReady(true);
    };

    initPixi();

    return () => {
      isCancelled = true;
      if (appRef.current) {
        appRef.current.destroy(true, { children: true });
        appRef.current = null;
      }
    };
  }, [width, height, setViewport]);

  // Render tiles
  useEffect(() => {
    if (!worldContainerRef.current || !isPixiReady) return;

    const worldContainer = worldContainerRef.current;

    // Clear existing tile graphics
    const existingTiles = worldContainer.children.filter(
      (child) => child.label === 'tile-layer'
    );
    existingTiles.forEach((child) => worldContainer.removeChild(child));

    // Create tile layer
    const tileLayer = new Container();
    tileLayer.label = 'tile-layer';
    tileLayer.zIndex = 0;

    // Draw visible tiles
    const startX = Math.floor(viewport.x / TILE_SIZE) - 1;
    const startY = Math.floor(viewport.y / TILE_SIZE) - 1;
    const endX = Math.ceil((viewport.x + viewport.width) / TILE_SIZE) + 1;
    const endY = Math.ceil((viewport.y + viewport.height) / TILE_SIZE) + 1;

    for (let x = startX; x <= endX; x++) {
      for (let y = startY; y <= endY; y++) {
        const tile = tiles.get(`${x},${y}`);
        const graphics = new Graphics();

        // Get color based on biome
        let color = BIOME_COLORS[tile?.biomeType || 'temperate_grassland'] || 0x7cba5f;

        // Adjust for elevation (clamp to reasonable range for color adjustment)
        if (tile) {
          // Clamp elevation factor to [-1, 1] range
          const elevationFactor = Math.max(-1, Math.min(tile.elevation / 500, 1));
          // Darken for lower elevation, lighten for higher
          if (tile.elevation < 0) {
            // Factor ranges from 0.5 (very deep) to 0.8 (near surface)
            color = adjustBrightness(color, 0.5 + (1 + elevationFactor) * 0.3);
          } else if (tile.elevation > 50) {
            color = adjustBrightness(color, 1 + elevationFactor * 0.2);
          }
        }

        graphics.rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
        graphics.fill(color);

        // Draw road overlay
        if (tile?.hasRoad) {
          graphics.rect(
            x * TILE_SIZE + TILE_SIZE * 0.3,
            y * TILE_SIZE,
            TILE_SIZE * 0.4,
            TILE_SIZE
          );
          graphics.fill(0x8b7355);
        }

        // Draw river overlay
        if (tile?.hasRiver) {
          graphics.rect(
            x * TILE_SIZE,
            y * TILE_SIZE + TILE_SIZE * 0.4,
            TILE_SIZE,
            TILE_SIZE * 0.2
          );
          graphics.fill(0x4a90d9);
        }

        // Draw settlement marker
        if (tile?.settlementId) {
          graphics.circle(
            x * TILE_SIZE + TILE_SIZE / 2,
            y * TILE_SIZE + TILE_SIZE / 2,
            TILE_SIZE * 0.3
          );
          graphics.fill(0xffd700);
          graphics.stroke({ width: 2, color: 0x8b4513 });
        }

        tileLayer.addChild(graphics);
      }
    }

    worldContainer.addChild(tileLayer);
  }, [tiles, viewport, isPixiReady]);

  // Render agents
  useEffect(() => {
    if (!worldContainerRef.current || !isPixiReady) return;

    const worldContainer = worldContainerRef.current;

    // Clear existing agent graphics
    const existingAgents = worldContainer.children.filter(
      (child) => child.label === 'agent-layer'
    );
    existingAgents.forEach((child) => worldContainer.removeChild(child));

    // Create agent layer
    const agentLayer = new Container();
    agentLayer.label = 'agent-layer';
    agentLayer.zIndex = 10;

    for (const agent of agents) {
      const agentContainer = new Container();
      agentContainer.x = agent.x * TILE_SIZE + TILE_SIZE / 2;
      agentContainer.y = agent.y * TILE_SIZE + TILE_SIZE / 2;

      // Agent body (circle for now, replace with sprite later)
      const body = new Graphics();
      body.circle(0, 0, TILE_SIZE * 0.4);

      // Color based on status
      const statusColors: Record<string, number> = {
        idle: 0x3498db,
        active: 0x2ecc71,
        sleeping: 0x9b59b6,
        dead: 0x7f8c8d,
      };
      body.fill(statusColors[agent.status] || 0x3498db);

      // Selection highlight
      if (agent.id === selectedAgentId) {
        body.stroke({ width: 3, color: 0xf1c40f });
      }

      agentContainer.addChild(body);

      // Agent name label
      const nameStyle = new TextStyle({
        fontSize: 10,
        fill: 0xffffff,
        fontFamily: 'Arial',
        stroke: { color: 0x000000, width: 2 },
      });
      const nameText = new Text({ text: agent.name, style: nameStyle });
      nameText.anchor.set(0.5, 0);
      nameText.y = TILE_SIZE * 0.5;
      agentContainer.addChild(nameText);

      // Make interactive
      agentContainer.eventMode = 'static';
      agentContainer.cursor = 'pointer';
      agentContainer.on('pointerdown', () => {
        selectAgent(agent.id);
      });

      agentLayer.addChild(agentContainer);
    }

    worldContainer.addChild(agentLayer);
  }, [agents, selectedAgentId, selectAgent, isPixiReady]);

  // Render local player
  useEffect(() => {
    if (!worldContainerRef.current || !localPlayer || !isPixiReady) return;

    const worldContainer = worldContainerRef.current;

    // Clear existing player graphics
    const existingPlayer = worldContainer.children.filter(
      (child) => child.label === 'player-layer'
    );
    existingPlayer.forEach((child) => worldContainer.removeChild(child));

    // Create player layer
    const playerLayer = new Container();
    playerLayer.label = 'player-layer';
    playerLayer.zIndex = 20;

    const playerContainer = new Container();
    playerContainer.x = localPlayer.x * TILE_SIZE + TILE_SIZE / 2;
    playerContainer.y = localPlayer.y * TILE_SIZE + TILE_SIZE / 2;

    // Player body (distinct from agents)
    const body = new Graphics();
    body.circle(0, 0, TILE_SIZE * 0.45);
    body.fill(0xe74c3c);
    body.stroke({ width: 3, color: 0xffd700 });

    playerContainer.addChild(body);

    // Player name
    const nameStyle = new TextStyle({
      fontSize: 12,
      fill: 0xffd700,
      fontFamily: 'Arial',
      fontWeight: 'bold',
      stroke: { color: 0x000000, width: 3 },
    });
    const nameText = new Text({ text: localPlayer.name, style: nameStyle });
    nameText.anchor.set(0.5, 0);
    nameText.y = TILE_SIZE * 0.5;
    playerContainer.addChild(nameText);

    playerLayer.addChild(playerContainer);
    worldContainer.addChild(playerLayer);
  }, [localPlayer, isPixiReady]);

  // Update viewport/camera position
  useEffect(() => {
    if (!worldContainerRef.current || !isPixiReady) return;
    worldContainerRef.current.x = -viewport.x;
    worldContainerRef.current.y = -viewport.y;
    worldContainerRef.current.scale.set(viewport.zoom);
  }, [viewport, isPixiReady]);

  // Mouse handlers for panning
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 0 || e.button === 1) {
        // Left or middle click
        setIsDragging(true);
        dragStartRef.current = { x: e.clientX, y: e.clientY };
        viewportStartRef.current = { x: viewport.x, y: viewport.y };
      }
    },
    [viewport]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      // Update hovered tile
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;
        const worldX = Math.floor((canvasX + viewport.x) / TILE_SIZE);
        const worldY = Math.floor((canvasY + viewport.y) / TILE_SIZE);
        setHoveredTile({ x: worldX, y: worldY });
      }

      // Handle dragging
      if (isDragging) {
        const dx = e.clientX - dragStartRef.current.x;
        const dy = e.clientY - dragStartRef.current.y;
        setViewport({
          x: viewportStartRef.current.x - dx,
          y: viewportStartRef.current.y - dy,
        });
      }
    },
    [isDragging, viewport, setViewport, setHoveredTile]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.25, Math.min(4, viewport.zoom * zoomFactor));
      setViewport({ zoom: newZoom });
    },
    [viewport.zoom, setViewport]
  );

  return (
    <div
      ref={canvasRef}
      className="cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      style={{ width, height }}
    />
  );
}

// Helper to adjust color brightness
function adjustBrightness(color: number, factor: number): number {
  const r = Math.min(255, Math.floor(((color >> 16) & 0xff) * factor));
  const g = Math.min(255, Math.floor(((color >> 8) & 0xff) * factor));
  const b = Math.min(255, Math.floor((color & 0xff) * factor));
  return (r << 16) | (g << 8) | b;
}
