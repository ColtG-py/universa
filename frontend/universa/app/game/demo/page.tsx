'use client';

/**
 * Demo Game Page
 * Loads a demo world with mock data for testing the visualization.
 */

import { useEffect } from 'react';
import { useGameStore } from '@/stores/gameStore';
import GameView from '@/components/game/GameView';
import type { TileData, GameAgent, PlayerCharacter, ChatMessage } from '@/types/game';

// Generate mock world tiles
function generateMockTiles(): TileData[] {
  const tiles: TileData[] = [];
  const biomes = [
    'temperate_grassland',
    'temperate_forest',
    'temperate_deciduous_forest',
    'beach',
    'ocean',
    'mountain',
    'desert',
    'savanna',
  ];

  for (let x = -50; x < 50; x++) {
    for (let y = -50; y < 50; y++) {
      // Simple noise-like biome selection based on position
      const noiseVal =
        Math.sin(x * 0.1) * Math.cos(y * 0.1) +
        Math.sin(x * 0.05 + y * 0.05) * 0.5;
      const biomeIndex = Math.floor(((noiseVal + 1.5) / 3) * biomes.length);
      const biome = biomes[Math.max(0, Math.min(biomes.length - 1, biomeIndex))];

      // Calculate elevation from distance to center
      const distFromCenter = Math.sqrt(x * x + y * y);
      const elevation = Math.max(0, 50 - distFromCenter * 0.5 + noiseVal * 20);

      // Add roads near settlements
      const hasRoad = (x === 0 || y === 0) && Math.abs(x) < 20 && Math.abs(y) < 20;

      // Add river
      const hasRiver = Math.abs(y - Math.sin(x * 0.2) * 5) < 1 && x > -30 && x < 30;

      // Settlement at center
      const isSettlement = Math.abs(x) < 3 && Math.abs(y) < 3;

      tiles.push({
        x,
        y,
        elevation,
        biomeType: biome,
        temperatureC: 15 + noiseVal * 10 - elevation * 0.1,
        hasRoad,
        hasRiver,
        settlementId: isSettlement ? 1 : null,
        settlementType: isSettlement ? 'village' : null,
        factionName: isSettlement ? 'Riverside' : null,
        resourceType: null,
      });
    }
  }

  return tiles;
}

// Generate mock agents
function generateMockAgents(): GameAgent[] {
  const names = [
    'Elena the Blacksmith',
    'Marcus the Innkeeper',
    'Sofia the Herbalist',
    'Thomas the Guard',
    'Isabella the Merchant',
    'William the Farmer',
    'Catherine the Baker',
    'Robert the Hunter',
  ];

  const occupations = [
    'Blacksmith',
    'Innkeeper',
    'Herbalist',
    'Guard',
    'Merchant',
    'Farmer',
    'Baker',
    'Hunter',
  ];

  const personalities = [
    ['friendly', 'hardworking'],
    ['jovial', 'talkative'],
    ['mysterious', 'knowledgeable'],
    ['stern', 'loyal'],
    ['shrewd', 'fair'],
    ['humble', 'patient'],
    ['cheerful', 'generous'],
    ['quiet', 'skilled'],
  ];

  return names.map((name, i) => ({
    id: `agent-${i}`,
    name,
    x: Math.floor(Math.random() * 10) - 5,
    y: Math.floor(Math.random() * 10) - 5,
    currentAction: i % 2 === 0 ? 'Working at their shop' : 'Walking around',
    status: 'active' as const,
    sprite: 'default',
    direction: 'down' as const,
    traits: {
      occupation: occupations[i],
      personality: personalities[i],
      goals: ['Earn a living', 'Help the community'],
    },
  }));
}

// Mock player character
const mockPlayer: PlayerCharacter = {
  id: 'player-1',
  name: 'Adventurer',
  x: 0,
  y: 0,
  isAgent: false,
  agentId: null,
  stats: {
    health: 85,
    maxHealth: 100,
    mana: 40,
    maxMana: 50,
    strength: 14,
    dexterity: 12,
    constitution: 13,
    intelligence: 10,
    wisdom: 11,
    charisma: 15,
  },
  inventory: [
    {
      id: 'item-1',
      name: 'Iron Sword',
      type: 'weapon',
      quantity: 1,
      description: 'A sturdy iron sword.',
    },
    {
      id: 'item-2',
      name: 'Health Potion',
      type: 'consumable',
      quantity: 3,
      description: 'Restores 25 HP.',
    },
    {
      id: 'item-3',
      name: 'Leather Armor',
      type: 'armor',
      quantity: 1,
      description: 'Basic protection.',
    },
  ],
};

// Welcome messages
const welcomeMessages: ChatMessage[] = [
  {
    id: 'msg-welcome-1',
    senderId: 'system',
    senderName: 'System',
    senderType: 'system',
    content: 'Welcome to Universa! This is a demo world.',
    timestamp: Date.now() - 5000,
    channel: 'global',
  },
  {
    id: 'msg-welcome-2',
    senderId: 'dm',
    senderName: 'Dungeon Master',
    senderType: 'dm',
    content:
      'You find yourself in a small village called Riverside. The morning sun casts long shadows across the cobblestone streets.',
    timestamp: Date.now() - 4000,
    channel: 'narration',
  },
  {
    id: 'msg-welcome-3',
    senderId: 'agent-0',
    senderName: 'Elena the Blacksmith',
    senderType: 'agent',
    content: 'Good morning, traveler! Looking for some new equipment?',
    timestamp: Date.now() - 3000,
    channel: 'global',
  },
];

export default function DemoGamePage() {
  const setWorld = useGameStore((state) => state.setWorld);
  const setTiles = useGameStore((state) => state.setTiles);
  const updateAgent = useGameStore((state) => state.updateAgent);
  const setLocalPlayer = useGameStore((state) => state.setLocalPlayer);
  const addMessage = useGameStore((state) => state.addMessage);
  const setViewport = useGameStore((state) => state.setViewport);

  // Initialize demo world
  useEffect(() => {
    // Set world info
    setWorld('demo-world', 'Demo Village');

    // Load tiles
    const tiles = generateMockTiles();
    setTiles(tiles);

    // Load agents
    const agents = generateMockAgents();
    agents.forEach((agent) => updateAgent(agent));

    // Set player
    setLocalPlayer(mockPlayer);

    // Don't set viewport position here - let it be set after canvas size is known
    // The GameCanvas will center properly when it initializes

    // Track if messages were already added to prevent duplicates in StrictMode
    return () => {
      // Cleanup handled by store reset if needed
    };
  }, []); // Empty deps - only run once on mount

  // Add welcome messages in separate effect with check to prevent duplicates
  useEffect(() => {
    const messages = useGameStore.getState().messages;
    // Only add welcome messages if they haven't been added yet
    if (messages.length === 0) {
      welcomeMessages.forEach((msg) => addMessage(msg));
    }
  }, [addMessage]);

  return <GameView worldId="demo-world" />;
}
