/**
 * Game Types
 * Core types for the game visualization and interaction.
 */

// World tile data from the builder
export interface TileData {
  x: number;
  y: number;
  elevation: number;
  biomeType: string;
  temperatureC: number;
  hasRoad: boolean;
  hasRiver: boolean;
  settlementId: number | null;
  settlementType: string | null;
  factionName: string | null;
  resourceType: string | null;
}

// Agent representation in the game
export interface GameAgent {
  id: string;
  name: string;
  x: number;
  y: number;
  currentAction: string | null;
  status: 'idle' | 'active' | 'sleeping' | 'dead';
  sprite: string;
  direction: 'up' | 'down' | 'left' | 'right';
  traits: AgentTraits;
}

export interface AgentTraits {
  occupation: string;
  personality: string[];
  goals: string[];
}

// Player character
export interface PlayerCharacter {
  id: string;
  name: string;
  x: number;
  y: number;
  isAgent: boolean;
  agentId: string | null;
  stats: CharacterStats;
  inventory: InventoryItem[];
}

export interface CharacterStats {
  health: number;
  maxHealth: number;
  mana: number;
  maxMana: number;
  strength: number;
  dexterity: number;
  constitution: number;
  intelligence: number;
  wisdom: number;
  charisma: number;
}

export interface InventoryItem {
  id: string;
  name: string;
  type: 'weapon' | 'armor' | 'consumable' | 'quest' | 'misc';
  quantity: number;
  description: string;
}

// Game session state
export interface GameSession {
  id: string;
  worldId: string;
  name: string;
  status: 'lobby' | 'active' | 'paused' | 'ended';
  players: PlayerCharacter[];
  dmAgentId: string | null;
  currentTurn: string | null;
  combatState: CombatState | null;
}

export interface CombatState {
  inCombat: boolean;
  turnOrder: string[];
  currentTurnIndex: number;
  enemies: CombatEntity[];
}

export interface CombatEntity {
  id: string;
  name: string;
  health: number;
  maxHealth: number;
  initiative: number;
}

// Chat/dialogue types
export interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  senderType: 'player' | 'agent' | 'dm' | 'system';
  content: string;
  timestamp: number;
  channel: 'global' | 'party' | 'whisper' | 'narration';
}

// World event for visual feedback
export interface WorldEvent {
  id: string;
  type: 'combat' | 'dialogue' | 'discovery' | 'trade' | 'movement';
  x: number;
  y: number;
  description: string;
  participants: string[];
  timestamp: number;
}

// Viewport state
export interface ViewportState {
  x: number;
  y: number;
  zoom: number;
  width: number;
  height: number;
}

// Input modes
export type InputMode =
  | 'explore'      // Free movement
  | 'dialogue'     // Talking to agent/NPC
  | 'combat'       // Combat mode
  | 'inventory'    // Managing inventory
  | 'menu';        // Game menu

// Action types for player commands
export interface PlayerAction {
  type: 'move' | 'speak' | 'interact' | 'attack' | 'use_item' | 'skill_check';
  targetX?: number;
  targetY?: number;
  targetId?: string;
  itemId?: string;
  skillName?: string;
  dialogueOption?: string;
}

// Game time state
export interface GameTime {
  hour: number;        // 0-23
  minute: number;      // 0-59
  day: number;         // Day count
}

// Current location info
export interface CurrentLocation {
  name: string;
  type: 'settlement' | 'wilderness' | 'dungeon' | 'road';
}

// Party member slot
export interface PartyMember {
  agentId: string;
  slot: 0 | 1 | 2 | 3;
  isActive: boolean;
}

// UI Panel types
export type UIPanel = 'settings' | 'journal' | 'quests' | 'inventory' | 'character' | null;

// Journal entry
export interface JournalEntry {
  id: string;
  title: string;
  content: string;
  category: 'story' | 'notes' | 'discoveries';
  timestamp: number;
  day: number;
}

// Quest types
export interface Quest {
  id: string;
  title: string;
  description: string;
  objectives: QuestObjective[];
  status: 'active' | 'completed' | 'failed';
  giverName?: string;
  reward?: string;
}

export interface QuestObjective {
  id: string;
  description: string;
  isCompleted: boolean;
  current?: number;
  target?: number;
}
