# UI Overhaul Implementation Plan

## Overview

This plan describes the refactoring required to transform the current UI layout into the target design. The primary goals are:

1. Remove the right-side chat panel (chat will be reimplemented differently)
2. Add HUD-style overlay elements around a fullscreen game canvas
3. Create new menu system with settings, journal, quests, and inventory
4. Add party list bar, minimap, and day/night indicator

## Current State

```
┌───────────────────────────────────┬──────────┐
│                                   │          │
│   Game Canvas (flex-1)            │  Chat    │
│   - Mode indicator (top-left)     │  Panel   │
│   - AgentInfoPanel (top-right)    │  (w-80)  │
│   - TileInfoPanel (bottom-left)   │          │
│   - Quick actions (bottom-right)  │          │
│                                   │          │
└───────────────────────────────────┴──────────┘
        + CharacterSheet modal overlay
```

## Target State

```
┌──────────────────────────────────────────────────────┐
│ <town_name / feature_name>       [day/night circle]  │
│                                                      │
│            ┌───────────────────────┐                 │
│            │   DM talking head     │  (placeholder)  │
│            │   (invisible)         │                 │
│            └───────────────────────┘                 │
│                                                      │
│                                                      │
│                   (fullscreen game canvas)           │
│                                                      │
│ ┌──────────┐                                         │
│ │ settings │                                         │
│ │ journal  │                   ┌──┬──┬──┬──┐ ┌─────┐│
│ │ quests   │                   │  │  │  │  │ │mini ││
│ │ inventory│                   └──┴──┴──┴──┘ │map  ││
│ └──────────┘                    party_list   └─────┘│
└──────────────────────────────────────────────────────┘
```

---

## Phase 1: State Management Updates

### 1.1 Extend gameStore with UI state

**File:** `src/stores/gameStore.ts`

Add the following state properties:

```typescript
// Game time state
gameTime: {
  hour: number;        // 0-23
  minute: number;      // 0-59
  day: number;         // Day count
  isNight: boolean;    // Derived: hour < 6 || hour >= 20
}

// Current location
currentLocation: {
  name: string;        // e.g., "Thornwood Village"
  type: 'settlement' | 'wilderness' | 'dungeon' | 'road';
} | null

// UI panel visibility
openPanel: 'settings' | 'journal' | 'quests' | 'inventory' | 'character' | null
```

Add corresponding actions:

```typescript
setGameTime: (time: Partial<GameState['gameTime']>) => void
setCurrentLocation: (location: GameState['currentLocation']) => void
setOpenPanel: (panel: GameState['openPanel']) => void
togglePanel: (panel: NonNullable<GameState['openPanel']>) => void
```

### 1.2 Add party member type support

Ensure party members are distinguishable from regular agents. Add to types:

```typescript
interface PartyMember {
  agentId: string;
  slot: 0 | 1 | 2 | 3;  // Position in party bar
  isActive: boolean;
}

// In GameState:
partySlots: [PartyMember | null, PartyMember | null, PartyMember | null, PartyMember | null]
```

---

## Phase 2: New UI Components

### 2.1 LocationDisplay Component

**File:** `src/components/ui/LocationDisplay.tsx`

- **Position:** Absolute top-left
- **Content:** Current location name with location type icon
- **Styling:** Semi-transparent background, amber text for location name
- **Behavior:** Updates when player moves between zones

```typescript
interface LocationDisplayProps {
  // Props pulled from store, no external props needed
}
```

### 2.2 TimeIndicator Component

**File:** `src/components/ui/TimeIndicator.tsx`

- **Position:** Absolute top-right
- **Visual:** Circular indicator showing sun/moon position
- **Content:**
  - Background gradient shifts from day (blue/yellow) to night (dark blue/purple)
  - Sun or moon icon based on time
  - Optional: hover to show exact time (HH:MM)
- **Size:** ~48-64px diameter circle

```typescript
interface TimeIndicatorProps {
  hour: number;
  minute: number;
  isNight: boolean;
}
```

### 2.3 MenuButtons Component

**File:** `src/components/ui/MenuButtons.tsx`

- **Position:** Absolute left side, vertically stacked
- **Buttons:** 4 square buttons arranged vertically
  1. Settings (gear icon)
  2. Journal (book icon)
  3. Quests (scroll/checklist icon)
  4. Inventory (backpack icon)
- **Styling:**
  - Semi-transparent background
  - Hover states
  - Active state when corresponding panel is open
- **Behavior:** Click toggles corresponding panel modal

```typescript
interface MenuButtonsProps {
  onOpenPanel: (panel: 'settings' | 'journal' | 'quests' | 'inventory') => void;
  activePanel: string | null;
}
```

### 2.4 PartyListBar Component

**File:** `src/components/ui/PartyListBar.tsx`

- **Position:** Absolute bottom-center
- **Visual:** 4 circular portrait slots in a horizontal row
- **Content per slot:**
  - Portrait/avatar (or placeholder if empty)
  - Health bar arc around the circle (optional)
  - Status indicator dot
  - Slot number (1-4)
- **Behavior:**
  - Click to select party member
  - Shows tooltip on hover with name and status
  - Empty slots show as dimmed circles
- **Size:** ~48-56px per portrait circle

```typescript
interface PartyListBarProps {
  partySlots: (PartyMember | null)[];
  selectedMemberId: string | null;
  onSelectMember: (memberId: string) => void;
}
```

### 2.5 Minimap Component

**File:** `src/components/ui/Minimap.tsx`

- **Position:** Absolute bottom-right
- **Visual:** Circular minimap with masked edges
- **Content:**
  - Simplified tile colors from surrounding area
  - Player position indicator (center)
  - Party member dots
  - North indicator
- **Size:** ~80-100px diameter
- **Behavior:**
  - Shows tiles within a fixed radius of player
  - Optional: click to open full map view

```typescript
interface MinimapProps {
  playerPosition: { x: number; y: number };
  tiles: Map<string, TileData>;
  partyPositions: { id: string; x: number; y: number }[];
  radius: number;  // Tile radius to show
}
```

### 2.6 DMPlaceholder Component

**File:** `src/components/ui/DMPlaceholder.tsx`

- **Position:** Absolute top-center
- **Visual:** Invisible container (for future DM talking head animation)
- **Size:** ~200-300px wide, ~100-150px tall
- **Behavior:** Renders nothing visually, but reserves space in DOM
- **Note:** This is a placeholder for future implementation

```typescript
interface DMPlaceholderProps {
  isVisible?: boolean;  // For future use when DM is "speaking"
}
```

---

## Phase 3: Panel/Modal Components

### 3.1 SettingsPanel Component

**File:** `src/components/ui/SettingsPanel.tsx`

- **Type:** Modal overlay (same pattern as CharacterSheet)
- **Content:**
  - Audio settings (volume sliders)
  - Graphics settings (quality, effects)
  - Gameplay settings (tooltips, auto-save)
  - Controls/keybindings reference
- **Behavior:** ESC or click outside to close

### 3.2 JournalPanel Component

**File:** `src/components/ui/JournalPanel.tsx`

- **Type:** Modal overlay
- **Content:**
  - List of journal entries (date-stamped)
  - Entry detail view
  - Categories/filters (story, notes, discoveries)
- **State needed:** Journal entries array in gameStore

### 3.3 QuestsPanel Component

**File:** `src/components/ui/QuestsPanel.tsx`

- **Type:** Modal overlay
- **Content:**
  - Active quests list
  - Completed quests (collapsible)
  - Quest detail view with objectives
- **State needed:** Quests array in gameStore with objectives tracking

### 3.4 InventoryPanel Component

**File:** `src/components/ui/InventoryPanel.tsx`

- **Type:** Modal overlay
- **Content:** Extract inventory section from CharacterSheet into standalone panel
- **Note:** CharacterSheet can either embed this or just link to it

---

## Phase 4: GameView Refactor

### 4.1 Remove ChatPanel Integration

**File:** `src/components/game/GameView.tsx`

- Remove `ChatPanel` import and usage
- Remove `handleSendMessage` callback
- Remove chat-related state
- Keep ChatPanel component file for future reimplementation

### 4.2 Fullscreen Canvas Layout

Update GameView structure:

```tsx
<div className="relative w-full h-screen bg-gray-950 overflow-hidden">
  {/* Fullscreen game canvas */}
  <GameCanvas width={windowWidth} height={windowHeight} />

  {/* HUD Overlay Layer */}
  <div className="absolute inset-0 pointer-events-none">
    {/* Top bar */}
    <div className="absolute top-4 left-4 pointer-events-auto">
      <LocationDisplay />
    </div>
    <div className="absolute top-4 right-4 pointer-events-auto">
      <TimeIndicator />
    </div>
    <div className="absolute top-4 left-1/2 -translate-x-1/2">
      <DMPlaceholder />
    </div>

    {/* Left menu */}
    <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-auto">
      <MenuButtons />
    </div>

    {/* Bottom bar */}
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-auto">
      <PartyListBar />
    </div>
    <div className="absolute bottom-4 right-4 pointer-events-auto">
      <Minimap />
    </div>
  </div>

  {/* Contextual overlays (existing) */}
  <AgentInfoPanel />
  <TileInfoPanel />

  {/* Modal panels */}
  <SettingsPanel isOpen={openPanel === 'settings'} onClose={closePanel} />
  <JournalPanel isOpen={openPanel === 'journal'} onClose={closePanel} />
  <QuestsPanel isOpen={openPanel === 'quests'} onClose={closePanel} />
  <InventoryPanel isOpen={openPanel === 'inventory'} onClose={closePanel} />
  <CharacterSheet isOpen={openPanel === 'character'} onClose={closePanel} />
</div>
```

### 4.3 Update Keyboard Shortcuts

```typescript
// New shortcuts
'i' -> Toggle inventory
'j' -> Toggle journal
'q' -> Toggle quests
'c' -> Toggle character sheet (existing)
'Escape' -> Close any open panel
'm' -> Toggle minimap expanded view (future)
```

### 4.4 Canvas Size Calculation

Update canvas size calculation to use full window:

```typescript
const updateSize = () => {
  setCanvasSize({
    width: window.innerWidth,
    height: window.innerHeight,
  });
};
```

---

## Phase 5: Cleanup and Polish

### 5.1 Remove/Archive Unused Components

- Keep `ChatPanel.tsx` but don't import (for future reference)
- Remove chat-related message handling from GameView

### 5.2 Update Component Index

If using barrel exports, update `src/components/ui/index.ts`

### 5.3 Responsive Considerations

- Menu buttons may need to collapse or resize on smaller screens
- Minimap size should scale with viewport
- Party bar should remain centered and accessible

### 5.4 Accessibility

- All interactive elements need proper ARIA labels
- Keyboard navigation between HUD elements
- Focus management for modals

---

## Implementation Order

1. **State updates** (Phase 1) - Foundation for new features
2. **LocationDisplay + TimeIndicator** (Phase 2.1-2.2) - Simple, isolated components
3. **MenuButtons** (Phase 2.3) - Required for panel access
4. **GameView refactor** (Phase 4) - Remove chat, restructure layout
5. **Panel modals** (Phase 3) - Settings, Journal, Quests, Inventory
6. **PartyListBar** (Phase 2.4) - Depends on party state
7. **Minimap** (Phase 2.5) - Most complex, can be placeholder initially
8. **DMPlaceholder** (Phase 2.6) - Simple placeholder, low priority
9. **Cleanup** (Phase 5)

---

## Open Questions

1. **Chat reimplementation**: How will the new chat system work? Should we design hooks/placeholders for it?

2. **Minimap complexity**: Should the minimap show real tile data initially, or start as a placeholder circle?

3. **Party system**: Is the party slot system (4 fixed slots) correct, or should it be dynamic?

4. **Modal vs overlay**: Should panels slide in from edges, or appear as centered modals?

5. **Existing overlays**: Should AgentInfoPanel and TileInfoPanel remain, or be redesigned to fit the new UI?

---

## File Changes Summary

### New Files
- `src/components/ui/LocationDisplay.tsx`
- `src/components/ui/TimeIndicator.tsx`
- `src/components/ui/MenuButtons.tsx`
- `src/components/ui/PartyListBar.tsx`
- `src/components/ui/Minimap.tsx`
- `src/components/ui/DMPlaceholder.tsx`
- `src/components/ui/SettingsPanel.tsx`
- `src/components/ui/JournalPanel.tsx`
- `src/components/ui/QuestsPanel.tsx`
- `src/components/ui/InventoryPanel.tsx`

### Modified Files
- `src/stores/gameStore.ts` - Add UI state, game time, location
- `src/components/game/GameView.tsx` - Major refactor
- `src/types/game.ts` - Add new type definitions

### Potentially Deprecated
- `src/components/ui/ChatPanel.tsx` - Keep file, remove from layout
