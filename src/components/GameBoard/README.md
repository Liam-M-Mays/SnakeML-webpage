# GameBoard Components

This directory contains game visualization components for different environments.

## Components to Implement

### SnakeBoard.jsx (TASK-001)

**Purpose**: Visualize Snake game state in real-time during training

**Props:**
```typescript
interface SnakeBoardProps {
  gameState: {
    grid_size: number;
    snake_position: Array<{x: number, y: number}>;
    food_position: {x: number, y: number};
    score: number;
    game_over: boolean;
    death_reason?: string;
  };
  cellSize?: number;  // Optional: cell size in pixels (auto-calculated if not provided)
}
```

**Expected behavior:**
- Render a grid of size `grid_size × grid_size`
- Display snake body at positions in `snake_position`
  - First element is the head (should be visually distinct)
  - Remaining elements are body segments
- Display food at `food_position`
- Show score
- Show game over state when `game_over=true`
- Update smoothly when props change

**Implementation options:**
- HTML5 Canvas (recommended for performance)
- CSS Grid (simpler, may be slower for large grids)

**Usage in App.jsx:**
```jsx
import SnakeBoard from './components/GameBoard/SnakeBoard';

// In Train tab
{progress && (
  <SnakeBoard gameState={progress.gameState} />
)}
```

**Data source:**
Game state comes from `training_progress` SocketIO event, which should include `gameState` field (see TASK-001 for backend changes needed).

---

### Future Components

When adding new game environments, create corresponding board components:
- `TetrisBoard.jsx` — Tetris game visualization
- `Connect4Board.jsx` — Connect-4 game visualization
- etc.

**Pattern:**
All game board components should accept a `gameState` prop matching the shape returned by that environment's `render_state()` method.
