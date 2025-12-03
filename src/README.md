# Frontend Documentation

This is the frontend for the AI/RL Playground, built with React and Vite.

## Directory Structure

```
src/
├── App.jsx               # Main app component with tab navigation
├── App.css               # Global styles
├── index.jsx             # Entry point
├── hooks/                # Custom React hooks
│   ├── useSocket.js      # SocketIO connection management
│   └── useTraining.js    # Training state and controls
├── utils/                # Utility functions
│   └── api.js            # Backend API client
└── (TODO) components/    # Reusable UI components
    ├── GameBoard/
    │   └── SnakeBoard.jsx        # Snake game visualization
    ├── NetworkBuilder/
    │   └── NetworkBuilder.jsx    # Visual network architecture builder
    ├── RewardDesigner/
    │   └── RewardDesigner.jsx    # Event-based reward configuration
    ├── InputSelector/
    │   └── InputSelector.jsx     # Input/observation configuration
    └── Charts/
        └── MetricsChart.jsx      # Training metrics visualization
```

## Current Structure

### App.jsx

**Purpose**: Main application component with tab-based navigation

**Tabs:**
1. **Train** — Training controls + live dashboard
2. **Configuration** — Algorithm, hyperparameters, rewards, network
3. **Models** — Saved model management
4. (TODO) **Replays** — View and playback recorded episodes

**State management:**
- `config` — Current training configuration (RunConfig)
- `activeTab` — Current tab selection
- `deviceInfo` — Backend device info (CPU/CUDA/MPS)
- `models` — List of saved models (for Models tab)

**Key hooks used:**
- `useTraining()` — Training session state and controls
- `useEffect()` — Load device info and default config on mount

**Data flow:**
1. User configures training in Configuration tab
2. Clicks "Start Training" in Train tab
3. `handleStartTraining()` calls `startTraining(config)` from `useTraining` hook
4. Training progress received via `useTraining` hook (SocketIO events)
5. UI updates reactively

---

### hooks/useTraining.js

**Purpose**: Manage training session state and provide control functions

**Exports:**
```javascript
const {
  isTraining,      // boolean: Is training currently running?
  runId,           // string: Current run ID
  status,          // object: Training status (episodes, scores, etc.)
  progress,        // object: Real-time progress updates
  episodes,        // array: Recent episode summaries
  error,           // string: Error message (if any)
  startTraining,   // function: Start training with config
  stopTraining,    // function: Stop current training
} = useTraining();
```

**SocketIO events listened:**
- `training_progress` → Updates `progress` state
- `episode_summary` → Appends to `episodes` array

**Usage:**
```javascript
// In a component
const { isTraining, progress, startTraining, stopTraining } = useTraining();

// Start training
await startTraining({
  algo: "dqn",
  env_name: "snake",
  // ... config
});

// Stop training
await stopTraining();

// Access live progress
console.log(progress.episode, progress.current_score, progress.best_score);
```

---

### hooks/useSocket.js

**Purpose**: Manage SocketIO connection lifecycle

**Exports:**
```javascript
const { socket, connected } = useSocket();
```

**Behavior:**
- Connects to backend SocketIO server on mount
- Disconnects on unmount
- Auto-reconnects on connection loss
- Provides `socket` instance for custom event listeners

**Usage:**
```javascript
const { socket, connected } = useSocket();

useEffect(() => {
  if (!socket) return;

  socket.on("custom_event", (data) => {
    console.log("Received custom event:", data);
  });

  return () => socket.off("custom_event");
}, [socket]);
```

---

### utils/api.js

**Purpose**: Backend API client (wraps fetch calls)

**Exports:**
```javascript
export const api = {
  // Environments
  getEnvironments,
  getEnvironmentMetadata,

  // Configuration
  getDefaultConfig,

  // Training
  startTraining,
  stopTraining,
  getTrainingStatus,

  // Models
  listModels,
  deleteModel,
  getModelConfig,

  // Metrics
  getMetrics,
  getDeathStats,

  // Replays
  listReplays,
  getReplay,
  deleteReplay,

  // System
  getDeviceInfo,
  healthCheck,
};
```

**All functions return Promises:**
```javascript
// Example usage
const config = await api.getDefaultConfig("dqn");
const models = await api.listModels("snake");
const deviceInfo = await api.getDeviceInfo();
```

**Adding a new API method:**
1. Add function to `api` object in `api.js`
2. Follow existing pattern (fetch → `.then(r => r.json())`)
3. Document in `docs/ARCHITECTURE.md`

---

## Tab Details

### Train Tab

**Purpose**: Start/stop training and monitor live progress

**Components:**
- Training controls (Start/Stop button, Max Speed checkbox)
- Live statistics (Episode, Current Score, Best Score, Avg Score)
- Episodes per second indicator (in max speed mode)
- Recent episodes table (score, reward, length, death reason, loss, epsilon/entropy)
- (TODO) Game board visualization

**State flow:**
1. User clicks "Start Training"
2. `handleStartTraining()` called
3. Sends config to backend via `api.startTraining()`
4. `useTraining` hook starts listening for SocketIO events
5. UI updates as `progress` and `episodes` state changes

**Missing feature (TASK-001):**
- Game board visualization (SnakeBoard component)
- Should render real-time game state during training

---

### Configuration Tab

**Purpose**: Configure all training parameters

**Sections:**

1. **Algorithm**
   - Dropdown: DQN or PPO
   - Loads default config when changed

2. **Environment**
   - Grid size slider/input
   - (TODO) Vision mode selector (TASK-006)
   - (TODO) Game selection (when multi-game support added)

3. **Rewards**
   - Numeric inputs for each event type (apple, death_wall, death_self, death_starv, step)
   - (TODO) Event-based reward designer (TASK-005)

4. **Hyperparameters**
   - Dynamic form based on selected algorithm
   - Learning rate, gamma, batch size, epsilon/entropy, etc.

5. **Network Architecture**
   - Currently: Read-only display of layers
   - (TODO) Visual network builder (TASK-004)

6. **(TODO) Device Selection** (TASK-003)
   - Dropdown: Auto, CPU, CUDA, MPS
   - Shows current device

**State:**
- All changes update local `config` state
- Config sent to backend when user starts training
- Not persisted locally (user must start training to save)

---

### Models Tab

**Purpose**: View and manage saved models

**Features:**
- Lists all saved models with metadata:
  - Run ID, algorithm, environment
  - Best score, avg reward, total episodes
  - Created timestamp
- Delete button per model
- (TODO) Load config button (TASK-002)
- (TODO) Resume training button (TASK-002)
- (TODO) Export model button (TASK-012)

**Data source:**
- Fetched from `api.listModels()` when tab becomes active
- Refreshes after deletion

**Missing features:**
- Load model configuration
- Resume training from saved weights
- Export/import models

---

## Planned Components (TODO)

### SnakeBoard Component (TASK-001)

**File**: `src/components/GameBoard/SnakeBoard.jsx`

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
  cellSize?: number;
}
```

**Behavior:**
- Renders game grid using HTML Canvas or CSS Grid
- Shows snake body (distinct head) and food
- Updates in real-time during training
- Displays game over state

**Usage:**
```jsx
<SnakeBoard gameState={progress.gameState} />
```

**Integration:**
- Used in Train tab
- Receives `gameState` from `training_progress` SocketIO event
- Backend must include `env.render_state()` in progress emissions

---

### NetworkBuilder Component (TASK-004)

**File**: `src/components/NetworkBuilder/NetworkBuilder.jsx`

**Props:**
```typescript
interface NetworkBuilderProps {
  networkConfig: {
    layers: Array<{
      type: "dense";
      units: number;
      activation: string;
    }>;
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Features:**
- Add/remove layers
- Adjust units and activation per layer
- Drag to reorder layers (optional)
- Presets (Small, Medium, Large)
- Real-time validation

**Usage:**
```jsx
<NetworkBuilder
  networkConfig={config.network_config}
  onChange={(newConfig) => setConfig({...config, network_config: newConfig})}
  disabled={isTraining}
/>
```

---

### RewardDesigner Component (TASK-005)

**File**: `src/components/RewardDesigner/RewardDesigner.jsx`

**Props:**
```typescript
interface RewardDesignerProps {
  rewardConfig: {
    apple: number;
    death_wall: number;
    // ... + event-based formulas
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Features:**
- Simple mode: Numeric inputs (current behavior)
- Advanced mode: Event-based formula editor
  - Select event triggers
  - Access to game variables
  - Formula builder with preview

**Usage:**
```jsx
<RewardDesigner
  rewardConfig={config.reward_config}
  onChange={(newConfig) => setConfig({...config, reward_config: newConfig})}
  disabled={isTraining}
/>
```

---

### InputSelector Component (TASK-006)

**File**: `src/components/InputSelector/InputSelector.jsx`

**Props:**
```typescript
interface InputSelectorProps {
  envConfig: {
    vision: number;
    seg_size: number;
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Features:**
- Vision mode selector (immediate, 3×3, 5×5, full grid)
- Body segment tracking slider
- Observation space size preview

**Usage:**
```jsx
<InputSelector
  envConfig={config.env_config}
  onChange={(newConfig) => setConfig({...config, env_config: newConfig})}
  disabled={isTraining}
/>
```

---

### MetricsChart Component (TASK-008)

**File**: `src/components/Charts/MetricsChart.jsx`

**Props:**
```typescript
interface MetricsChartProps {
  metrics: Array<{
    episode: number;
    score: number;
    reward: number;
    loss: number;
    epsilon?: number;
    entropy?: number;
  }>;
}
```

**Charts:**
- Score over episodes
- Average reward over episodes
- Loss over episodes
- Epsilon/Entropy over episodes

**Usage:**
```jsx
const metrics = await api.getMetrics(envName, runId, 1000);
<MetricsChart metrics={metrics} />
```

---

## Styling

### Global Styles

**File**: `App.css`

**Variables:**
```css
:root {
  --bg-dark: #1a1a1a;
  --bg-card: #2a2a2a;
  --text-primary: #ffffff;
  --text-secondary: #b0b0b0;
  --accent: #4a9eff;
  --success: #4caf50;
  --danger: #f44336;
}
```

**Component classes:**
- `.stat-card` — Stat display cards
- `.config-section` — Configuration sections
- `.btn`, `.btn-primary`, `.btn-danger` — Buttons
- `.model-card` — Model display cards

**Adding new styles:**
1. For global styles, add to `App.css`
2. For component-specific styles, create `ComponentName.css` next to component file
3. Import in component: `import "./ComponentName.css"`

---

## State Management

**Current approach**: Local component state + custom hooks

**State locations:**
- `App.jsx` — Global app state (config, activeTab, models)
- `useTraining` hook — Training session state
- `useSocket` hook — SocketIO connection state

**Future considerations:**
- If state becomes complex, consider Redux or Zustand
- For now, prop drilling and hooks are sufficient

---

## API Integration

### Starting Training

```javascript
// 1. User configures in Configuration tab
const config = {
  algo: "dqn",
  env_name: "snake",
  env_config: { grid_size: 10, ... },
  reward_config: { apple: 1.0, ... },
  network_config: { layers: [...] },
  hyperparams: { learning_rate: 0.001, ... },
};

// 2. User clicks "Start Training"
await api.startTraining({ ...config, max_speed: true });

// 3. Backend responds with run_id and starts training

// 4. SocketIO events stream progress
socket.on("training_progress", (data) => {
  // { episode, current_score, best_score, avg_score, ... }
});

socket.on("episode_summary", (data) => {
  // { episode, score, reward, length, death_reason, loss, ... }
});
```

### Fetching Models

```javascript
const { models } = await api.listModels("snake");
// Returns array of model metadata

// Get full config
const config = await api.getModelConfig("snake", runId);
```

### Fetching Metrics

```javascript
const { metrics } = await api.getMetrics("snake", runId, 1000);
// Returns last 1000 metrics

const { death_stats } = await api.getDeathStats("snake", runId);
// Returns { wall: 45, self: 32, starvation: 23 }
```

---

## Development Workflow

### Running Locally

```bash
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

Backend must be running on: `http://127.0.0.1:5000`

### Hot Module Replacement (HMR)

Vite provides instant HMR. Changes to `.jsx` files update without page reload.

### Building for Production

```bash
npm run build
```

Output: `dist/` directory

Serve with any static file server or integrate with Flask.

---

## Common Issues

### Backend Connection Errors

If frontend can't connect to backend:
- Ensure backend is running: `python backend/App.py`
- Check `API_BASE` in `src/utils/api.js` (should be `http://127.0.0.1:5000/api`)
- Check browser console for CORS errors

### SocketIO Not Connecting

If SocketIO events aren't received:
- Check `useSocket.js` connection URL (should be `http://127.0.0.1:5000`)
- Verify backend has SocketIO enabled and CORS configured
- Check browser console for connection errors

### State Not Updating

If UI doesn't update during training:
- Verify SocketIO is connected (`useSocket` hook)
- Check `useTraining` hook is listening for events
- Check browser console for errors

---

## Future Enhancements

See `docs/AGENT_TASKS.md` for specific tasks.

**High priority:**
- Snake game board visualization (TASK-001)
- Model loading/resuming (TASK-002)
- Device selection UI (TASK-003)

**Medium priority:**
- Visual network builder (TASK-004)
- Event-based reward designer (TASK-005)
- Input/observation selector (TASK-006)
- Replay viewer (TASK-007)

**Low priority:**
- Metrics charts (TASK-008)
- Death analytics dashboard (TASK-009)
- Hyperparameter presets (TASK-011)

**See also:**
- `docs/ARCHITECTURE.md` — Overall system architecture
- `docs/AGENT_TASKS.md` — Specific development tasks
- `backend/README.md` — Backend documentation
