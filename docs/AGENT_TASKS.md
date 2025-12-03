# Agent Development Tasks

This document contains specific, bounded tasks for future development agents. Each task is designed to be worked on independently without breaking existing functionality.

## Task Format

Each task includes:
- **Task ID**: Unique identifier
- **Title**: Brief description
- **Priority**: High / Medium / Low
- **Complexity**: Simple / Medium / Complex
- **Files to modify**: Specific files that can/should be changed
- **Contracts to respect**: Interfaces and APIs that MUST NOT be broken
- **Acceptance criteria**: What "done" looks like

---

## Development Guidelines

### Before Starting a Task

1. **Read the contracts**: See `docs/ARCHITECTURE.md` for all interface definitions
2. **Check dependencies**: Some tasks depend on others being completed first
3. **Understand boundaries**: Only modify files listed in "Files to modify"
4. **Respect interfaces**: Do not change API shapes, component props, or method signatures unless explicitly instructed

### While Working

1. **Test incrementally**: Ensure the app still runs after each change
2. **Use TODO comments**: Mark incomplete portions clearly
3. **Follow existing patterns**: Match the style and structure of surrounding code
4. **Add minimal dependencies**: Avoid adding new libraries unless necessary

### After Completing

1. **Test end-to-end**: Verify the feature works as expected
2. **Update docs**: If you added new interfaces, document them in ARCHITECTURE.md
3. **Clean up**: Remove debug code, console.logs, and commented-out code
4. **Commit**: Use descriptive commit messages

---

## High Priority Tasks

### TASK-001: Snake Game Board Visualization

**Priority**: High
**Complexity**: Medium
**Dependencies**: None

**Description:**
Implement a visual Snake game board component that renders the game state in real-time during training. This is the most glaring missing feature—users currently cannot see the snake playing.

**Files to modify:**
- `src/components/GameBoard/SnakeBoard.jsx` (create)
- `src/components/GameBoard/SnakeBoard.css` (create)
- `src/App.jsx` (import and use SnakeBoard in Train tab)

**Contracts to respect:**
- Component receives `gameState` prop matching the shape returned by `SnakeEnv.render_state()`:
  ```typescript
  {
    grid_size: number;
    snake_position: Array<{x: number, y: number}>;
    food_position: {x: number, y: number};
    score: number;
    game_over: boolean;
    death_reason?: string;
  }
  ```
- Backend's `render_state()` method in `backend/envs/snake_env.py` MUST NOT be changed

**Implementation details:**
- Use HTML Canvas or CSS Grid for rendering
- Cell size should auto-scale based on `grid_size`
- Snake head should be visually distinct from body
- Food should be clearly visible
- Show game over state when `game_over=true`
- Update smoothly when state changes

**Data source:**
- In training mode: Game state should be included in `training_progress` SocketIO events
  - **Backend change needed**: Modify `training_manager.py` line 222 to include `env.render_state()` in the emitted progress
- Hook into `useTraining` hook to receive state
- Pass to `<SnakeBoard gameState={progress.gameState} />`

**Acceptance criteria:**
- [ ] SnakeBoard component renders a grid matching `grid_size`
- [ ] Snake body is displayed at correct positions
- [ ] Food is displayed at correct position
- [ ] Component updates in real-time during training
- [ ] Game over state is visually indicated
- [ ] Works in both normal and max speed modes
- [ ] No console errors

---

### TASK-002: Model Loading / Resume Training

**Priority**: High
**Complexity**: Medium
**Dependencies**: None

**Description:**
Add ability to load a saved model's configuration and optionally resume training from saved weights. Currently, users can save models but cannot do anything with them.

**Files to modify:**
- `backend/App.py` (add new endpoint)
- `backend/training_manager.py` (support loading weights)
- `backend/agents/dqn_agent.py` (ensure `load()` works correctly)
- `backend/agents/ppo_agent.py` (ensure `load()` works correctly)
- `src/utils/api.js` (add API client methods)
- `src/App.jsx` (add "Load Config" and "Resume Training" buttons in Models tab)

**Contracts to respect:**
- Storage layout MUST remain unchanged (see `docs/ARCHITECTURE.md` Storage section)
- RunConfig schema MUST remain compatible
- Existing API endpoints MUST NOT be modified (only add new ones)

**New API endpoints to add:**

1. `POST /api/training/resume`
   - Body: `{"env_name": str, "run_id": str, "continue_training": bool}`
   - Response: Same as `/api/training/start`
   - Behavior:
     - Load config from storage
     - If `continue_training=true`, load weights into agent
     - Start training session (keeping same run_id if continuing, or new run_id if cloning)

**Frontend changes:**
- In Models tab, add two buttons per model:
  - "Load Config" → Fetches config, populates Configuration tab, switches to Config tab
  - "Resume Training" → Calls `/api/training/resume` with `continue_training=true`, switches to Train tab

**Backend changes:**
- In `TrainingSession.__init__()`, accept optional `weights_path` parameter
- After creating agent, call `agent.load(weights_path)` if provided
- Verify DQNAgent and PPOAgent `load()` methods properly restore optimizer state

**Acceptance criteria:**
- [ ] "Load Config" button populates Configuration tab with saved settings
- [ ] "Resume Training" button loads weights and continues training
- [ ] Resumed training maintains episode count and metrics
- [ ] Metrics append to existing metrics file
- [ ] New models can be created from loaded configs (clone functionality)
- [ ] No errors when loading models saved with different PyTorch versions (handle gracefully)

---

### TASK-003: Device Selection UI

**Priority**: High
**Complexity**: Simple
**Dependencies**: None

**Description:**
Add UI controls for device selection (CPU/CUDA/auto) in the Configuration tab. The backend already supports device selection via `FORCE_DEVICE` env variable, but there's no UI for it.

**Files to modify:**
- `src/App.jsx` (add device selector in Configuration tab)
- `backend/App.py` (add endpoint to change device preference)
- `backend/utils/device.py` (add function to set device preference programmatically)

**Contracts to respect:**
- `get_device()` behavior in `utils/device.py` MUST remain backward compatible
- Device info shape returned by `/api/device` MUST NOT change

**Implementation details:**

**Backend changes:**
1. Add new endpoint:
   ```python
   @app.route("/api/device/set", methods=["POST"])
   def set_device_preference():
       """Set device preference for future training sessions."""
       # Body: {"device": "cpu" | "cuda" | "mps" | "auto"}
       # Store preference in memory (or config file)
       # Return new device info
   ```

2. Modify `training_manager.py` to respect preference when creating session

**Frontend changes:**
1. In Configuration tab, add a new section:
   ```jsx
   <div className="config-section">
     <h3>Device</h3>
     <select value={devicePreference} onChange={handleDeviceChange}>
       <option value="auto">Auto (recommended)</option>
       <option value="cpu">CPU</option>
       <option value="cuda">CUDA (NVIDIA GPU)</option>
       <option value="mps">MPS (Apple Silicon)</option>
     </select>
     <div className="device-info">
       Current: {deviceInfo.name} ({deviceInfo.type})
     </div>
   </div>
   ```

2. Call `/api/device/set` when user changes selection
3. Show current device info from `/api/device`

**Acceptance criteria:**
- [ ] Device selector dropdown appears in Configuration tab
- [ ] Shows current device status
- [ ] Selection persists across page reloads (stored in backend)
- [ ] Training sessions use selected device
- [ ] "Auto" mode works correctly (MPS > CUDA > CPU priority)
- [ ] Gracefully handles unavailable devices (e.g., user selects CUDA but no GPU)

---

## Medium Priority Tasks

### TASK-004: Visual Network Builder

**Priority**: Medium
**Complexity**: Complex
**Dependencies**: None

**Description:**
Create a visual network architecture builder component to replace the current JSON-based network configuration. Users should be able to add/remove layers, adjust units, and change activations through a UI.

**Files to modify:**
- `src/components/NetworkBuilder/NetworkBuilder.jsx` (create)
- `src/components/NetworkBuilder/NetworkBuilder.css` (create)
- `src/App.jsx` (replace network display in Configuration tab with NetworkBuilder)

**Contracts to respect:**
- `network_config` schema MUST remain:
  ```typescript
  {
    layers: Array<{
      type: "dense";
      units: number;
      activation: "relu" | "leaky_relu" | "tanh" | "sigmoid" | "linear";
    }>
  }
  ```
- Backend's `network_builder.py` MUST NOT require changes

**Component interface:**
```typescript
interface NetworkBuilderProps {
  networkConfig: NetworkConfig;
  onChange: (newConfig: NetworkConfig) => void;
  disabled?: boolean;
}
```

**Features to implement:**
1. **Layer list view** (vertical stack of layer cards)
   - Each card shows: layer index, units, activation
   - "Add Layer" button (inserts after current position)
   - "Remove Layer" button (min 1 layer)
   - Drag to reorder (optional, nice-to-have)

2. **Layer editor** (inline in each card)
   - Units: number input (min: 1, max: 2048, step: 1)
   - Activation: dropdown (relu, leaky_relu, tanh, sigmoid, linear)

3. **Presets** (optional, nice-to-have)
   - Dropdown with common architectures:
     - Small (128, 128)
     - Medium (128, 256, 128)
     - Large (256, 512, 256)
     - Custom (user-defined)
   - "Save as preset" button to save current config

**Acceptance criteria:**
- [ ] Can add/remove layers
- [ ] Can modify units and activation per layer
- [ ] Changes update parent component via `onChange`
- [ ] Disabled state prevents edits
- [ ] Validation: at least 1 layer required
- [ ] Presets work (if implemented)
- [ ] Visually clear and intuitive

---

### TASK-005: Event-Based Reward Designer

**Priority**: Medium
**Complexity**: Complex
**Dependencies**: None (backend + frontend changes)

**Description:**
Extend the reward system to support event-based rewards with formulas and access to game variables. Currently, rewards are simple numeric constants.

**Files to modify:**
- `backend/envs/snake_env.py` (extend reward calculation)
- `backend/agents/config.py` (extend reward_config schema)
- `src/components/RewardDesigner/RewardDesigner.jsx` (create)
- `src/components/RewardDesigner/RewardDesigner.css` (create)
- `src/App.jsx` (use RewardDesigner in Configuration tab)

**Contracts to respect:**
- Backward compatibility: existing numeric rewards MUST still work
- `env.step()` return signature MUST NOT change

**Backend changes:**

1. Extend `reward_config` schema to support formulas:
   ```python
   reward_config = {
       # Legacy: simple numeric rewards (still supported)
       "apple": 1.0,
       "death_wall": -1.0,
       # ...

       # New: event-based formulas (optional)
       "formulas": [
           {
               "event": "apple_eaten",
               "formula": "1.0 + 0.1 * snake_length",
               "enabled": True
           },
           {
               "event": "step",
               "formula": "-0.001 * (1 + hunger / max_hunger)",
               "enabled": True
           },
           # ...
       ]
   }
   ```

2. In `SnakeEnv.step()`, evaluate formulas:
   ```python
   def _calculate_reward(self, event_type):
       # Check for formula-based reward
       if "formulas" in self.reward_config:
           for formula_config in self.reward_config["formulas"]:
               if formula_config["event"] == event_type and formula_config.get("enabled"):
                   # Evaluate formula with variables
                   return self._eval_formula(formula_config["formula"])

       # Fallback to legacy numeric rewards
       return self.reward_config.get(event_type, 0.0)

   def _eval_formula(self, formula_str):
       # Safe evaluation with access to variables
       variables = {
           "snake_length": len(self.snake_position),
           "hunger": self.hunger,
           "max_hunger": self.starvation_limit,
           "grid_size": self.grid_size,
           "score": self.score,
           # ...
       }
       # Use safe eval (e.g., simpleeval library) or manual parsing
       # DO NOT use Python's eval() directly (security risk)
       return safe_eval(formula_str, variables)
   ```

**Frontend changes:**

Create `RewardDesigner` component with:
1. **Simple mode** (default): Numeric inputs (current behavior)
2. **Advanced mode**: Event-based formula editor
   - List of events (apple_eaten, death_wall, death_self, step, etc.)
   - Per-event: enable/disable toggle
   - Formula input field with variable suggestions
   - Available variables listed (snake_length, hunger, distance_to_food, etc.)
   - Preview calculated reward with example values

**Security note:**
Use a safe expression evaluator (e.g., `simpleeval` library) to avoid code injection. DO NOT use Python's `eval()`.

**Acceptance criteria:**
- [ ] Legacy numeric rewards still work
- [ ] Can define formula-based rewards
- [ ] Formulas have access to game variables
- [ ] Formulas are evaluated safely (no code injection)
- [ ] UI clearly shows available variables
- [ ] Can toggle between simple and advanced modes
- [ ] Invalid formulas show error messages

---

### TASK-006: Input/Observation Configuration UI

**Priority**: Medium
**Complexity**: Medium
**Dependencies**: None

**Description:**
Add a UI component for configuring what the agent "sees" as input (observation space). Currently, `vision` and `seg_size` are hardcoded or hidden.

**Files to modify:**
- `src/components/InputSelector/InputSelector.jsx` (create)
- `src/components/InputSelector/InputSelector.css` (create)
- `src/App.jsx` (add InputSelector to Configuration tab)

**Contracts to respect:**
- `env_config` schema (keys: `grid_size`, `vision`, `seg_size`)
- `SnakeEnv` observation space calculation MUST NOT change

**Component interface:**
```typescript
interface InputSelectorProps {
  envConfig: {
    vision: number;     // 0=immediate, >0=window, -1=full grid
    seg_size: number;   // Body segments to track
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Features:**

1. **Vision Mode** selector:
   - Radio buttons or dropdown:
     - "Immediate Danger (4 directions)" → `vision: 0`
     - "3×3 Window" → `vision: 1`
     - "5×5 Window" → `vision: 2`
     - "Full Grid" → `vision: -1`
   - Show description of each mode
   - Indicate resulting observation space size

2. **Body Segment Tracking** slider:
   - Range: 1-10
   - Shows how many body segments are tracked in observation
   - Explain that this helps the agent infer body shape without seeing full grid

3. **Observation Space Preview**:
   - Show calculated total observation size
   - Break down by component (food position, direction, hunger, segments, danger)

**Acceptance criteria:**
- [ ] Can select vision mode
- [ ] Can adjust segment tracking
- [ ] Shows observation space size
- [ ] Changes propagate to parent config
- [ ] Disabled state works
- [ ] Clear descriptions for each mode

---

### TASK-007: Replay Viewer

**Priority**: Medium
**Complexity**: Complex
**Dependencies**: TASK-001 (SnakeBoard component)

**Description:**
Create a replay viewer tab where users can watch recorded episodes. Replays are already saved, but there's no UI to view them.

**Files to modify:**
- `src/pages/ReplaysPage.jsx` (create)
- `src/App.jsx` (add Replays tab)
- `src/hooks/useReplays.js` (create, manages replay playback)
- `src/utils/replayEngine.js` (create, handles replay execution)

**Contracts to respect:**
- Replay data format (see `storage_manager.py` line 160-166):
  ```typescript
  {
    replay_id: string;
    seed: number;
    actions: number[];
    score: number;
    length: number;
    death_reason: string;
    timestamp: string;
  }
  ```
- Use existing `/api/replays/` endpoints

**Implementation:**

1. **ReplaysPage component:**
   - Model selector dropdown (fetch from `/api/models`)
   - Replay list (fetch from `/api/replays/<env>/<run_id>`)
     - Show: replay_id, score, length, death_reason, timestamp
     - Sort by score (best first)
   - Click replay → load and play

2. **Replay engine:**
   ```javascript
   class ReplayEngine {
     constructor(envName, replayData) {
       // Initialize environment with seed
       // Prepare actions queue
     }

     async step() {
       // Execute next action
       // Update game state
       // Return new state
     }

     reset() { /* ... */ }
     getCurrentState() { /* ... */ }
   }
   ```

3. **Playback controls:**
   - Play/Pause button
   - Step forward/backward buttons
   - Speed control slider (0.5x, 1x, 2x, 5x)
   - Progress bar showing current step / total steps

4. **Visualization:**
   - Reuse `SnakeBoard` component (from TASK-001)
   - Display current score, step count
   - Highlight when food is eaten

**Technical note:**
Since environments run on the backend, you'll need to either:
- (Option A) Replay on backend and stream states via SocketIO
- (Option B) Implement lightweight JS version of SnakeEnv for client-side replay

Option B is recommended for better UX (no backend dependency, instant seek).

**Acceptance criteria:**
- [ ] Can select a model and see its replays
- [ ] Can click a replay to load it
- [ ] Playback controls work (play/pause, step, speed)
- [ ] Game board shows replay animation
- [ ] Can seek to any point in replay
- [ ] Shows final score and death reason
- [ ] Replays deterministic (same seed → same episode)

---

## Lower Priority Tasks

### TASK-008: Metrics Charts

**Priority**: Low
**Complexity**: Medium
**Dependencies**: None

**Description:**
Add charts to visualize training metrics over time (score, reward, loss, epsilon/entropy).

**Files to modify:**
- `src/components/Charts/MetricsChart.jsx` (create)
- `src/App.jsx` (add chart to Train tab)
- Use a charting library (e.g., Chart.js, Recharts, or Victory)

**Data source:**
- Fetch from `/api/metrics/<env>/<run_id>?limit=1000`
- Update in real-time as new episodes complete

**Charts to include:**
1. Score over episodes (line chart)
2. Average reward over episodes (line chart)
3. Loss over episodes (line chart, DQN/PPO)
4. Epsilon (DQN) or Entropy (PPO) over episodes

**Acceptance criteria:**
- [ ] Charts render correctly
- [ ] Update in real-time during training
- [ ] Can toggle which metrics to show
- [ ] Responsive design

---

### TASK-009: Death Analytics Dashboard

**Priority**: Low
**Complexity**: Simple
**Dependencies**: None

**Description:**
Add a section showing death reason statistics (pie chart or bar chart) to help users understand how their agent is failing.

**Files to modify:**
- `src/App.jsx` (add death stats section in Train tab)

**Data source:**
- Fetch from `/api/metrics/<env>/<run_id>/death_stats`
- Returns: `{"wall": 45, "self": 32, "starvation": 23}`

**Visualization:**
- Pie chart or bar chart showing percentage breakdown
- Update periodically during training

**Acceptance criteria:**
- [ ] Death stats displayed clearly
- [ ] Updates during training
- [ ] Visually intuitive (colors, labels)

---

### TASK-010: Rename App to "AI Playground"

**Priority**: Low
**Complexity**: Simple
**Dependencies**: None

**Description:**
Update branding from "SnakeML Playground" to "AI Playground" or "RL Playground" to reflect multi-game vision.

**Files to modify:**
- `src/App.jsx` (header title, line 76)
- `src/index.html` or `index.html` (page title)
- `README.md` (update references)
- `backend/App.py` (startup message, line 259)

**Contracts to respect:**
- No API or functionality changes
- Only branding/text updates

**Acceptance criteria:**
- [ ] App header shows "AI Playground" (or chosen name)
- [ ] Browser tab title updated
- [ ] Backend startup message updated
- [ ] README reflects new name

---

### TASK-011: Hyperparameter Presets

**Priority**: Low
**Complexity**: Simple
**Dependencies**: None

**Description:**
Add preset configurations for common use cases (Fast Training, Stable Training, Max Performance).

**Files to modify:**
- `backend/agents/config.py` (add preset configs)
- `src/App.jsx` (add preset selector in Configuration tab)

**Presets to add:**
1. **Fast Training** (DQN):
   - Lower buffer size, higher epsilon decay
   - Smaller network
2. **Stable Training** (DQN):
   - Default config
3. **Max Performance** (DQN):
   - Larger network, more training steps

**Acceptance criteria:**
- [ ] Preset dropdown in Configuration tab
- [ ] Selecting preset populates all fields
- [ ] User can still manually adjust after selecting preset

---

### TASK-012: Model Export/Import

**Priority**: Low
**Complexity**: Medium
**Dependencies**: None

**Description:**
Add ability to export a model as a single file (zip) and import it on another machine.

**Files to modify:**
- `backend/App.py` (add export/import endpoints)
- `backend/storage/storage_manager.py` (add export/import methods)
- `src/App.jsx` (add export/import buttons in Models tab)

**Export format:**
- Zip file containing: `config.json`, `weights.pt`, `metrics.jsonl`, `replays/`

**New endpoints:**
- `GET /api/models/<env>/<run_id>/export` → Download zip file
- `POST /api/models/import` → Upload zip file, extract, register model

**Acceptance criteria:**
- [ ] Export button downloads zip file
- [ ] Import button accepts zip file
- [ ] Imported model appears in Models tab
- [ ] Imported model can be loaded and used

---

## Task Dependencies Graph

```
TASK-001 (SnakeBoard)
    └──> TASK-007 (Replay Viewer) [requires SnakeBoard]

TASK-002 (Model Loading) [independent]

TASK-003 (Device UI) [independent]

TASK-004 (Network Builder) [independent]

TASK-005 (Reward Designer) [independent]

TASK-006 (Input Selector) [independent]

TASK-007 (Replay Viewer)
    └──> TASK-001 (SnakeBoard) [dependency]

TASK-008 (Metrics Charts) [independent]

TASK-009 (Death Analytics) [independent]

TASK-010 (Rename App) [independent, can be done anytime]

TASK-011 (Presets) [independent]

TASK-012 (Export/Import) [independent]
```

---

## How to Claim a Task

When starting work on a task:

1. **Announce intent**: Comment on the issue/PR or notify the team
2. **Check dependencies**: Ensure prerequisite tasks are complete
3. **Read contracts**: Review `docs/ARCHITECTURE.md` for relevant interfaces
4. **Create branch**: `git checkout -b task-NNN-short-description`
5. **Implement**: Follow the task spec
6. **Test**: Ensure app runs and feature works
7. **Commit**: Descriptive commit messages
8. **PR**: Create pull request with task ID in title

---

## Questions?

If a task is unclear or you encounter issues:
- Check `docs/ARCHITECTURE.md` for interface definitions
- Check `backend/README.md` or `src/README.md` for module-specific details
- Ask for clarification before making assumptions
