# Architecture Overview

This document describes the architecture of the AI/RL Playground (formerly SnakeML), designed to support multiple games and RL algorithms in a modular, extensible way.

## Design Philosophy

1. **Modular**: Easy to add new games, algorithms, and features without breaking existing code
2. **Contract-based**: Clear interfaces between backend/frontend and between modules
3. **Parallel development**: Multiple agents can work on different features simultaneously
4. **Device-agnostic**: Automatic CPU/CUDA/MPS selection with override capability

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (React)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Configure │  │Dashboard │  │  Models  │  │ Replays  │   │
│  │   Tab    │  │   Tab    │  │   Tab    │  │   Tab    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         │              │              │              │       │
│         └──────────────┴──────────────┴──────────────┘       │
│                          │                                    │
│                    API Client (api.js)                       │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP/REST + SocketIO
                            │
┌───────────────────────────┴──────────────────────────────────┐
│                    Backend (Flask + Python)                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               App.py (Flask + SocketIO)                │  │
│  │         REST API + Real-time Training Events           │  │
│  └────────────────────────────────────────────────────────┘  │
│         │              │              │              │        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   envs   │  │  agents  │  │ storage  │  │  utils   │    │
│  │          │  │          │  │          │  │          │    │
│  │SnakeEnv  │  │DQN Agent │  │  Models  │  │  Device  │    │
│  │(future:  │  │PPO Agent │  │ Metrics  │  │ Selection│    │
│  │Tetris,   │  │(future:  │  │ Replays  │  │          │    │
│  │Connect4) │  │A3C, SAC) │  │          │  │          │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│         │              │              │                       │
│  ┌──────────────────────────────────────┐                    │
│  │      TrainingManager                 │                    │
│  │  (Orchestrates training loop)        │                    │
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## Backend Architecture

### Directory Structure

```
backend/
├── App.py                    # Flask app with REST + SocketIO endpoints
├── training_manager.py       # TrainingSession orchestration
├── envs/
│   ├── __init__.py           # Environment registry and factory
│   ├── base.py               # BaseEnvironment interface
│   ├── snake_env.py          # Snake implementation
│   └── [future: tetris_env.py, connect4_env.py, ...]
├── agents/
│   ├── __init__.py
│   ├── config.py             # Training configuration schemas
│   ├── network_builder.py    # Network architecture builder
│   ├── dqn_agent.py          # DQN/Double DQN implementation
│   ├── ppo_agent.py          # PPO implementation
│   └── [future: a3c_agent.py, sac_agent.py, ...]
├── storage/
│   ├── __init__.py
│   └── storage_manager.py    # Model/metrics/replay persistence
└── utils/
    ├── __init__.py
    └── device.py             # CPU/CUDA/MPS device selection
```

### Core Contracts

#### 1. Environment Interface (BaseEnvironment)

**File**: `backend/envs/base.py`

All game environments must implement:

```python
class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, **kwargs) -> Any:
        """Reset environment, return initial observation"""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take action, return (observation, reward, done, info)"""

    @abstractmethod
    def render_state(self) -> Dict:
        """Return JSON-serializable state for frontend visualization"""

    @abstractmethod
    def get_observation_space(self) -> Dict:
        """Describe observation space (shape, type, etc.)"""

    @abstractmethod
    def get_action_space(self) -> Dict:
        """Describe action space (discrete/continuous, size, etc.)"""

    def get_metadata(self) -> Dict:
        """Return env metadata (name, description, config options)"""
```

**Adding a new environment:**

1. Create `backend/envs/your_env.py`
2. Implement `BaseEnvironment`
3. Register in `backend/envs/__init__.py`:
   ```python
   ENV_REGISTRY = {
       "snake": SnakeEnv,
       "your_env": YourEnv,  # Add here
   }
   ```

#### 2. Agent Interface

**File**: `backend/agents/` (pattern used by DQNAgent and PPOAgent)

All RL agents should implement:

```python
class BaseAgent:
    def select_action(self, state, training=True):
        """Select action given state (may return action + extras like log_prob)"""

    def store_transition(self, *args):
        """Store experience in buffer"""

    def train_step(self) -> Dict:
        """Perform one training update, return metrics dict"""

    def save(self, path):
        """Save model weights"""

    def load(self, path):
        """Load model weights"""
```

#### 3. Run Configuration (RunConfig)

**File**: `backend/agents/config.py`

A training run is defined by:

```python
{
    "run_id": str,              # Unique identifier
    "algo": str,                # "dqn", "ppo", etc.
    "env_name": str,            # "snake", etc.
    "env_config": {             # Environment-specific config
        "grid_size": int,
        "vision": int,          # Vision mode
        "seg_size": int,        # Body segment tracking
        ...
    },
    "reward_config": {          # Reward shaping
        "apple": float,
        "death_wall": float,
        "death_self": float,
        "death_starv": float,
        "step": float,
        # TODO: Event-based rewards with formulas
    },
    "network_config": {         # Neural network architecture
        "layers": [
            {"type": "dense", "units": int, "activation": str},
            ...
        ]
    },
    "hyperparams": {            # Algorithm hyperparameters
        "learning_rate": float,
        "gamma": float,
        "batch_size": int,
        ...
    }
}
```

#### 4. API Endpoints

**File**: `backend/App.py`

All endpoints follow RESTful conventions:

**Environments:**
- `GET /api/environments` → List all environments
- `GET /api/environments/<env_name>/metadata` → Get env details

**Configuration:**
- `GET /api/config/default/<algo>` → Get default config for algorithm

**Training:**
- `POST /api/training/start` (body: RunConfig) → Start training session
- `POST /api/training/stop` → Stop current session
- `GET /api/training/status` → Get current status

**Models:**
- `GET /api/models?env_name=<env>` → List saved models
- `GET /api/models/<env>/<run_id>/config` → Get model config
- `DELETE /api/models/<env>/<run_id>` → Delete model

**Metrics:**
- `GET /api/metrics/<env>/<run_id>?limit=<n>` → Get training metrics
- `GET /api/metrics/<env>/<run_id>/death_stats` → Get death statistics

**Replays:**
- `GET /api/replays/<env>/<run_id>` → List replays
- `GET /api/replays/<env>/<run_id>/<replay_id>` → Get replay data
- `DELETE /api/replays/<env>/<run_id>/<replay_id>` → Delete replay

**System:**
- `GET /api/device` → Get device info (CPU/CUDA/MPS)
- `GET /api/health` → Health check

#### 5. SocketIO Events

**Emitted by server:**

- `training_progress` — Real-time progress updates
  ```python
  {
      "run_id": str,
      "episode": int,
      "total_steps": int,
      "current_score": int,
      "best_score": int,
      "avg_score": float,
      "episodes_per_second": float,
      "is_max_speed": bool
  }
  ```

- `episode_summary` — Emitted when episode ends
  ```python
  {
      "run_id": str,
      "episode": int,
      "score": int,
      "reward": float,
      "length": int,
      "death_reason": str,
      "loss": float,        # (DQN)
      "epsilon": float,     # (DQN)
      "entropy": float,     # (PPO)
      ...
  }
  ```

#### 6. Storage Layout

**File**: `backend/storage/storage_manager.py`

```
models/
  <env_name>/
    <run_id>/
      config.json       # Full RunConfig + metadata
      weights.pt        # PyTorch model weights
      metrics.jsonl     # Training metrics (one JSON per line)
      replays/
        <replay_id>.json  # Replay data (seed + actions)
```

**StorageManager API:**
- `save_config(env_name, run_id, config)`
- `load_config(env_name, run_id) -> Dict`
- `get_weights_path(env_name, run_id) -> Path`
- `append_metrics(env_name, run_id, metrics)`
- `load_metrics(env_name, run_id, limit=None) -> List[Dict]`
- `save_replay(env_name, run_id, replay_id, data)`
- `load_replay(env_name, run_id, replay_id) -> Dict`
- `list_replays(env_name, run_id) -> List[Dict]`
- `list_runs(env_name=None) -> List[Dict]`
- `delete_run(env_name, run_id) -> bool`

#### 7. Device Selection

**File**: `backend/utils/device.py`

Device selection priority:
1. `FORCE_DEVICE` environment variable (`cpu`, `cuda`, `mps`)
2. MPS (Apple Silicon) if available
3. CUDA (NVIDIA GPU) if available
4. CPU fallback

**API:**
- `get_device() -> torch.device` — Select best device
- `get_device_info() -> Dict` — Get device details (type, name, memory)
- `set_global_device(device=None)` — Set global device
- `get_global_device() -> torch.device` — Get global device

**Usage:**
```python
from utils.device import get_global_device

device = get_global_device()
model = model.to(device)
tensor = tensor.to(device)
```

**Environment variable override:**
```bash
FORCE_DEVICE=cuda python backend/App.py
FORCE_DEVICE=cpu python backend/App.py
```

---

## Frontend Architecture

### Directory Structure

```
src/
├── App.jsx               # Main app component with tab navigation
├── App.css               # Global styles
├── index.jsx             # Entry point
├── pages/                # (TODO) Full-page views
│   ├── ConfigurePage.jsx     # Configuration UI
│   ├── DashboardPage.jsx     # Training dashboard + visualization
│   ├── ModelsPage.jsx        # Model management
│   └── ReplaysPage.jsx       # Replay viewer
├── components/           # (TODO) Reusable components
│   ├── GameBoard/            # Game visualization components
│   │   ├── SnakeBoard.jsx        # Snake game board
│   │   └── [future: TetrisBoard.jsx, ...]
│   ├── NetworkBuilder/       # Visual network architecture builder
│   │   └── NetworkBuilder.jsx
│   ├── RewardDesigner/       # Event-based reward configuration
│   │   └── RewardDesigner.jsx
│   ├── InputSelector/        # Input/observation mode selector
│   │   └── InputSelector.jsx
│   ├── Charts/               # Metric visualization
│   │   └── MetricsChart.jsx
│   └── ModelCard/            # Model display card
│       └── ModelCard.jsx
├── hooks/
│   ├── useSocket.js      # SocketIO connection management
│   ├── useTraining.js    # Training state + controls
│   └── (TODO) useReplays.js, useMetrics.js
├── utils/
│   ├── api.js            # Backend API client
│   └── (TODO) types.ts   # TypeScript type definitions
└── types/                # (TODO) TypeScript interfaces
    └── index.ts
```

### Frontend Contracts

#### 1. API Client

**File**: `src/utils/api.js`

All backend communication goes through the `api` object. See file for full interface.

**Key methods:**
- `api.getEnvironments()`
- `api.startTraining(config)`
- `api.stopTraining()`
- `api.listModels(envName?)`
- `api.getMetrics(envName, runId, limit?)`
- `api.listReplays(envName, runId)`
- `api.getDeviceInfo()`

#### 2. Tab Structure

**Current tabs** (in `App.jsx`):
1. **Train** — Training controls + live dashboard
2. **Configuration** — Algorithm, hyperparams, rewards, network
3. **Models** — Saved model management

**Planned tabs:**
4. **Replays** — View and playback recorded episodes

#### 3. Component Contracts (TODO)

The following components need to be implemented by future agents. Their expected interfaces are defined below:

##### SnakeBoard Component

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
  cellSize?: number;  // Default: auto-calculated
}
```

**Behavior:**
- Renders Snake game grid with snake body and food
- Shows score and game over state
- Updates in real-time when `gameState` changes

**Integration:**
- Used in Dashboard tab
- Receives state from `training_progress` SocketIO events via `render_state()` env method

##### NetworkBuilder Component

**File**: `src/components/NetworkBuilder/NetworkBuilder.jsx`

**Props:**
```typescript
interface NetworkBuilderProps {
  networkConfig: {
    layers: Array<{
      type: "dense";
      units: number;
      activation: "relu" | "leaky_relu" | "tanh" | "sigmoid" | "linear";
    }>;
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Behavior:**
- Visual drag-and-drop or list-based network editor
- Add/remove layers
- Adjust units and activation per layer
- Save/load network templates (presets)

**Integration:**
- Used in Configuration tab
- Updates `config.network_config`

##### RewardDesigner Component

**File**: `src/components/RewardDesigner/RewardDesigner.jsx`

**Props:**
```typescript
interface RewardDesignerProps {
  rewardConfig: {
    // Current simple rewards
    apple: number;
    death_wall: number;
    death_self: number;
    death_starv: number;
    step: number;

    // TODO: Event-based rewards
    events?: Array<{
      event: "apple_eaten" | "death" | "step" | ...;
      formula: string;  // e.g., "1.0 + 0.1 * snake_length"
      variables: string[];  // e.g., ["snake_length", "distance_to_food"]
    }>;
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Behavior:**
- Currently: Simple numeric inputs for each event type
- Future: Event-based reward designer with formulas
  - Select event triggers (apple eaten, death, per step, etc.)
  - Access to variables (snake length, distance to food, hunger, etc.)
  - Formula builder (simple math expressions)

**Integration:**
- Used in Configuration tab
- Updates `config.reward_config`

##### InputSelector Component

**File**: `src/components/InputSelector/InputSelector.jsx`

**Props:**
```typescript
interface InputSelectorProps {
  envConfig: {
    vision: number;     // 0=immediate danger, >0=window, -1=full grid
    seg_size: number;   // Body segments to track
    // TODO: More input options
  };
  onChange: (newConfig) => void;
  disabled?: boolean;
}
```

**Behavior:**
- Choose what the network "sees" as input
- Vision modes:
  - Immediate danger (4 directions)
  - Window around head (3×3, 5×5, etc.)
  - Full grid
- Body segment tracking configuration
- Future: Custom feature selection (relationships to tail, etc.)

**Integration:**
- Used in Configuration tab
- Updates `config.env_config.vision` and related fields

---

## Key Workflows

### 1. Configure → Train Flow

**User journey:**

1. User navigates to **Configuration** tab
2. Selects algorithm (DQN/PPO)
3. Configures:
   - Environment settings (grid size, vision, etc.)
   - Rewards (numeric or event-based)
   - Network architecture (visual builder or presets)
   - Hyperparameters (learning rate, gamma, etc.)
   - Device selection (CPU/CUDA/auto) — TODO: Add UI
4. Switches to **Train** tab
5. Optionally enables "Max Speed" mode
6. Clicks "Start Training"

**Backend flow:**

1. `POST /api/training/start` with RunConfig
2. `TrainingSession` created with config
3. Environment and agent instantiated
4. Training loop starts in background thread
5. Emits `training_progress` every step (or throttled in max speed)
6. Emits `episode_summary` on episode end
7. Saves metrics to JSONL file
8. Records replays (sampled)

**Frontend flow:**

1. `useTraining` hook calls `api.startTraining(config)`
2. `useSocket` hook connects to SocketIO
3. Listens for `training_progress` and `episode_summary` events
4. Updates UI with live stats
5. Renders game board (TODO: currently no visualization)
6. Shows episode table with recent episodes

### 2. Stop → Save Flow

**User clicks "Stop Training":**

1. Frontend calls `api.stopTraining()`
2. Backend sets `stop_event`
3. Training loop exits
4. Model weights saved to `models/<env>/<run_id>/weights.pt`
5. Config already saved at session start
6. Metrics already persisted incrementally

**Model now appears in Models tab**

### 3. Load Model Flow (TODO)

**Current state:** Can view and delete models, but cannot load/resume training

**Planned flow:**

1. User navigates to **Models** tab
2. Clicks "Load" on a saved model
3. Frontend fetches config via `api.getModelConfig(env, runId)`
4. Populates Configuration tab with saved config
5. User can:
   - Resume training (load weights + continue)
   - Clone config for new run
   - View/analyze metrics

**Backend changes needed:**
- Add `POST /api/training/resume` endpoint
- Accept `run_id` to load existing weights
- Load weights into agent before starting training

### 4. Replay Viewing Flow (TODO)

**Current state:** Replays saved, but no UI to view them

**Planned flow:**

1. User navigates to **Replays** tab
2. Selects a saved model from dropdown
3. Sees list of replays for that run (sorted by score/timestamp)
4. Clicks on a replay
5. Replay loads:
   - Environment reset with saved seed
   - Actions replayed step-by-step
   - Game board animates the replay
6. Controls: Play/Pause, step forward/back, speed control

**Frontend changes needed:**
- Create `ReplaysPage` component
- Implement replay playback engine
- Reuse `SnakeBoard` (and future game boards) for visualization

---

## Future Architecture Extensions

### Multi-Game Support

**Current:** Only Snake implemented

**To add a new game (e.g., Tetris):**

1. **Backend:**
   - Create `backend/envs/tetris_env.py` implementing `BaseEnvironment`
   - Register in `ENV_REGISTRY`
   - Define observation/action spaces
   - Implement `render_state()` for frontend

2. **Frontend:**
   - Create `src/components/GameBoard/TetrisBoard.jsx`
   - Add game selection dropdown in Configuration tab
   - Conditionally render appropriate game board in Dashboard

### Multi-Algorithm Support

**Current:** DQN and PPO implemented

**To add a new algorithm (e.g., A3C):**

1. Create `backend/agents/a3c_agent.py`
2. Implement agent interface (select_action, store_transition, train_step, save, load)
3. Add default config in `backend/agents/config.py`
4. Update `TrainingSession` to handle new algorithm
5. Frontend: Add to algorithm dropdown (no other changes needed)

### Advanced Reward System

**Current:** Simple numeric rewards per event type

**Planned:**

- Event-based reward designer
- Access to game variables (snake length, distance to food, hunger, etc.)
- Formula support (simple math expressions)
- Backend changes:
  - Extend `reward_config` to support formulas
  - Evaluate formulas in environment step
- Frontend changes:
  - Build `RewardDesigner` component with formula editor

### Visual Network Builder

**Current:** Network config is JSON, displayed as text

**Planned:**

- Drag-and-drop layer builder
- Presets (Small, Medium, Large, Custom)
- Real-time validation (compatible activations, sizes)
- Save/load templates
- Frontend changes:
  - Build `NetworkBuilder` component

### Parallel Training

**Current:** Single training session at a time

**Planned:**

- Multiple concurrent sessions
- Vectorized environments (parallel episodes)
- Multi-GPU support
- Backend changes:
  - Session manager tracking multiple `TrainingSession` instances
  - Resource allocation per session

---

## Development Constraints

When implementing new features, follow these guidelines:

1. **Respect interfaces**: Don't change BaseEnvironment, API endpoints, or SocketIO event shapes without updating ALL dependent code

2. **Add, don't replace**: Prefer adding new components/endpoints over modifying existing ones

3. **Test incrementally**: Ensure app still runs after each change

4. **Document as you go**: Update this file if you add new interfaces or contracts

5. **Use TODO comments**: Mark incomplete features clearly

6. **Avoid breaking changes**: If you must change an interface, provide backwards compatibility or migration path

---

## Questions / Contact

For questions about this architecture:
- See `docs/AGENT_TASKS.md` for specific development tasks
- Check `backend/README.md` and `src/README.md` for module-specific details
