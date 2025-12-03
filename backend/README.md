# Backend Documentation

This is the backend for the AI/RL Playground, built with Flask, SocketIO, and PyTorch.

## Directory Structure

```
backend/
├── App.py                    # Main Flask application (REST API + SocketIO)
├── training_manager.py       # TrainingSession orchestration
├── envs/                     # Game environments
│   ├── __init__.py           # Environment registry and factory
│   ├── base.py               # BaseEnvironment interface
│   └── snake_env.py          # Snake game implementation
├── agents/                   # RL algorithms
│   ├── __init__.py
│   ├── config.py             # Training configuration schemas
│   ├── network_builder.py    # Neural network architecture builder
│   ├── dqn_agent.py          # DQN/Double DQN implementation
│   └── ppo_agent.py          # PPO implementation
├── storage/                  # Persistence layer
│   ├── __init__.py
│   └── storage_manager.py    # Model/metrics/replay storage
└── utils/                    # Utilities
    ├── __init__.py
    └── device.py             # CPU/CUDA/MPS device selection
```

## Core Modules

### App.py

**Purpose**: Flask application providing REST API and SocketIO endpoints

**Key responsibilities:**
- HTTP endpoints for configuration, training control, model management
- SocketIO events for real-time training metrics streaming
- Global state management (current training session)

**Important endpoints:**
- `POST /api/training/start` — Start training with RunConfig
- `POST /api/training/stop` — Stop current training
- `GET /api/models` — List saved models
- `GET /api/metrics/<env>/<run_id>` — Get training metrics
- `GET /api/device` — Get device info

**SocketIO events emitted:**
- `training_progress` — Real-time progress updates (throttled to 500ms in max speed)
- `episode_summary` — Episode completion with full stats

**To add a new endpoint:**
1. Add route decorator and function in App.py
2. Update `src/utils/api.js` with corresponding client method
3. Document in `docs/ARCHITECTURE.md`

---

### training_manager.py

**Purpose**: Orchestrates training sessions in background threads

**Key class: TrainingSession**

**Lifecycle:**
1. `__init__(config, socketio)` — Initialize environment, agent, storage
2. `start(max_speed)` — Launch training loop in background thread
3. `_train_loop()` — Main training loop (runs until stopped)
4. `stop()` — Signal stop and save model
5. `get_status()` — Get current training state

**Training loop flow:**
1. Select action from agent
2. Step environment
3. Store transition in agent buffer
4. Train agent
5. Emit progress via SocketIO
6. Save metrics to storage
7. Record replays (sampled)

**To modify training loop:**
- Edit `_train_loop()` method
- Ensure thread safety (use locks if accessing shared state)
- Keep SocketIO emissions throttled (avoid flooding frontend)

---

### envs/

**Purpose**: Game environment implementations

**Key files:**
- `base.py` — `BaseEnvironment` abstract class (interface)
- `snake_env.py` — Snake game implementation
- `__init__.py` — Environment registry and factory functions

**Adding a new environment:**

1. Create `envs/your_env.py`:
   ```python
   from .base import BaseEnvironment

   class YourEnv(BaseEnvironment):
       def reset(self, **kwargs):
           # Initialize game state
           return observation

       def step(self, action):
           # Execute action, update state, calculate reward
           return observation, reward, done, info

       def render_state(self):
           # Return JSON-serializable state for frontend
           return {"game_state": ...}

       def get_observation_space(self):
           return {"type": "box", "shape": [obs_size], ...}

       def get_action_space(self):
           return {"type": "discrete", "n": num_actions, ...}

       def get_metadata(self):
           return {"display_name": "Your Game", ...}
   ```

2. Register in `envs/__init__.py`:
   ```python
   from .your_env import YourEnv

   ENV_REGISTRY = {
       "snake": SnakeEnv,
       "your_env": YourEnv,  # Add here
   }
   ```

3. Add default config in `agents/config.py`

4. Create frontend visualization component (see `docs/AGENT_TASKS.md`)

**Environment contract:**
- `reset()` must return initial observation matching `get_observation_space()`
- `step(action)` must return (observation, reward, done, info)
- `render_state()` must return JSON-serializable dict
- Observations and actions must match declared spaces

---

### agents/

**Purpose**: RL algorithm implementations

**Key files:**
- `config.py` — Configuration schemas and defaults
- `network_builder.py` — Builds PyTorch networks from config
- `dqn_agent.py` — DQN/Double DQN with replay buffer
- `ppo_agent.py` — PPO with rollout buffer

**Agent interface (informal):**
```python
class Agent:
    def select_action(self, state, training=True):
        """Select action (and optionally return log_prob, value for PPO)"""

    def store_transition(self, ...):
        """Store experience in buffer"""

    def train_step(self) -> Dict:
        """Perform one training update, return metrics dict"""

    def save(self, path: str):
        """Save model weights to path"""

    def load(self, path: str):
        """Load model weights from path"""
```

**Adding a new algorithm:**

1. Create `agents/your_algo_agent.py`
2. Implement agent interface
3. Add default config in `config.py`:
   ```python
   def get_default_your_algo_config():
       return {
           "algo": "your_algo",
           "hyperparams": {...},
           ...
       }
   ```
4. Update `training_manager.py` to instantiate your agent:
   ```python
   elif algo == "your_algo":
       self.agent = YourAlgoAgent(...)
   ```

**Network config format:**
```python
{
    "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 256, "activation": "relu"},
    ]
}
```

Supported activations: `relu`, `leaky_relu`, `tanh`, `sigmoid`, `linear`

---

### storage/

**Purpose**: Persist models, metrics, and replays to disk

**Key file: storage_manager.py**

**Storage layout:**
```
models/
  <env_name>/
    <run_id>/
      config.json       # RunConfig + metadata
      weights.pt        # PyTorch state dict
      metrics.jsonl     # Training metrics (one JSON per line)
      replays/
        <replay_id>.json  # Replay data
```

**StorageManager API:**

**Config:**
- `save_config(env_name, run_id, config)` — Save run configuration
- `load_config(env_name, run_id)` — Load run configuration

**Weights:**
- `get_weights_path(env_name, run_id)` — Get path to weights file
- `weights_exist(env_name, run_id)` — Check if weights exist

**Metrics:**
- `append_metrics(env_name, run_id, metrics)` — Append metrics (JSONL format)
- `load_metrics(env_name, run_id, limit=None)` — Load metrics (optionally limit to last N)

**Replays:**
- `save_replay(env_name, run_id, replay_id, data)` — Save replay
- `load_replay(env_name, run_id, replay_id)` — Load specific replay
- `list_replays(env_name, run_id)` — List all replays with metadata

**Runs:**
- `list_runs(env_name=None)` — List all runs (optionally filtered by env)
- `delete_run(env_name, run_id)` — Delete entire run

**Analytics:**
- `get_death_stats(env_name, run_id)` — Get death reason statistics

**Usage:**
```python
from storage import get_storage_manager

storage = get_storage_manager()
storage.save_config("snake", "run-123", config)
metrics = storage.load_metrics("snake", "run-123", limit=100)
```

---

### utils/

**Purpose**: Utility functions

**Key file: device.py**

**Device selection:**

Priority:
1. `FORCE_DEVICE` environment variable (`cpu`, `cuda`, `mps`)
2. MPS (Apple Silicon) if available
3. CUDA (NVIDIA GPU) if available
4. CPU fallback

**API:**
- `get_device()` — Auto-select best device
- `get_device_info()` — Get device details (type, name, memory)
- `set_global_device(device=None)` — Set global device
- `get_global_device()` — Get global device (lazily initialized)

**Usage:**
```python
from utils.device import get_global_device

device = get_global_device()
model = MyModel().to(device)
tensor = torch.tensor([1, 2, 3], device=device)
```

**Environment variable override:**
```bash
FORCE_DEVICE=cpu python App.py
FORCE_DEVICE=cuda python App.py
```

**Checking available device:**
```bash
curl http://127.0.0.1:5000/api/device
# Returns: {"device": "cuda", "type": "cuda", "name": "NVIDIA ...", ...}
```

---

## Running the Backend

### Development Mode

```bash
cd backend
python App.py
```

Backend runs on: `http://127.0.0.1:5000`

### Production Mode

For production deployment, use a WSGI server (e.g., gunicorn):

```bash
pip install gunicorn eventlet
gunicorn -k eventlet -w 1 -b 127.0.0.1:5000 App:app
```

**Note:** Use `-w 1` (single worker) because training state is in-memory.

---

## Configuration

### Environment Variables

- `FORCE_DEVICE` — Override device selection (`cpu`, `cuda`, `mps`)

### Default Ports

- Backend: `127.0.0.1:5000`
- Frontend: `localhost:5173` (Vite dev server)

### CORS

CORS is enabled for all origins in development. For production, restrict to frontend domain:

```python
CORS(app, resources={r"/*": {"origins": "https://your-frontend-domain.com"}})
```

---

## Testing

### Manual Testing

1. Start backend: `python App.py`
2. Check health: `curl http://127.0.0.1:5000/api/health`
3. List environments: `curl http://127.0.0.1:5000/api/environments`
4. Get default config: `curl http://127.0.0.1:5000/api/config/default/dqn`

### Unit Testing (TODO)

Unit tests should be added in `backend/tests/`:
- `tests/test_envs.py` — Environment interface tests
- `tests/test_agents.py` — Agent behavior tests
- `tests/test_storage.py` — Storage manager tests

---

## Common Issues

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Ensure you're in the backend directory
cd backend
python App.py

# Or add parent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python backend/App.py
```

### CUDA Out of Memory

If training crashes with OOM:
- Reduce `batch_size` in hyperparams
- Reduce network size in `network_config`
- Use `FORCE_DEVICE=cpu` if GPU memory is limited

### SocketIO Connection Issues

If frontend can't connect to SocketIO:
- Check CORS settings in `App.py`
- Verify backend is running on `127.0.0.1:5000`
- Check browser console for connection errors

---

## Future Improvements

See `docs/AGENT_TASKS.md` for specific development tasks.

**Backend enhancements:**
- Model loading/resuming (TASK-002)
- Device selection API endpoint (TASK-003)
- Event-based reward formulas (TASK-005)
- Multi-session support (allow multiple concurrent training runs)
- Model export/import (TASK-012)

**See also:**
- `docs/ARCHITECTURE.md` — Overall system architecture
- `docs/AGENT_TASKS.md` — Specific development tasks
