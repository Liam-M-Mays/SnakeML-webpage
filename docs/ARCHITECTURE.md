# SnakeML Architecture Documentation

## Overview

SnakeML is a web-based reinforcement learning playground where users can train AI agents to play Snake. The application consists of a Python/Flask backend for RL training and a React frontend for visualization and configuration.

## Stack

- **Backend**: Python, Flask, Flask-SocketIO, PyTorch
- **Frontend**: React, Vite, Socket.IO Client
- **Communication**: REST API + WebSocket (Socket.IO)

## Directory Structure

```
SnakeML-webpage/
├── backend/
│   ├── App.py              # Flask app, Socket.IO handlers, REST API
│   ├── Session.py          # Training session management
│   ├── GameEngine.py       # Snake game environment
│   ├── QNet.py             # DQN agent implementation
│   ├── PPO.py              # PPO agent implementation
│   └── utils/
│       └── device.py       # Device management utilities
├── src/
│   ├── App.jsx             # Main React component
│   ├── App.css             # Application styles
│   ├── CustomComp.jsx      # UI components (GameSettings, GameBoard, etc.)
│   └── index.jsx           # React entry point
├── docs/
│   └── ARCHITECTURE.md     # This file
└── package.json
```

## Backend Architecture

### App.py - Main Application

The Flask application exposes:

#### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/device` | GET | Get device configuration and status |
| `/api/device/set` | POST | Set device preference |

**Device API Request/Response:**

```json
// POST /api/device/set
// Request:
{ "device": "auto" | "cpu" | "cuda" | "mps" }

// Response (both GET and POST):
{
  "preference": "auto",
  "forced": false,
  "force_device": null,
  "resolved": "cpu",
  "available": ["cpu"],
  "name": "CPU",
  "fallback": false
}
```

#### Socket.IO Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `init` | Client → Server | Initialize game session |
| `step` | Client → Server | Manual step (human mode) |
| `AI_step` | Client → Server | Single AI step |
| `AI_loop` | Client → Server | Start continuous AI training |
| `stop_loop` | Client → Server | Stop AI training loop |
| `reset_game` | Client → Server | Reset game state |
| `game_update` | Server → Client | Game state update |
| `set_highscore` | Server → Client | New highscore notification |

**`game_update` Payload:**
```json
{
  "score": 5,
  "food_position": {"x": 3, "y": 7},
  "snake_position": [{"x": 5, "y": 5}, {"x": 4, "y": 5}],
  "game_over": false,
  "episode": 42
}
```

### Session.py - Training Session

Manages individual training sessions:
- Creates game environment and AI agent
- Routes actions through the selected agent
- Tracks highscore and episode count
- Uses resolved device from `utils/device.py`

### Device Management (utils/device.py)

Device selection priority (when preference is "auto"):
1. MPS (Apple Silicon) - if available
2. CUDA (NVIDIA GPU) - if available
3. CPU - always available

Key functions:
- `set_device_preference(pref)` - Set preference ("auto", "cpu", "cuda", "mps")
- `get_device_preference()` - Get current preference
- `get_device_info()` - Get full device status
- `get_device()` / `resolve_device()` - Get actual torch.device

Environment variable `FORCE_DEVICE` overrides user preference.

### AI Agents

#### QNet (DQN)
- Double DQN with dueling architecture
- Experience replay buffer
- Epsilon-greedy exploration with decay

#### PPO
- Actor-critic architecture
- GAE (Generalized Advantage Estimation)
- Entropy regularization with decay

## Frontend Architecture

### App.jsx - Main Component

State management:
- Game state (score, snake, food, episode, highscore)
- Configuration (gridSize, boardSize, gameSpeed, colors)
- Control mode (human, qnet, ppo)
- Device info (fetched from backend)

Key behaviors:
- **Human mode**: Arrow keys control snake, game_over triggers reset
- **AI mode**: Backend handles episode resets, frontend only displays updates

### CustomComp.jsx - UI Components

#### DeviceSelector
- Fetches device info from `/api/device` on mount
- Updates preference via `/api/device/set`
- Shows current device and fallback status

#### GameSettings
- Tabbed interface: Board, Colors, Speed, Device
- Control mode selection
- "Start Training" button (AI modes only)

#### GameBoard
- Renders snake and food on grid
- Start/Pause/Reset controls

## Training Lifecycle

1. User configures settings (grid size, AI algorithm, hyperparameters, device)
2. User clicks "Start Training"
3. Frontend sends `init` event with configuration
4. Backend creates Session with agent on selected device
5. For AI training:
   - If gameSpeed > 0: Frontend polls via `AI_step` at interval
   - If gameSpeed = 0: `AI_loop` runs continuous training on backend
6. Backend emits `game_update` events (throttled in loop mode)
7. User clicks "Pause" → Frontend sends `stop_loop`

## Important Notes

### AI Training and game_over
During AI training, `game_over=true` signals end of an episode, NOT end of training.
The backend handles episode resets internally. The frontend should NOT call `resetGame()`
when `game_over=true` in AI mode - this would stop training prematurely.

### Device Selection
Device preference is set at the backend level and persists for the server's lifetime.
Changing device while training is running won't affect the current session - the new
device will be used for the next session.

### Socket.IO Event Dependencies
- Frontend re-registers listeners when `isRunning` or `controlMode` changes
- Always send `stop_loop` before starting a new training mode

## Future Considerations

- Visual network architecture builder (planned)
- Replay viewer (planned)
- Additional games beyond Snake
- Model save/load functionality
