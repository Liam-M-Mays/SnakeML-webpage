# 🐍 SnakeML Playground

A modular reinforcement learning playground for training AI agents to play Snake (and future games). Built with **PyTorch**, **Flask**, and **React**, featuring a modern web UI for configuring, training, and analyzing deep RL agents.

---

## ✨ Features

### 🎮 Environments
- **Snake**: Classic snake game with customizable grid size
- **Modular design**: Easy to add new games (Connect-4, Tetris, etc.)
- **Configurable observations**: Multiple vision modes (immediate danger, window-based, full grid)

### 🧠 Algorithms
- **DQN** (Deep Q-Network): Double DQN with dueling architecture
- **PPO** (Proximal Policy Optimization): Actor-critic with clipped objective
- Easily extensible for new algorithms

### ⚙️ Configuration & Customization
- **Hyperparameter tuning**: Learning rate, gamma, batch size, epsilon decay, and more
- **Network builder**: Configurable network architectures (layers, units, activations)
- **Reward designer**: Customize reward values for apples, deaths, steps, etc.
- **Device detection**: Automatic MPS (Apple Silicon) / CUDA / CPU selection

### 📊 Training Dashboard
- **Live metrics streaming**: Episodes per second, scores, rewards, losses
- **Max speed mode**: Train at maximum speed with periodic UI updates
- **Episode tracking**: View recent episodes with detailed stats (score, reward, length, death reason)
- **Death analytics**: Aggregate statistics on how the agent dies (wall, self, starvation)

### 💾 Model Management
- **Auto-save**: Models automatically saved when training stops
- **Metrics persistence**: All training metrics saved to JSONL files
- **Model library**: Browse, load, and delete saved models
- **Configuration export**: Save and reload complete training configurations

### 🎬 Replay System
- **Episode recording**: Automatically record episodes (seed + action sequence)
- **Replay viewer**: Step through recorded episodes (future feature)
- **Smart sampling**: Save interesting episodes (high scores, milestones)

### 🎨 Modern UI
- **Dark theme**: Easy on the eyes for long training sessions
- **Responsive design**: Works on desktop and laptop screens
- **Real-time updates**: SocketIO-based streaming of training progress
- **Intuitive controls**: Simple start/stop, configuration panels

---

## 🚀 Quick Start

### Prerequisites

**Required:**
- Python 3.11+ (3.11 recommended)
- Node.js 18+ and npm
- pip (Python package manager)

**Platform Notes:**
- **macOS (Apple Silicon)**: MPS acceleration supported
- **Linux/Windows with NVIDIA GPU**: CUDA acceleration supported
- **Any platform**: CPU fallback always available

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Liam-M-Mays/SnakeML-webpage.git
   cd SnakeML-webpage
   ```

2. **Install dependencies** (automatic via dev script)

   The `dev.py` script will install both Python and Node dependencies for you.

3. **Start the development servers**
   ```bash
   python dev.py
   ```

   This will:
   - Install Python dependencies from `requirements.txt`
   - Install Node dependencies from `package.json`
   - Start the Flask backend on `http://127.0.0.1:5000`
   - Start the Vite frontend dev server on `http://localhost:5173`

4. **Open your browser**

   Navigate to `http://localhost:5173` to access the SnakeML Playground.

### Alternative: Manual Installation

If you prefer manual control:

```bash
# Backend
pip install -r requirements.txt

# Frontend
npm install

# Start backend (in one terminal)
cd backend
python App.py

# Start frontend (in another terminal)
npm run dev
```

### Optional: Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda env create -f environment.yml
conda activate snakeppo
```

---

## 📖 Usage Guide

### Starting a Training Run

1. **Navigate to the Configuration tab**
   - Choose an algorithm: DQN or PPO
   - Adjust environment settings (grid size, etc.)
   - Customize rewards for different events
   - Modify hyperparameters (learning rate, gamma, batch size, etc.)

2. **Go to the Train tab**
   - Optionally enable "Max Speed" for faster training (less visualization)
   - Click "Start Training"

3. **Monitor progress**
   - Watch live statistics: episode count, current score, best score, average score
   - View recent episodes in the table with detailed metrics
   - Training metrics are automatically saved

4. **Stop training**
   - Click "Stop Training" when satisfied
   - Model weights are automatically saved to `models/<env>/<run_id>/`

### Understanding the Dashboard

- **Episode**: Current episode number
- **Current Score**: Score in the current ongoing episode
- **Best Score**: Highest score achieved during this run
- **Avg Score**: Average score over the last 100 episodes
- **Reward**: Cumulative reward for an episode
- **Length**: Number of steps in an episode
- **Death Reason**: How the episode ended (wall, self, starvation)
- **Loss**: Training loss (DQN) or policy/value loss (PPO)
- **Epsilon / Entropy**: Exploration parameter (DQN uses epsilon, PPO uses entropy)

### Max Speed Mode

When enabled:
- Trains as fast as possible (no artificial delays)
- UI updates every 500ms instead of every step
- Shows episodes per second
- Use this for efficient training; disable for visualization

### Managing Models

**Models Tab:**
- View all saved models with metadata (algorithm, best score, avg reward, total episodes)
- Delete models you no longer need
- Load model configurations to resume training or compare settings

**Storage Structure:**
```
models/
  snake/
    <run_id>/
      config.json       # Full run configuration
      weights.pt        # PyTorch model weights
      metrics.jsonl     # Training metrics (one JSON object per line)
      replays/          # Recorded episodes
        <replay_id>.json
```

---

## 🏗️ Architecture

### Backend (Python)

```
backend/
├── App.py                    # Main Flask application with REST API + SocketIO
├── training_manager.py       # Training loop orchestration
├── envs/
│   ├── base.py               # BaseEnvironment interface
│   ├── snake_env.py          # Snake environment implementation
│   └── __init__.py           # Environment registry
├── agents/
│   ├── config.py             # Training configuration system
│   ├── network_builder.py    # Configurable network architectures
│   ├── dqn_agent.py          # DQN implementation
│   └── ppo_agent.py          # PPO implementation
├── storage/
│   └── storage_manager.py    # Model/metrics/replay persistence
└── utils/
    └── device.py             # MPS/CUDA/CPU device detection
```

**Key Design Principles:**
- **Environment abstraction**: Easy to add new games by implementing `BaseEnvironment`
- **Agent modularity**: Algorithms separated from game logic
- **Device agnostic**: Automatically selects best available device
- **Streaming architecture**: Real-time metrics via SocketIO

### Frontend (React)

```
src/
├── App.jsx                   # Main application component
├── App.css                   # Styles
├── index.jsx                 # Entry point
├── hooks/
│   ├── useSocket.js          # SocketIO connection hook
│   └── useTraining.js        # Training state management hook
└── utils/
    └── api.js                # Backend API client
```

**UI Structure:**
- **Train Tab**: Start/stop training, view live dashboard
- **Configuration Tab**: Adjust all training parameters
- **Models Tab**: Manage saved models

---

## 🔧 Configuration Reference

### DQN Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | Optimizer learning rate |
| `gamma` | 0.9 | Discount factor for future rewards |
| `batch_size` | 128 | Number of experiences per training batch |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.1 | Minimum exploration rate |
| `epsilon_decay` | 0.999 | Epsilon decay rate per episode |
| `buffer_size` | 10000 | Replay buffer capacity |
| `target_update_freq` | 50 | Episodes between target network updates |

### PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | Optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `batch_size` | 128 | Mini-batch size for updates |
| `buffer_size` | 1000 | Rollout buffer size (steps before update) |
| `clip_range` | 0.15 | PPO clipping parameter |
| `value_coef` | 0.5 | Value loss coefficient |
| `entropy_coef_start` | 0.05 | Initial entropy bonus |
| `entropy_coef_end` | 0.01 | Final entropy bonus |
| `entropy_decay_steps` | 1000 | Steps to decay entropy coefficient |
| `n_epochs` | 8 | Epochs per PPO update |

### Reward Configuration

| Event | Default | Description |
|-------|---------|-------------|
| `apple` | 1.0 | Reward for eating food |
| `death_wall` | -1.0 | Penalty for hitting wall |
| `death_self` | -1.0 | Penalty for hitting self |
| `death_starv` | -0.5 | Penalty for starvation |
| `step` | -0.001 | Small penalty per step (encourages efficiency) |

**Design Philosophy:**
- Positive reward for goal (eating apple)
- Negative reward for failure (deaths)
- Small negative reward per step to encourage speed
- Adjust these to shape agent behavior

### Network Architecture

Default (Medium):
```python
{
  "layers": [
    {"type": "dense", "units": 128, "activation": "relu"},
    {"type": "dense", "units": 256, "activation": "relu"},
    {"type": "dense", "units": 128, "activation": "relu"},
  ]
}
```

Supported activations: `relu`, `leaky_relu`, `tanh`, `sigmoid`, `linear`

**Tips:**
- Larger networks: Better capacity, slower training, risk of overfitting
- Smaller networks: Faster, more stable, may underfit
- For Snake, 2-3 hidden layers with 128-256 units works well

---

## 🎯 Tips for Training

### Getting Started
1. **Use default configs first**: They're tuned to work reasonably well
2. **Start with DQN**: Simpler than PPO, easier to debug
3. **Monitor death reasons**: If agent always hits walls, increase wall death penalty
4. **Use max speed**: Much faster training, check progress periodically

### Improving Performance
1. **Adjust rewards**: Make successful behaviors more rewarding
2. **Tune epsilon decay (DQN)**: Too fast → agent stops exploring; too slow → random for too long
3. **Adjust entropy coefficient (PPO)**: Higher → more exploration
4. **Increase network size**: If agent plateaus quickly
5. **Decrease learning rate**: If training is unstable

### Debugging
- **Agent not learning**: Check reward config, ensure positive rewards are achievable
- **Agent suicidal**: Death penalties might be too small or reward too sparse
- **Training unstable**: Reduce learning rate, increase batch size
- **Agent plateaus**: Increase network size, adjust exploration

---

## 🔮 Future Expansion

The codebase is designed to support:

### New Environments
- **Connect-4**: Two-player turn-based game
- **Tetris**: Continuous action space, different reward structure
- **Custom games**: Implement `BaseEnvironment` interface

### New Algorithms
- **A3C**: Asynchronous actor-critic
- **SAC**: Soft actor-critic for continuous control
- **Rainbow DQN**: Combines multiple DQN improvements
- **Evolutionary strategies**: Population-based training

### Advanced Features
- **Multi-environment parallel training**: Vectorized environments
- **Curriculum learning**: Start easy, gradually increase difficulty
- **Meta-learning**: Train agents to adapt quickly to new tasks
- **Advanced network builder UI**: Drag-and-drop network design
- **Replay viewer**: Visual playback of recorded episodes
- **Experiment comparison**: Side-by-side comparison of runs
- **Hyperparameter optimization**: Automatic tuning

---

## 📚 API Reference

### REST Endpoints

**Environments**
- `GET /api/environments` - List all environments
- `GET /api/environments/<env_name>/metadata` - Get environment details

**Configuration**
- `GET /api/config/default/<algo>` - Get default config for algorithm

**Training**
- `POST /api/training/start` - Start training (body: config JSON)
- `POST /api/training/stop` - Stop training
- `GET /api/training/status` - Get current status

**Models**
- `GET /api/models` - List all models (optional: `?env_name=snake`)
- `GET /api/models/<env>/<run_id>/config` - Get model config
- `DELETE /api/models/<env>/<run_id>` - Delete model

**Metrics**
- `GET /api/metrics/<env>/<run_id>` - Get metrics (optional: `?limit=100`)
- `GET /api/metrics/<env>/<run_id>/death_stats` - Get death statistics

**Replays**
- `GET /api/replays/<env>/<run_id>` - List replays
- `GET /api/replays/<env>/<run_id>/<replay_id>` - Get replay
- `DELETE /api/replays/<env>/<run_id>/<replay_id>` - Delete replay

**System**
- `GET /api/device` - Get device info
- `GET /api/health` - Health check

### SocketIO Events

**Emitted by Server:**
- `training_progress` - Periodic progress updates (episode, scores, speed)
- `episode_summary` - Emitted when episode ends (full episode stats)

---

## 🛠️ Development

### Adding a New Environment

1. Create a new file in `backend/envs/` (e.g., `connect4_env.py`)
2. Implement `BaseEnvironment` interface:
   ```python
   from .base import BaseEnvironment

   class Connect4Env(BaseEnvironment):
       def reset(self): ...
       def step(self, action): ...
       def render_state(self): ...
       def get_observation_space(self): ...
       def get_action_space(self): ...
   ```
3. Register in `backend/envs/__init__.py`:
   ```python
   ENV_REGISTRY = {
       "snake": SnakeEnv,
       "connect4": Connect4Env,  # Add this
   }
   ```

### Adding a New Algorithm

1. Create `backend/agents/<algo>_agent.py`
2. Implement `select_action()`, `store_transition()`, `train_step()`, `save()`, `load()`
3. Add to `training_manager.py` in the agent creation logic
4. Add default config in `backend/agents/config.py`

### Environment Variables

- `FORCE_DEVICE`: Override device selection (`cpu`, `cuda`, or `mps`)

---

## 🐛 Troubleshooting

### Node/npm not found (when running dev.py)

**Symptoms**: `dev.py` reports "Node.js and/or npm not found in PATH"

**Solutions**:
- **macOS**: `brew install node` or download from [nodejs.org](https://nodejs.org/)
- **Windows**: Download installer from [nodejs.org](https://nodejs.org/), ensure "Add to PATH" is checked
- **Linux**: `sudo apt install nodejs npm` (Ubuntu/Debian) or `sudo dnf install nodejs npm` (Fedora)
- **If using nvm**: Run `nvm use` in the project directory

After installing, restart your terminal.

### Training not starting

- Check browser console for errors
- Verify backend is running on `http://127.0.0.1:5000`
- Check backend logs for errors
- Ensure all dependencies are installed

### Training very slow

- Enable "Max Speed" mode
- Reduce batch size or network size
- Check if GPU is being used: look for "Device: CUDA" or "Device: MPS" in backend logs

### Agent not learning

- Check reward configuration: ensure positive rewards are reachable
- Try default config first
- Monitor death reasons: if all deaths are the same type, adjust that penalty
- Ensure sufficient exploration (epsilon for DQN, entropy for PPO)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Flask](https://flask.palletsprojects.com/), [React](https://react.dev/), and [Vite](https://vitejs.dev/)
- Inspired by DeepMind's DQN paper and OpenAI's PPO implementation
- Thanks to the open-source RL community for countless tutorials and resources

---

## 📞 Contact & Contributing

Found a bug? Have a feature request? Want to contribute?

- **Issues**: Open an issue on GitHub
- **Pull Requests**: Contributions welcome!
- **Discussions**: Share your trained models and configurations

Happy training! 🚀🐍
