# SnakeML - Machine Learning Playground

An interactive web application for playing Snake and watching AI agents learn through reinforcement learning. Train DQN and PPO agents in real-time, visualize their progress, and save/load trained models.

## Features

- **Play Snake**: Use arrow keys to control the snake manually
- **Train AI Agents**: Create and train DQN or PPO agents
- **Real-time Visualization**: Watch the AI learn in real-time or train at maximum speed
- **Model Management**: Save, load, and manage trained models
- **Configurable Settings**: Adjust grid size, colors, and hyperparameters

## Architecture

```
SnakeML/
├── src/                    # React Frontend
│   ├── App.jsx             # Main application component
│   ├── components/         # Reusable UI components
│   │   ├── AISettings.jsx  # AI agent controls
│   │   ├── ErrorBoundary.jsx
│   │   ├── Sidebar.jsx
│   │   └── SpeedControl.jsx
│   ├── context/            # React Context providers
│   │   ├── SocketContext.jsx
│   │   ├── GameContext.jsx
│   │   └── AgentContext.jsx
│   ├── games/              # Game-specific UI
│   │   └── snake/
│   └── utils/              # Utilities
│       └── validation.js
├── server/                 # Flask Backend
│   ├── app.py              # SocketIO server
│   ├── session.py          # Game session management
│   ├── players.py          # Player abstractions
│   └── validation.py       # Input validation
├── games/                  # Game Environments
│   ├── base.py             # Abstract game interface
│   └── snake.py            # Snake game implementation
├── networks/               # Neural Networks
│   ├── base.py             # Abstract network interface
│   ├── dqn.py              # Deep Q-Network
│   ├── ppo.py              # Proximal Policy Optimization
│   └── replay_buffer.py    # Experience buffers
├── models/                 # Saved model weights
└── tests/                  # Test suite
```

## Setup

### Prerequisites

- Node.js 18+
- Python 3.11+
- PyTorch

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SnakeML-webpage
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install flask flask-socketio flask-cors eventlet torch numpy
   ```

   Or using conda:
   ```bash
   conda env create -f environment.yml
   conda activate snakeml
   ```

### Running the Application

1. **Start the backend** (in one terminal)
   ```bash
   python -m server.app
   ```
   The server runs on `http://127.0.0.1:5000`

2. **Start the frontend** (in another terminal)
   ```bash
   npm run dev
   ```
   The app opens at `http://localhost:3000`

## Usage

### Playing Manually

1. Click "Play Human"
2. Use arrow keys to control the snake
3. Eat food to grow and score points
4. Avoid walls and your own tail

### Training AI

1. Click "+ New Agent" in the right sidebar
2. Enter a name and select network type (DQN or PPO)
3. Adjust hyperparameters if desired
4. Click "Train" to start training
5. Use the speed slider to adjust training speed (0 = max speed)
6. Click "Save Model" to save the trained agent

### Hyperparameters

**DQN (Deep Q-Network)**
| Parameter | Description | Default |
|-----------|-------------|---------|
| Buffer Size | Experience replay buffer capacity | 10,000 |
| Batch Size | Training batch size | 128 |
| Gamma | Discount factor (0-1) | 0.9 |
| Epsilon Decay | Exploration decay rate | 0.999 |

**PPO (Proximal Policy Optimization)**
| Parameter | Description | Default |
|-----------|-------------|---------|
| Buffer Size | Rollout buffer size | 1,000 |
| Batch Size | Training batch size | 128 |
| Gamma | Discount factor (0-1) | 0.99 |
| Entropy Decay | Steps for entropy to decay | 1,000 |
| PPO Epochs | Training epochs per update | 8 |

**MANN (Mixture of Experts)**
| Parameter | Description | Default |
|-----------|-------------|---------|
| Buffer Size | Rollout buffer size | 1,000 |
| Batch Size | Training batch size | 128 |
| Gamma | Discount factor (0-1) | 0.99 |
| Entropy Decay | Steps for entropy to decay | 1,000 |
| Training Epochs | Epochs per update | 8 |
| Num Experts | Number of expert networks (2-8) | 4 |

MANN uses a Mixture of Experts architecture where multiple expert networks are dynamically blended based on the current game state. A gating network learns which expert to trust for different situations, allowing specialization (e.g., one expert for exploration, another for precise maneuvering).

## Testing

Run the test suite:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_snake_env.py -v
pytest tests/test_validation.py -v
pytest tests/test_networks.py -v
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000,http://127.0.0.1:3000` |

Example:
```bash
CORS_ORIGINS=http://localhost:3000,https://myapp.com python -m server.app
```

## Tech Stack

**Frontend**
- React 18
- Vite
- Socket.IO Client

**Backend**
- Flask
- Flask-SocketIO
- Flask-CORS

**Machine Learning**
- PyTorch
- NumPy

## License

MIT
