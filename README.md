# SnakeML Web Playground

SnakeML is a reinforcement learning playground for experimenting with Snake agents. The project now exposes modular environments, configurable rewards, live training metrics via Socket.IO, and replay/model management.

## Quick Start

1. **Install dependencies**
   - Python 3.10+
   - Node.js 18+

2. **Run the dev helper**
   ```bash
   python dev.py
   ```
   The script installs Python/Node dependencies (idempotently) and starts:
   - Backend Socket.IO + REST server on `http://127.0.0.1:5000`
   - Frontend Vite dev server on `http://127.0.0.1:3000`

3. Open the frontend URL and launch a training run.

### Platform notes
- Device detection prefers `FORCE_DEVICE` (cpu/cuda/mps), then MPS on Apple Silicon, CUDA if available, else CPU.
- On macOS, ensure PyTorch is installed with MPS support for best performance.
- Override device selection with `FORCE_DEVICE=cpu python dev.py` if you need CPU-only execution.

## UI Highlights
- **Training Dashboard**: live metrics (reward/length/loss/epsilon) streamed over Socket.IO, plus episodes-per-second progress.
- **Reward Designer**: edit per-event Snake rewards and persist defaults via the backend.
- **Network Builder**: configure dense layers and activations for the policy network.
- **Replays**: list and inspect episode action traces saved under each run.
- **Models**: browse saved runs, reload, or delete artifacts.

## API surface
- Environment metadata: `GET /api/envs` and `/api/envs/<name>/metadata`
- Reward defaults: `GET/POST /api/rewards/<env>`
- Training runs: `POST /api/runs/start`, `GET /api/runs`, metrics/replays/config/death stats under `/api/runs/<env>/<run>`

## Backend
- Flask + Flask-SocketIO, modular environments under `backend/envs`, centralized device detection in `backend/utils/device.py`.
- Training runs persist under `models/<env>/<run_id>/` with `config.json`, `weights.pt`, `metrics.jsonl`, and `replays/`.

Install backend deps manually (if skipping `dev.py`):
```bash
python -m pip install -r requirements.txt
```

## Frontend
- React + Vite app under `src/` with hooks/components for dashboards, rewards, networks, and replays.
- Run manually with:
```bash
npm install
npm run dev
```

