# SnakeML Web Playground

SnakeML is a lightweight reinforcement learning playground for experimenting with Snake agents in the browser. The project pairs a React frontend with a Flask + PyTorch backend and is being refactored to support richer training dashboards, configurable rewards, and eventually multiple games.

## Quick Start

1. **Install dependencies**
   - Python 3.10+
   - Node.js 18+

2. **Run the dev helper**
   ```bash
   python dev.py
   ```
   The script installs Python/Node dependencies (idempotently) and starts:
   - Backend Socket.IO server on `http://127.0.0.1:5000`
   - Frontend Vite dev server on `http://127.0.0.1:3000`

3. Open the frontend URL in your browser and start playing or running the AI.

### Notes for macOS, Windows, and Linux
- The backend auto-detects the best available PyTorch device:
  - Apple Silicon: `mps`
  - NVIDIA GPUs: `cuda`
  - Otherwise: `cpu`
- Override with `FORCE_DEVICE=cpu python dev.py` if you need deterministic CPU-only runs.

## Project Structure
- `backend/` — Flask Socket.IO server, game engine, RL agents, and utilities.
- `src/` — React UI for the Snake playground.
- `dev.py` — One-command launcher for both backend and frontend.
- `docs/upgrade-plan.md` — Ongoing design plan for modular environments, training dashboards, and model management.

## Backend Dependencies
Install them manually (if you skip `dev.py` installation):
```bash
python -m pip install -r requirements.txt
```

## Frontend Scripts
```bash
npm install
npm run dev
npm run build
```

## Roadmap
See `docs/upgrade-plan.md` for the step-by-step refactor plan toward multi-environment RL support, live training dashboards, reward designer, network builder, and replay tooling.
