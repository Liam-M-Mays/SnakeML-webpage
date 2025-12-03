# Upgrade Plan

This document outlines the intended architecture and upcoming work items to evolve the Snake reinforcement learning playground into a multi-environment RL website.

## Device Detection
- Centralize device selection in `backend/utils/device.py` with runtime detection for MPS, CUDA, or CPU.
- Allow override via `FORCE_DEVICE` environment variable for reproducibility or debugging.
- Reuse the helper in training loops, model initialization, and inference endpoints.

## Environment Abstraction
- Introduce an environment registry mapping environment names to environment classes (starting with `SnakeEnv`).
- Standardize interface: `reset()`, `step(action)`, `render_state()`, `observation_space`, and `action_space`.
- Expose metadata endpoints so the frontend can populate defaults for network sizes and controls.

## Training & Metrics Streaming
- Use Socket.IO events (`training_progress`, `training_metrics`, `episode_summary`) to stream live training updates.
- Persist metrics per run (e.g., `models/<env>/<run_id>/metrics.jsonl`) for later comparison.
- Add dashboard endpoints to resume or inspect past runs.

## Hyperparameters & Network Builder
- Accept structured configs from the frontend including algo type, environment, hyperparameters, reward configuration, and network architecture.
- Implement a `build_network(config)` helper to translate architecture JSON (dense layers v1) into PyTorch modules for Q-networks and PPO policies.
- Surface configurable defaults for learning rate, gamma, epsilon schedule, batch sizes, PPO clip/value/entropy, etc.

## Reward Designer & Death Analytics
- Pass a `reward_config` object into Snake environment instances to control per-event rewards and penalties.
- Track death reasons (`wall`, `self`, `timeout`) and include them in episode summaries and replay metadata.
- Provide aggregated death statistics for the frontend to visualize.

## Replays & Model Management
- Record episode traces (seed, action sequence, rewards, length, death reason) when thresholds are met or when the frontend requests a recording.
- Store run artifacts under `models/<env>/<run_id>/` including configs, weights, metrics, and `replays/`.
- Implement CRUD endpoints to list, save, load, and delete stored models and replays.

## Frontend Structure & UI Enhancements
- Adopt a folder layout with `components/`, `pages/`, `hooks/`, `utils/`, and `styles/`.
- Build a Training Dashboard showing live metrics, high-speed training indicators, and status (algo, hyperparams, device).
- Add panels for Network Builder, Reward Designer, Replays, and Model Management with responsive, clear styling.
- Provide charts for reward, loss, epsilon, and death reasons per run.

## Developer Experience
- Provide a one-command dev workflow (e.g., `python dev.py`) that installs dependencies and launches backend and frontend.
- Keep documentation up to date with setup steps for macOS (MPS), Windows, and Linux.
