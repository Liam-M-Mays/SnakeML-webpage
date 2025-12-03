"""Training run orchestration with Socket.IO streaming and persistence."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask_socketio import SocketIO

from backend.envs.registry import create_env, env_metadata, list_envs
from backend.envs.snake import default_reward_config
from .config import RunConfig
from .dqn import DQNTrainer


class RunManager:
    def __init__(self, models_root: Path, socketio: SocketIO):
        self.models_root = models_root
        self.socketio = socketio
        self.active_runs: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    # ---------- environment info ----------
    def available_envs(self):
        return [{"name": name, **env_metadata(name)} for name in list_envs()]

    def reward_defaults(self):
        return default_reward_config()

    # ---------- run lifecycle ----------
    def start_run(self, cfg_dict: Dict[str, Any]):
        cfg = RunConfig.from_dict(cfg_dict)
        run_dir = self.models_root / cfg.env / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2))

        thread = self.socketio.start_background_task(self._train_loop, cfg, run_dir)
        with self.lock:
            self.active_runs[cfg.run_id] = thread
        return cfg.run_id

    def list_runs(self):
        runs: List[Dict[str, Any]] = []
        if not self.models_root.exists():
            return runs
        for env_dir in self.models_root.iterdir():
            if not env_dir.is_dir():
                continue
            for run_dir in env_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                cfg_path = run_dir / "config.json"
                cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
                metrics_path = run_dir / "metrics.jsonl"
                best_score = None
                if metrics_path.exists():
                    for line in metrics_path.read_text().splitlines():
                        m = json.loads(line)
                        best_score = max(best_score or m.get("reward", 0), m.get("reward", 0))
                runs.append({
                    "env": env_dir.name,
                    "run_id": run_dir.name,
                    "config": cfg,
                    "best_reward": best_score,
                })
        return runs

    def metrics_for_run(self, env: str, run_id: str):
        path = self.models_root / env / run_id / "metrics.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def replays_for_run(self, env: str, run_id: str):
        replay_dir = self.models_root / env / run_id / "replays"
        if not replay_dir.exists():
            return []
        items = []
        for rp in replay_dir.iterdir():
            if rp.suffix != ".json":
                continue
            data = json.loads(rp.read_text())
            items.append({"id": rp.stem, **{k: v for k, v in data.items() if k != "actions"}})
        return items

    def replay_detail(self, env: str, run_id: str, replay_id: str):
        rp = self.models_root / env / run_id / "replays" / f"{replay_id}.json"
        if not rp.exists():
            return None
        return json.loads(rp.read_text())

    def delete_run(self, env: str, run_id: str):
        import shutil
        path = self.models_root / env / run_id
        if path.exists():
            shutil.rmtree(path)

    # ---------- training ----------
    def _train_loop(self, cfg: RunConfig, run_dir: Path):
        env = create_env(cfg.env, reward_config={**default_reward_config(), **cfg.reward_config})
        obs = env.reset()
        trainer = DQNTrainer(len(obs), env.action_space.n or 3, cfg.to_dict(), run_dir)
        trainer.save_config(cfg.to_dict())

        metrics_path = run_dir / "metrics.jsonl"
        replay_dir = run_dir / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)

        max_episodes = int(cfg.hyperparameters.get("maxEpisodes", 50))
        for episode in range(1, max_episodes + 1):
            obs = env.reset(seed=int(time.time()))
            done = False
            total_reward = 0.0
            steps = 0
            actions_taken: List[int] = []
            start_time = time.time()
            while not done:
                action = trainer.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                trainer.replay.push(obs, action, reward, next_obs, done)
                loss = trainer.optimize()
                trainer.decay_epsilon()
                obs = next_obs
                total_reward += reward
                steps += 1
                actions_taken.append(action)
            trainer.update_target()
            trainer.save_weights()
            elapsed = time.time() - start_time
            metrics_entry = {
                "run_id": cfg.run_id,
                "episode": episode,
                "reward": total_reward,
                "length": steps,
                "loss": loss if loss is not None else 0.0,
                "epsilon": trainer.eps,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(metrics_entry) + "\n")
            self.socketio.emit("training_metrics", metrics_entry)
            self.socketio.emit(
                "training_progress",
                {
                    "run_id": cfg.run_id,
                    "episode": episode,
                    "episodes_per_second": 1.0 / elapsed if elapsed > 0 else 0.0,
                    "current_score": total_reward,
                    "best_score": self._best_reward(metrics_path),
                },
            )
            self._write_replay(cfg, replay_dir, episode, actions_taken, env, info)
            self.socketio.emit(
                "episode_summary",
                {
                    "run_id": cfg.run_id,
                    "episode": episode,
                    "score": total_reward,
                    "length": steps,
                    "death_reason": info.get("death_reason"),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    def _write_replay(self, cfg: RunConfig, replay_dir: Path, episode: int, actions: List[int], env, info):
        payload = {
            "episode": episode,
            "env": cfg.env,
            "run_id": cfg.run_id,
            "seed": int(time.time()),
            "actions": actions,
            "score": env.score,
            "length": len(actions),
            "death_reason": info.get("death_reason"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        path = replay_dir / f"{episode}.json"
        path.write_text(json.dumps(payload, indent=2))

    def _best_reward(self, metrics_path: Path):
        if not metrics_path.exists():
            return 0
        best = 0
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            best = max(best, data.get("reward", 0))
        return best

    def death_stats(self, env: str, run_id: str):
        replays = self.replays_for_run(env, run_id)
        stats: Dict[str, int] = {}
        for rp in replays:
            reason = rp.get("death_reason", "unknown") or "unknown"
            stats[reason] = stats.get(reason, 0) + 1
        return stats
