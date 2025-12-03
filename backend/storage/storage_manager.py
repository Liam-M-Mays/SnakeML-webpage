"""
Storage manager for models, metrics, and replays.

Directory structure:
    models/
        <env_name>/
            <run_id>/
                config.json       - Run configuration
                weights.pt        - Model weights
                metrics.jsonl     - Training metrics (one JSON per line)
                replays/
                    <replay_id>.json  - Replay data
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage of models, metrics, and replays."""

    def __init__(self, base_dir="models"):
        """
        Initialize storage manager.

        Args:
            base_dir: Base directory for all storage (default: "models")
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, env_name: str, run_id: str) -> Path:
        """Get the directory for a specific run."""
        run_dir = self.base_dir / env_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_replay_dir(self, env_name: str, run_id: str) -> Path:
        """Get the replays directory for a run."""
        replay_dir = self._get_run_dir(env_name, run_id) / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)
        return replay_dir

    # ========== CONFIG ==========

    def save_config(self, env_name: str, run_id: str, config: Dict):
        """Save run configuration."""
        run_dir = self._get_run_dir(env_name, run_id)
        config_path = run_dir / "config.json"

        # Add metadata
        config_with_meta = {
            **config,
            "created_at": datetime.now().isoformat(),
            "env_name": env_name,
            "run_id": run_id,
        }

        with open(config_path, "w") as f:
            json.dump(config_with_meta, f, indent=2)

        logger.info(f"Saved config for {env_name}/{run_id}")

    def load_config(self, env_name: str, run_id: str) -> Optional[Dict]:
        """Load run configuration."""
        run_dir = self._get_run_dir(env_name, run_id)
        config_path = run_dir / "config.json"

        if not config_path.exists():
            return None

        with open(config_path, "r") as f:
            return json.load(f)

    # ========== WEIGHTS ==========

    def get_weights_path(self, env_name: str, run_id: str) -> Path:
        """Get path for model weights file."""
        return self._get_run_dir(env_name, run_id) / "weights.pt"

    def weights_exist(self, env_name: str, run_id: str) -> bool:
        """Check if weights file exists."""
        return self.get_weights_path(env_name, run_id).exists()

    # ========== METRICS ==========

    def append_metrics(self, env_name: str, run_id: str, metrics: Dict):
        """
        Append metrics to the run's metrics file (JSONL format).

        Args:
            env_name: Environment name
            run_id: Run identifier
            metrics: Metrics dict (will add timestamp if not present)
        """
        run_dir = self._get_run_dir(env_name, run_id)
        metrics_path = run_dir / "metrics.jsonl"

        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()

        # Append as single line
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def load_metrics(self, env_name: str, run_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Load metrics from a run.

        Args:
            env_name: Environment name
            run_id: Run identifier
            limit: Maximum number of metrics to load (most recent, None = all)

        Returns:
            List of metrics dicts
        """
        run_dir = self._get_run_dir(env_name, run_id)
        metrics_path = run_dir / "metrics.jsonl"

        if not metrics_path.exists():
            return []

        metrics = []
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))

        # Return most recent if limit specified
        if limit:
            return metrics[-limit:]

        return metrics

    # ========== REPLAYS ==========

    def save_replay(self, env_name: str, run_id: str, replay_id: str, replay_data: Dict):
        """
        Save a replay.

        Args:
            env_name: Environment name
            run_id: Run identifier
            replay_id: Unique replay identifier
            replay_data: Replay data dict (should include: seed, actions, score, length, death_reason, timestamp)
        """
        replay_dir = self._get_replay_dir(env_name, run_id)
        replay_path = replay_dir / f"{replay_id}.json"

        # Add metadata
        replay_with_meta = {
            **replay_data,
            "replay_id": replay_id,
            "run_id": run_id,
            "env_name": env_name,
            "saved_at": datetime.now().isoformat(),
        }

        with open(replay_path, "w") as f:
            json.dump(replay_with_meta, f, indent=2)

    def load_replay(self, env_name: str, run_id: str, replay_id: str) -> Optional[Dict]:
        """Load a specific replay."""
        replay_dir = self._get_replay_dir(env_name, run_id)
        replay_path = replay_dir / f"{replay_id}.json"

        if not replay_path.exists():
            return None

        with open(replay_path, "r") as f:
            return json.load(f)

    def list_replays(self, env_name: str, run_id: str) -> List[Dict]:
        """
        List all replays for a run with metadata.

        Returns:
            List of replay metadata dicts
        """
        replay_dir = self._get_replay_dir(env_name, run_id)

        if not replay_dir.exists():
            return []

        replays = []
        for replay_file in replay_dir.glob("*.json"):
            try:
                with open(replay_file, "r") as f:
                    replay_data = json.load(f)
                    # Extract just the metadata
                    replays.append({
                        "replay_id": replay_data.get("replay_id", replay_file.stem),
                        "score": replay_data.get("score", 0),
                        "length": replay_data.get("length", 0),
                        "death_reason": replay_data.get("death_reason"),
                        "timestamp": replay_data.get("timestamp") or replay_data.get("saved_at"),
                    })
            except Exception as e:
                logger.warning(f"Failed to load replay {replay_file}: {e}")

        # Sort by timestamp (most recent first)
        replays.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return replays

    def delete_replay(self, env_name: str, run_id: str, replay_id: str) -> bool:
        """Delete a replay."""
        replay_dir = self._get_replay_dir(env_name, run_id)
        replay_path = replay_dir / f"{replay_id}.json"

        if replay_path.exists():
            replay_path.unlink()
            return True

        return False

    # ========== RUN MANAGEMENT ==========

    def list_runs(self, env_name: Optional[str] = None) -> List[Dict]:
        """
        List all runs, optionally filtered by environment.

        Args:
            env_name: Environment name to filter by (None = all)

        Returns:
            List of run metadata dicts
        """
        runs = []

        if env_name:
            env_dirs = [self.base_dir / env_name] if (self.base_dir / env_name).exists() else []
        else:
            env_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for env_dir in env_dirs:
            env = env_dir.name
            for run_dir in env_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_id = run_dir.name
                config = self.load_config(env, run_id)

                if config:
                    # Calculate stats from metrics
                    metrics = self.load_metrics(env, run_id)
                    best_score = max((m.get("score", 0) for m in metrics), default=0)
                    avg_reward = sum(m.get("reward", 0) for m in metrics) / len(metrics) if metrics else 0
                    total_episodes = len(metrics)

                    runs.append({
                        "env_name": env,
                        "run_id": run_id,
                        "algo": config.get("algo", "unknown"),
                        "name": config.get("name", run_id),
                        "created_at": config.get("created_at"),
                        "best_score": best_score,
                        "avg_reward": round(avg_reward, 3),
                        "total_episodes": total_episodes,
                        "has_weights": self.weights_exist(env, run_id),
                    })

        # Sort by creation time (most recent first)
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        return runs

    def delete_run(self, env_name: str, run_id: str) -> bool:
        """Delete an entire run (config, weights, metrics, replays)."""
        run_dir = self._get_run_dir(env_name, run_id)

        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
            logger.info(f"Deleted run {env_name}/{run_id}")
            return True

        return False

    # ========== DEATH ANALYTICS ==========

    def get_death_stats(self, env_name: str, run_id: str) -> Dict[str, int]:
        """
        Get death reason statistics for a run.

        Returns:
            Dict mapping death_reason -> count
        """
        metrics = self.load_metrics(env_name, run_id)

        death_counts = {}
        for m in metrics:
            death_reason = m.get("death_reason")
            if death_reason:
                death_counts[death_reason] = death_counts.get(death_reason, 0) + 1

        return death_counts


# Global storage manager instance
_storage_manager = None


def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager
