from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

# Ensure package imports work when running as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.envs.registry import env_metadata, list_envs  # noqa: E402
from backend.envs.snake import default_reward_config  # noqa: E402
from backend.training.manager import RunManager  # noqa: E402

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"]}})
socketio = SocketIO(app, cors_allowed_origins="*")

MODELS_ROOT = ROOT / "models"
MODELS_ROOT.mkdir(exist_ok=True)
run_manager = RunManager(MODELS_ROOT, socketio)


# ----------------------- Environment info -----------------------
@app.route("/api/envs", methods=["GET"])
def api_list_envs():
    return jsonify(run_manager.available_envs())


@app.route("/api/envs/<env_name>/metadata", methods=["GET"])
def api_env_metadata(env_name: str):
    try:
        return jsonify(env_metadata(env_name))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ----------------------- Rewards -----------------------
@app.route("/api/rewards/<env_name>", methods=["GET", "POST"])
def api_reward(env_name: str):
    reward_path = MODELS_ROOT / env_name / "default_reward.json"
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    if request.method == "GET":
        if reward_path.exists():
            return jsonify(json.loads(reward_path.read_text()))
        if env_name == "snake":
            return jsonify(default_reward_config())
        return jsonify({})
    payload = request.get_json(force=True) or {}
    reward_path.write_text(json.dumps(payload, indent=2))
    return jsonify({"status": "saved", "reward_config": payload})


# ----------------------- Runs -----------------------
@app.route("/api/runs", methods=["GET"])
def api_runs():
    return jsonify(run_manager.list_runs())


@app.route("/api/runs/start", methods=["POST"])
def api_start_run():
    data: Any = request.get_json(force=True) or {}
    run_id = run_manager.start_run(data)
    return jsonify({"run_id": run_id})


@app.route("/api/runs/<env_name>/<run_id>/config", methods=["GET"])
def api_run_config(env_name: str, run_id: str):
    cfg_path = MODELS_ROOT / env_name / run_id / "config.json"
    if not cfg_path.exists():
        return jsonify({"error": "missing config"}), 404
    return jsonify(json.loads(cfg_path.read_text()))


@app.route("/api/runs/<env_name>/<run_id>/metrics", methods=["GET"])
def api_run_metrics(env_name: str, run_id: str):
    return jsonify(run_manager.metrics_for_run(env_name, run_id))


@app.route("/api/runs/<env_name>/<run_id>/replays", methods=["GET"])
def api_run_replays(env_name: str, run_id: str):
    return jsonify(run_manager.replays_for_run(env_name, run_id))


@app.route("/api/runs/<env_name>/<run_id>/replays/<replay_id>", methods=["GET"])
def api_replay_detail(env_name: str, run_id: str, replay_id: str):
    data = run_manager.replay_detail(env_name, run_id, replay_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@app.route("/api/runs/<env_name>/<run_id>/death_stats", methods=["GET"])
def api_death_stats(env_name: str, run_id: str):
    return jsonify(run_manager.death_stats(env_name, run_id))


@app.route("/api/runs/<env_name>/<run_id>", methods=["DELETE"])
def api_delete_run(env_name: str, run_id: str):
    run_manager.delete_run(env_name, run_id)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
