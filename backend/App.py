"""
Main Flask application for SnakeML Playground.

Provides REST API and SocketIO endpoints for:
- Environment management
- Training control
- Metrics streaming
- Model management
- Replay management
- Configuration
"""

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging

from envs import list_environments, create_env
from agents.config import get_default_config
from storage import get_storage_manager
from utils.device import get_device_info, set_device_preference, set_global_device
from training_manager import TrainingSession, create_session_from_run

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Global state
current_session: TrainingSession = None
storage = get_storage_manager()

# Initialize device
set_global_device()


# ========== ENVIRONMENT ENDPOINTS ==========

@app.route("/api/environments", methods=["GET"])
def get_environments():
    """List all available environments."""
    envs = list_environments()
    return jsonify({"environments": envs})


@app.route("/api/environments/<env_name>/metadata", methods=["GET"])
def get_environment_metadata(env_name):
    """Get metadata for a specific environment."""
    try:
        env = create_env(env_name)
        metadata = env.get_metadata()
        return jsonify(metadata)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ========== CONFIGURATION ENDPOINTS ==========

@app.route("/api/config/default/<algo>", methods=["GET"])
def get_default_configuration(algo):
    """Get default configuration for an algorithm."""
    try:
        config = get_default_config(algo)
        return jsonify(config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ========== TRAINING ENDPOINTS ==========

@app.route("/api/training/start", methods=["POST"])
def start_training():
    """Start a new training session."""
    global current_session

    if current_session and current_session.is_running:
        return jsonify({"error": "Training already running"}), 400

    config = request.json
    max_speed = config.pop("max_speed", False)

    try:
        current_session = TrainingSession(config, socketio=socketio)
        current_session.start(max_speed=max_speed)

        return jsonify({
            "message": "Training started",
            "run_id": current_session.run_id,
            "status": current_session.get_status()
        })
    except Exception as e:
        logging.error(f"Failed to start training: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/resume", methods=["POST"])
def resume_training():
    """Resume or clone a training session from saved weights."""
    global current_session

    if current_session and current_session.is_running:
        return jsonify({"error": "Training already running"}), 400

    data = request.json or {}
    env_name = data.get("env_name")
    run_id = data.get("run_id")
    continue_training = data.get("continue_training", True)
    max_speed = data.get("max_speed", False)

    if not env_name or not run_id:
        return jsonify({"error": "env_name and run_id are required"}), 400

    try:
        current_session = create_session_from_run(
            env_name,
            run_id,
            continue_training=continue_training,
            socketio=socketio,
        )
        current_session.start(max_speed=max_speed)

        return jsonify({
            "message": "Training resumed" if continue_training else "Training cloned and started",
            "run_id": current_session.run_id,
            "status": current_session.get_status()
        })
    except Exception as e:
        logging.error(f"Failed to resume training: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """Stop current training session."""
    global current_session

    if not current_session:
        return jsonify({"error": "No training session"}), 400

    current_session.stop()

    return jsonify({
        "message": "Training stopped",
        "status": current_session.get_status()
    })


@app.route("/api/training/status", methods=["GET"])
def get_training_status():
    """Get current training status."""
    if not current_session:
        return jsonify({"is_running": False})

    return jsonify(current_session.get_status())


# ========== MODEL ENDPOINTS ==========

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all saved models."""
    env_name = request.args.get("env_name")
    models = storage.list_runs(env_name=env_name)
    return jsonify({"models": models})


@app.route("/api/models/<env_name>/<run_id>", methods=["DELETE"])
def delete_model(env_name, run_id):
    """Delete a model."""
    success = storage.delete_run(env_name, run_id)

    if success:
        return jsonify({"message": "Model deleted"})
    else:
        return jsonify({"error": "Model not found"}), 404


@app.route("/api/models/<env_name>/<run_id>/config", methods=["GET"])
def get_model_config(env_name, run_id):
    """Get configuration for a saved model."""
    config = storage.load_config(env_name, run_id)

    if config:
        return jsonify(config)
    else:
        return jsonify({"error": "Config not found"}), 404


# ========== METRICS ENDPOINTS ==========

@app.route("/api/metrics/<env_name>/<run_id>", methods=["GET"])
def get_metrics(env_name, run_id):
    """Get metrics for a run."""
    limit = request.args.get("limit", type=int)
    metrics = storage.load_metrics(env_name, run_id, limit=limit)

    return jsonify({"metrics": metrics})


@app.route("/api/metrics/<env_name>/<run_id>/death_stats", methods=["GET"])
def get_death_stats(env_name, run_id):
    """Get death statistics for a run."""
    stats = storage.get_death_stats(env_name, run_id)

    return jsonify({"death_stats": stats})


# ========== REPLAY ENDPOINTS ==========

@app.route("/api/replays/<env_name>/<run_id>", methods=["GET"])
def list_replays(env_name, run_id):
    """List replays for a run."""
    replays = storage.list_replays(env_name, run_id)
    return jsonify({"replays": replays})


@app.route("/api/replays/<env_name>/<run_id>/<replay_id>", methods=["GET"])
def get_replay(env_name, run_id, replay_id):
    """Get a specific replay."""
    replay = storage.load_replay(env_name, run_id, replay_id)

    if replay:
        return jsonify(replay)
    else:
        return jsonify({"error": "Replay not found"}), 404


@app.route("/api/replays/<env_name>/<run_id>/<replay_id>", methods=["DELETE"])
def delete_replay(env_name, run_id, replay_id):
    """Delete a replay."""
    success = storage.delete_replay(env_name, run_id, replay_id)

    if success:
        return jsonify({"message": "Replay deleted"})
    else:
        return jsonify({"error": "Replay not found"}), 404


# ========== SYSTEM ENDPOINTS ==========

@app.route("/api/device", methods=["GET"])
def get_device():
    """Get device information."""
    device_info = get_device_info()
    return jsonify(device_info)


@app.route("/api/device/set", methods=["POST"])
def set_device_preference_endpoint():
    """Set device preference for future training sessions."""

    payload = request.get_json(silent=True) or {}
    preference = payload.get("device")

    if preference not in ["auto", "cpu", "cuda", "mps"]:
        return jsonify({"error": "Invalid device preference"}), 400

    try:
        set_device_preference(preference)
        # Keep the cached global device aligned with the new preference
        set_global_device()
    except ValueError:
        return jsonify({"error": "Invalid device preference"}), 400

    return jsonify(get_device_info())


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": get_device_info(),
    })


# ========== SOCKETIO EVENTS (for legacy human play mode) ==========

@socketio.on("init")
def handle_init(data):
    """Initialize a manual play session (legacy)."""
    grid_size = data.get("grid_size", 10)

    # Create environment for manual play
    env = create_env("snake", grid_size=grid_size)
    state = env.reset()
    render = env.render_state()

    emit("game_update", {
        **render,
        "episode": 0,
    })


@socketio.on("step")
def handle_step(data):
    """Handle manual step (legacy - would need session management for full support)."""
    # This is a simplified version; full implementation would track per-client sessions
    emit("game_update", {"message": "Use the new training API for AI play"})


# ========== MAIN ==========

if __name__ == "__main__":
    print("=" * 60)
    print("🐍 SnakeML Playground Backend")
    print("=" * 60)
    print(f"Device: {get_device_info()}")
    print("Backend running on: http://127.0.0.1:5000")
    print("API documentation: http://127.0.0.1:5000/api/health")
    print("=" * 60)

    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        log_output=False
    )
