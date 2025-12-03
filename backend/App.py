from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from Session import Session
from utils.device import (
    set_device_preference,
    get_device_preference,
    get_device_info,
    VALID_PREFERENCES
)

import time
import logging
import traceback

# ========== LOGGING SETUP ==========
# Configure logging format for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Create app logger
logger = logging.getLogger('SnakeML')
logger.setLevel(logging.INFO)

# Reduce noise from Flask and SocketIO internals
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}})

# Let Flask-SocketIO auto-pick async mode (will pick "eventlet" if installed)
socketio = SocketIO(app, cors_allowed_origins=["*"])

# ========== GLOBAL STATE ==========
session = None
loop = False


# ========== REST API ENDPOINTS ==========

@app.route('/api/device', methods=['GET'])
def get_device():
    """Get current device configuration and status."""
    try:
        info = get_device_info()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/device/set', methods=['POST'])
def set_device():
    """
    Set device preference.
    Body: { "device": "auto" | "cpu" | "cuda" | "mps" }
    """
    try:
        data = request.get_json()
        if not data or 'device' not in data:
            return jsonify({"error": "Missing 'device' field in request body"}), 400

        device_pref = data['device'].lower().strip()
        if device_pref not in VALID_PREFERENCES:
            return jsonify({
                "error": f"Invalid device: {device_pref}",
                "valid_options": list(VALID_PREFERENCES)
            }), 400

        success = set_device_preference(device_pref)
        if success:
            info = get_device_info()
            logger.info(f"Device preference changed to: {device_pref} (resolved: {info['resolved']})")
            return jsonify(info), 200
        else:
            return jsonify({"error": "Failed to set device preference"}), 500

    except Exception as e:
        logger.error(f"Error setting device: {e}")
        return jsonify({"error": str(e)}), 500


# ========== SOCKET.IO EVENT HANDLERS ==========

@socketio.on('init')
def handle_init(data):
    global session
    try:
        grid_size = data.get('grid_size')
        AI = data.get('AI_mode')
        Model = data.get('modelType')
        parms = data.get('params')

        device_info = get_device_info()
        logger.info(f"Initializing session: grid={grid_size}, AI={AI}, model={Model}, device={device_info['resolved']}")

        session = Session(grid_size=grid_size, AI_mode=AI, model=Model, params=parms)
        score, food_position, snake_position, game_over = session.get_state()

        emit('game_update', {
            'score': score,
            'food_position': food_position,
            'snake_position': snake_position,
            'game_over': game_over,
            'episode': session.episodes
        })
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        logger.error(traceback.format_exc())

@socketio.on('reset_game')
def handle_reset_game(data):
    global session
    if session is None: return
    score, food_position, snake_position = session.reset()
    emit('game_update', {
        'score': score,
        'food_position': food_position,
        'snake_position': snake_position,
        'game_over': True,
        'episode': session.episodes
    })

@socketio.on('step')
def handle_step(data):
    global session
    if session is None: return
    action = data.get('action')
    score, food_position, snake_position, game_over = session.step(action)
    if game_over:
        score, food_position, snake_position = session.reset()
    emit('game_update', {
        'score': score,
        'food_position': food_position,
        'snake_position': snake_position,
        'game_over': game_over,
        'episode': session.episodes
    })
    emit('set_highscore', {
        'highscore': session.highscore
    })

@socketio.on('AI_step')
def handle_AI_step():
    global session
    if session is None: return
    score, food_position, snake_position, game_over = session.AI_step()
    if game_over:
        score, food_position, snake_position = session.reset()
    emit('game_update', {
        'score': score,
        'food_position': food_position,
        'snake_position': snake_position,
        'game_over': False,
        'episode': session.episodes
    })
    emit('set_highscore', {
        'highscore': session.highscore
    })

@socketio.on('AI_loop')
def handle_AI_loop():
    global session
    if session is None:
        logger.warning("AI_loop called but no session exists")
        return
    global loop
    loop = True
    highscore = session.highscore
    step = 0
    total_time = 0
    start_episode = session.episodes

    logger.info(f"AI training loop started (fast mode) - starting at episode {start_episode}")

    try:
        while loop:
            step += 1
            stamp = time.perf_counter()
            score, food_position, snake_position, game_over = session.AI_step()
            total_time += (time.perf_counter() - stamp)*1000

            if game_over:
                score, food_position, snake_position = session.reset()

            # Emit updates every 500ms to avoid overwhelming the client
            if total_time > 500:
                total_time = 0
                emit('game_update', {
                    'score': score,
                    'food_position': {"x": None, "y": None},
                    'snake_position': [{"x": None, "y": None}],
                    'game_over': False,
                    'episode': session.episodes
                })

            if session.highscore > highscore:
                highscore = session.highscore
                emit('set_highscore', {
                    'highscore': session.highscore
                })

            socketio.sleep(0)  # Yield to allow other events to process

        # Log when training stops
        episodes_completed = session.episodes - start_episode
        logger.info(f"AI training loop stopped by user - completed {episodes_completed} episodes, highscore: {highscore}")

    except Exception as e:
        logger.error(f"Error in AI training loop: {e}")
        logger.error(traceback.format_exc())
        loop = False

@socketio.on('stop_loop')
def handle_stop_training():
    global loop
    if loop:
        logger.info("Stop training requested by user")
    loop = False


if __name__ == '__main__':
    logger.info("Starting SnakeML server on http://127.0.0.1:5000")
    device_info = get_device_info()
    logger.info(f"Device: {device_info['name']} (preference: {device_info['preference']}, available: {device_info['available']})")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=True, log_output=False)