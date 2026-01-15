"""
Flask SocketIO server for SnakeML.

This handles WebSocket communication between the frontend and backend.
"""
import sys
import os
import time
import logging
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import torch

from games.snake import SnakeEnv
from server.session import Session
from server.players import create_player, HumanPlayer, NetworkPlayer
from server.validation import (
    validate_model_name,
    validate_grid_size,
    validate_hyperparameters,
    validate_filename
)


def get_available_devices():
    """Detect available compute devices."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    return devices


AVAILABLE_DEVICES = get_available_devices()

# Models directory
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Suppress verbose logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

# CORS configuration
# In production, set CORS_ORIGINS environment variable to restrict origins
# Example: CORS_ORIGINS=http://localhost:3000,https://myapp.com
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000')
if CORS_ORIGINS == '*':
    allowed_origins = '*'
else:
    allowed_origins = [origin.strip() for origin in CORS_ORIGINS.split(',')]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": allowed_origins}})
socketio = SocketIO(app, cors_allowed_origins=allowed_origins)


@app.route('/')
def index():
    """Health check endpoint."""
    return {'status': 'ok', 'message': 'SnakeML backend running'}

# Global session (single-user mode)
session = None
training_loop = False
debug_settings = {'vision': False, 'path': False, 'segments': False}  # Debug visualization toggles


def create_game(grid_size: int = 10, use_cnn: bool = False,
                random_start_state: bool = False, random_max_length: int = None,
                inputs: dict = None, rewards: dict = None):
    """
    Factory function for creating Snake game.

    Args:
        grid_size: Size of the game grid
        use_cnn: Whether the network will use CNN
        random_start_state: Whether to start with random length/direction
        random_max_length: Max snake length for random start
        inputs: Input configuration (which features to include)
        rewards: Reward values configuration

    Returns:
        SnakeEnv instance
    """
    return SnakeEnv(
        grid_size=grid_size,
        use_cnn=use_cnn,
        starvation_limit=grid_size ** 2,  # For AI training
        random_start_state=random_start_state,
        random_max_length=random_max_length,
        inputs=inputs or {},
        rewards=rewards or {}
    )


@socketio.on('init')
def handle_init(data):
    """
    Initialize a new game session.

    Expected data:
        - grid_size: int
        - control_mode: 'human', 'dqn', 'ppo', 'mann', or 'mapo'
        - params: dict of network hyperparameters (for AI modes)
        - device: 'cpu', 'cuda', 'mps', or None for auto-detect
    """
    global session

    # Validate grid size
    grid_size_raw = data.get('grid_size', 10)
    is_valid, error, grid_size = validate_grid_size(grid_size_raw)
    if not is_valid:
        print(f"[INIT] Grid size validation warning: {error}")

    control_mode = data.get('control_mode', 'human')
    # Support old 'AI_mode' and 'modelType' for backwards compatibility
    if data.get('AI_mode') and control_mode == 'human':
        control_mode = data.get('modelType', 'dqn')

    # Validate control mode
    if control_mode not in ('human', 'dqn', 'ppo', 'mann', 'mapo'):
        print(f"[INIT] Invalid control mode: {control_mode}, defaulting to human")
        control_mode = 'human'

    # Validate hyperparameters for AI modes
    params = data.get('params', {})
    if control_mode in ('dqn', 'ppo', 'mann', 'mapo'):
        is_valid, errors, params = validate_hyperparameters(params, control_mode)
        if not is_valid:
            print(f"[INIT] Hyperparameter validation warnings: {errors}")

    # Validate device selection
    device = data.get('device', None)
    if device is not None and device not in AVAILABLE_DEVICES:
        print(f"[INIT] Device {device} not available, will auto-detect")
        device = None

    random_start_state = data.get('random_start_state', False)
    random_max_length = data.get('random_max_length', None)
    inputs = data.get('inputs', {})
    rewards = data.get('rewards', {})

    # Create game
    use_cnn = False  # Could be a param in the future
    game = create_game(
        grid_size=grid_size,
        use_cnn=use_cnn,
        random_start_state=random_start_state,
        random_max_length=random_max_length,
        inputs=inputs,
        rewards=rewards,
    )

    # Get state dimension from game
    state, board = game.get_state_for_network()
    state_dim = len(state)

    # Create player with specified device
    player = create_player(
        control_mode=control_mode,
        params=params,
        state_dim=state_dim,
        use_cnn=use_cnn,
        device=device
    )

    # Create session
    session = Session(game, player)

    # Disable high score tracking if random start state is enabled
    session.set_track_highscore(not random_start_state)

    # Send initial state
    game_state = session.get_state()
    emit('game_update', {
        **game_state,
        'episode': session.episodes
    })

    # Send current device info
    current_device = None
    if isinstance(player, NetworkPlayer):
        current_device = player.network.device
    emit('devices_list', {
        'available': AVAILABLE_DEVICES,
        'current': current_device
    })


@socketio.on('reset_game')
def handle_reset_game(data=None):
    """Reset the current game."""
    global session
    if session is None:
        return

    game_state = session.reset()
    emit('game_update', {
        **game_state,
        'game_over': False,
        'episode': session.episodes
    })


@socketio.on('set_random_start_state')
def handle_set_random_start_state(data):
    """
    Toggle random start state during gameplay.

    When enabled: snake starts with random length/direction, high score disabled
    When disabled: normal start, high score tracking resumes (resets to 0)

    Expected data:
        - enabled: bool
    """
    global session
    if session is None:
        return

    enabled = data.get('enabled', False)

    # Update game setting
    if hasattr(session.game, 'set_random_start_state'):
        session.game.set_random_start_state(enabled)

    # Update high score tracking (inverse of random start state)
    session.set_track_highscore(not enabled)

    # Notify frontend of the change
    emit('random_start_state_changed', {
        'enabled': enabled,
        'highscore': session.highscore  # Send current (possibly reset) high score
    })


@socketio.on('set_random_max_length')
def handle_set_random_max_length(data):
    """
    Update max random snake length during gameplay.

    Expected data:
        - max_length: int
    """
    global session
    if session is None:
        return

    max_length = data.get('max_length', None)
    if max_length is not None and hasattr(session.game, 'set_random_max_length'):
        session.game.set_random_max_length(int(max_length))


@socketio.on('set_debug_settings')
def handle_set_debug_settings(data):
    """
    Update debug visualization settings.

    Can be called during gameplay to toggle debug overlays.

    Expected data:
        - vision: bool (show danger detection cells)
        - path: bool (show shortest path to food)
        - segments: bool (show segment tracking)
    """
    global debug_settings

    if 'vision' in data:
        debug_settings['vision'] = bool(data['vision'])
    if 'path' in data:
        debug_settings['path'] = bool(data['path'])
    if 'segments' in data:
        debug_settings['segments'] = bool(data['segments'])


@socketio.on('step')
def handle_step(data):
    """
    Execute one game step.

    For human players: action should be provided in data
    For AI players: action is computed by the network

    Expected data:
        - action: int (0=forward, 1=right, 2=left) - required for human
    """
    global session, debug_settings
    if session is None:
        return

    # If human player, update their pending action
    if isinstance(session.player, HumanPlayer):
        action = data.get('action', 0)
        session.player.set_action(action)

    # Execute step
    game_state = session.tick()

    # Reset if game over
    if game_state['game_over']:
        game_state = session.reset()

    # Build game update
    update_data = {
        **game_state,
        'episode': session.episodes
    }

    # Add debug visualization data if enabled
    if debug_settings['vision'] or debug_settings['path'] or debug_settings['segments']:
        if hasattr(session.game, 'get_debug_info'):
            debug_info = session.game.get_debug_info(
                debug_vision=debug_settings['vision'],
                debug_path=debug_settings['path'],
                debug_segments=debug_settings['segments']
            )
            update_data['debug'] = debug_info

    emit('game_update', update_data)
    emit('set_highscore', {'highscore': session.highscore})

    # Emit real-time expert weights for MANN during step-by-step training
    if isinstance(session.player, NetworkPlayer):
        network = session.player.network
        if hasattr(network, 'last_blend_weights'):
            emit('expert_weights_realtime', {
                'weights': network.last_blend_weights
            })

        # Emit current input state for visualization
        state, _ = session.game.get_state_for_network()
        emit('input_state', {
            'features': state.tolist() if hasattr(state, 'tolist') else list(state)
        })


@socketio.on('AI_loop')
def handle_ai_loop():
    """
    Run AI training loop at maximum speed.

    Runs until 'stop_loop' is received.
    Periodically emits updates (every 500ms) to avoid overwhelming frontend.
    """
    global session, training_loop
    if session is None:
        return

    training_loop = True
    highscore = session.highscore
    last_emit_time = time.perf_counter()
    steps_per_yield = 50  # Batch steps between yields for performance

    while training_loop:
        # Run a batch of steps
        for _ in range(steps_per_yield):
            if not training_loop:
                break

            # Execute step
            game_state = session.tick()

            # Reset if game over
            if game_state['game_over']:
                game_state = session.reset()

            # Emit highscore if it changed
            if session.highscore > highscore:
                highscore = session.highscore
                emit('set_highscore', {'highscore': session.highscore})

        # Emit periodic updates (every 500ms)
        current_time = time.perf_counter()
        if current_time - last_emit_time > 0.5:
            last_emit_time = current_time
            emit('game_update', {
                'score': game_state.get('score', 0),
                'game_over': False,
                'episode': session.episodes
            })

            # Emit training metrics
            emit('training_metrics', session.metrics.get_chart_data())

            # Emit current input state for visualization
            state, _ = session.game.get_state_for_network()
            emit('input_state', {
                'features': state.tolist() if hasattr(state, 'tolist') else list(state)
            })

        # Yield to event loop with small sleep to allow stop_loop to be processed
        socketio.sleep(0.001)


@socketio.on('stop_loop')
def handle_stop_loop():
    """Stop the AI training loop."""
    global training_loop
    training_loop = False


@socketio.on('get_metrics')
def handle_get_metrics():
    """
    Get current training metrics.

    Returns chart data for visualization.
    """
    global session
    if session is None:
        emit('training_metrics', {'rewards': [], 'scores': [], 'losses': [], 'episodes': 0})
        return

    emit('training_metrics', session.metrics.get_chart_data())


# ========== DEVICE MANAGEMENT ==========

@socketio.on('get_devices')
def handle_get_devices():
    """
    Get list of available compute devices.

    Returns list of device names and current device if session exists.
    """
    global session
    current_device = None
    if session and isinstance(session.player, NetworkPlayer):
        current_device = session.player.network.device

    emit('devices_list', {
        'available': AVAILABLE_DEVICES,
        'current': current_device
    })


@socketio.on('set_device')
def handle_set_device(data):
    """
    Switch the compute device for the current network.

    Can be called during training - will briefly pause to move tensors.
    Recreates the optimizer to ensure internal state is on correct device.

    Expected data:
        - device: str - 'cpu', 'cuda', or 'mps'
    """
    global session

    device = data.get('device', 'cpu')

    # Validate device
    if device not in AVAILABLE_DEVICES:
        emit('device_changed', {
            'success': False,
            'error': f'Device {device} not available. Available: {AVAILABLE_DEVICES}'
        })
        return

    if session is None:
        emit('device_changed', {
            'success': False,
            'error': 'No active session'
        })
        return

    if not isinstance(session.player, NetworkPlayer):
        emit('device_changed', {
            'success': False,
            'error': 'No AI network in current session'
        })
        return

    network = session.player.network
    old_device = network.device

    if old_device == device:
        emit('device_changed', {
            'success': True,
            'device': device,
            'message': f'Already on {device}'
        })
        return

    try:
        # Acquire lock if network has one (prevents race conditions during training)
        lock = getattr(network, '_lock', None)
        if lock:
            lock.acquire()

        try:
            # Move network to new device
            network.to(device)
            network.device = device

            # For DQN, also move target network
            if hasattr(network, 'target_net'):
                network.target_net.to(device)

            # CRITICAL: Recreate optimizer with parameters now on new device
            # Adam's internal state (momentum, variance) are device-specific
            if hasattr(network, 'optimizer') and hasattr(network, 'lr'):
                import torch.optim as optim
                network.optimizer = optim.Adam(network.parameters(), lr=network.lr)
                print(f"[DEVICE] Recreated optimizer on {device}")

        finally:
            if lock:
                lock.release()

        print(f"[DEVICE] Switched from {old_device} to {device}")

        emit('device_changed', {
            'success': True,
            'device': device,
            'message': f'Switched from {old_device} to {device}'
        })

    except Exception as e:
        print(f"[DEVICE] Error switching to {device}: {e}")
        emit('device_changed', {
            'success': False,
            'error': str(e)
        })


# Keep old handlers for backwards compatibility
@socketio.on('AI_step')
def handle_ai_step():
    """
    Execute one AI step.

    Kept for backwards compatibility - just calls handle_step.
    """
    handle_step({})


# ========== MODEL SAVE/LOAD ==========

@socketio.on('save_model')
def handle_save_model(data):
    """
    Save the current network to a file.

    Expected data:
        - name: str - Name for the saved model
        - agent_id: str - ID of the agent (for metadata)
        - agent_name: str - Display name of the agent
    """
    global session
    if session is None:
        emit('save_model_result', {'success': False, 'error': 'No active session'})
        return

    if not isinstance(session.player, NetworkPlayer):
        emit('save_model_result', {'success': False, 'error': 'No AI network to save'})
        return

    name = data.get('name', 'model')
    agent_id = data.get('agent_id', '')
    agent_name = data.get('agent_name', name)

    # Validate and sanitize model name
    is_valid, error, safe_name = validate_model_name(name)
    if not is_valid:
        emit('save_model_result', {'success': False, 'error': error})
        return

    # Determine network type
    network = session.player.network
    network_type = 'dqn' if 'DQN' in type(network).__name__ else 'ppo'

    # Create filename with timestamp
    timestamp = int(time.time())
    filename = f"{safe_name}_{network_type}_{timestamp}"

    # Save model weights
    model_path = MODELS_DIR / f"{filename}.pt"
    network.save(str(model_path))

    # Save metadata
    metadata = {
        'name': agent_name,
        'filename': filename,
        'network_type': network_type,
        'episodes': network.episode_count,
        'timestamp': timestamp,
        'agent_id': agent_id,
        'game': 'snake',
    }
    meta_path = MODELS_DIR / f"{filename}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    emit('save_model_result', {
        'success': True,
        'filename': filename,
        'metadata': metadata
    })


@socketio.on('load_model')
def handle_load_model(data):
    """
    Load a saved model into the current network.

    Expected data:
        - filename: str - Name of saved model (without extension)
    """
    global session
    print(f"[LOAD] Received load_model request: {data}")

    if session is None:
        print("[LOAD] Error: No active session")
        emit('load_model_result', {'success': False, 'error': 'No active session'})
        return

    if not isinstance(session.player, NetworkPlayer):
        print("[LOAD] Error: No AI network to load into")
        emit('load_model_result', {'success': False, 'error': 'No AI network to load into'})
        return

    filename = data.get('filename', '')

    # Validate filename to prevent path traversal
    is_valid, error = validate_filename(filename)
    if not is_valid:
        print(f"[LOAD] Error: {error}")
        emit('load_model_result', {'success': False, 'error': error})
        return

    model_path = MODELS_DIR / f"{filename}.pt"
    meta_path = MODELS_DIR / f"{filename}.json"

    if not model_path.exists():
        print(f"[LOAD] Error: Model not found: {filename}")
        emit('load_model_result', {'success': False, 'error': f'Model not found: {filename}'})
        return

    try:
        # Load metadata if exists
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

        # Check network type matches
        network = session.player.network
        current_type = 'dqn' if 'DQN' in type(network).__name__ else 'ppo'
        saved_type = metadata.get('network_type', current_type)

        if current_type != saved_type:
            print(f"[LOAD] Error: Type mismatch - model is {saved_type}, network is {current_type}")
            emit('load_model_result', {
                'success': False,
                'error': f'Type mismatch: model is {saved_type}, current network is {current_type}'
            })
            return

        # Load the model
        print(f"[LOAD] Loading weights from {model_path}")
        network.load(str(model_path))
        print(f"[LOAD] Successfully loaded model: {filename}")

        emit('load_model_result', {
            'success': True,
            'filename': filename,
            'metadata': metadata
        })

    except Exception as e:
        print(f"[LOAD] Exception: {e}")
        emit('load_model_result', {'success': False, 'error': str(e)})


@socketio.on('list_models')
def handle_list_models(data=None):
    """
    List all saved models.

    Returns list of model metadata.
    """
    models = []

    for meta_path in MODELS_DIR.glob('*.json'):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                # Check that the .pt file also exists
                pt_path = MODELS_DIR / f"{metadata['filename']}.pt"
                if pt_path.exists():
                    models.append(metadata)
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by timestamp, newest first
    models.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

    emit('models_list', {'models': models})


@socketio.on('delete_model')
def handle_delete_model(data):
    """
    Delete a saved model.

    Expected data:
        - filename: str - Name of model to delete (without extension)
    """
    filename = data.get('filename', '')

    # Validate filename to prevent path traversal
    is_valid, error = validate_filename(filename)
    if not is_valid:
        emit('delete_model_result', {'success': False, 'error': error})
        return

    model_path = MODELS_DIR / f"{filename}.pt"
    meta_path = MODELS_DIR / f"{filename}.json"

    deleted = False
    if model_path.exists():
        model_path.unlink()
        deleted = True
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    emit('delete_model_result', {
        'success': deleted,
        'filename': filename
    })


if __name__ == '__main__':
    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=True,
        log_output=False
    )
