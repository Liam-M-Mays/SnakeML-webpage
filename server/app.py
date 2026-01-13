"""
Flask SocketIO server for MLplayground.

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

from games.snake import SnakeEnv
from server.session import Session
from server.players import create_player, HumanPlayer, NetworkPlayer

# Models directory
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Suppress verbose logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    """Health check endpoint."""
    return {'status': 'ok', 'message': 'MLplayground backend running'}

# Global session (single-user mode)
session = None
training_loop = False


def create_game(game_type: str, grid_size: int, use_cnn: bool = False):
    """
    Factory function for creating games.

    Args:
        game_type: Type of game ('snake', etc.)
        grid_size: Size of the game grid
        use_cnn: Whether the network will use CNN

    Returns:
        GameEnv instance
    """
    if game_type == 'snake':
        return SnakeEnv(
            grid_size=grid_size,
            use_cnn=use_cnn,
            starvation_limit=grid_size ** 2  # For AI training
        )
    else:
        raise ValueError(f"Unknown game type: {game_type}")


@socketio.on('init')
def handle_init(data):
    """
    Initialize a new game session.

    Expected data:
        - grid_size: int
        - control_mode: 'human', 'dqn', or 'ppo' (replaces AI_mode)
        - params: dict of network hyperparameters (for AI modes)
        - game_type: 'snake' (optional, defaults to snake)
    """
    global session

    grid_size = data.get('grid_size', 10)
    control_mode = data.get('control_mode', 'human')
    # Support old 'AI_mode' and 'modelType' for backwards compatibility
    if data.get('AI_mode') and control_mode == 'human':
        control_mode = data.get('modelType', 'dqn')
    params = data.get('params', {})
    game_type = data.get('game_type', 'snake')

    # Create game
    use_cnn = False  # Could be a param in the future
    game = create_game(game_type, grid_size, use_cnn)

    # Get state dimension from game
    state, board = game.get_state_for_network()
    state_dim = len(state)

    # Create player
    player = create_player(
        control_mode=control_mode,
        params=params,
        state_dim=state_dim,
        use_cnn=use_cnn
    )

    # Create session
    session = Session(game, player)

    # Send initial state
    game_state = session.get_state()
    emit('game_update', {
        'score': game_state['score'],
        'food_position': game_state['food_position'],
        'snake_position': game_state['snake_position'],
        'game_over': game_state['game_over'],
        'episode': session.episodes
    })


@socketio.on('reset_game')
def handle_reset_game(data=None):
    """Reset the current game."""
    global session
    if session is None:
        return

    game_state = session.reset()
    emit('game_update', {
        'score': game_state['score'],
        'food_position': game_state['food_position'],
        'snake_position': game_state['snake_position'],
        'game_over': False,
        'episode': session.episodes
    })


@socketio.on('step')
def handle_step(data):
    """
    Execute one game step.

    For human players: action should be provided in data
    For AI players: action is computed by the network

    Expected data:
        - action: int (0=forward, 1=right, 2=left) - required for human
    """
    global session
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

    emit('game_update', {
        'score': game_state['score'],
        'food_position': game_state['food_position'],
        'snake_position': game_state['snake_position'],
        'game_over': game_state['game_over'],
        'episode': session.episodes
    })
    emit('set_highscore', {'highscore': session.highscore})


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
    total_time = 0

    while training_loop:
        stamp = time.perf_counter()

        # Execute step
        game_state = session.tick()

        # Reset if game over
        if game_state['game_over']:
            game_state = session.reset()

        total_time += (time.perf_counter() - stamp) * 1000

        # Emit update every 500ms
        if total_time > 500:
            total_time = 0
            emit('game_update', {
                'score': game_state['score'],
                'food_position': {'x': None, 'y': None},  # Don't render during fast training
                'snake_position': [{'x': None, 'y': None}],
                'game_over': False,
                'episode': session.episodes
            })

        # Emit highscore if it changed
        if session.highscore > highscore:
            highscore = session.highscore
            emit('set_highscore', {'highscore': session.highscore})

        # Yield to event loop
        socketio.sleep(0)


@socketio.on('stop_loop')
def handle_stop_loop():
    """Stop the AI training loop."""
    global training_loop
    training_loop = False


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
        - game: str - Game type (snake, tictactoe, etc.)
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
    game = data.get('game', 'snake')

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in '-_').strip()
    if not safe_name:
        safe_name = 'model'

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
        'game': game,
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
