from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from Session import Session

import time
import logging

# At the top of your Python file, before running the app
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only show errors, not INFO messages

# If using SocketIO
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

# Let Flask-SocketIO auto-pick async mode (will pick "eventlet" if installed)
socketio = SocketIO(app, cors_allowed_origins=["*"])
#prob want to make a instance of the game handler that calls the game script functions

session = None
loop = False

@socketio.on('init')
def handle_init(data):
    global session
    grid_size = data.get('grid_size')
    AI = data.get('AI_mode')
    Model = data.get('modelType')
    parms = data.get('params')
    session = Session(grid_size=grid_size, AI_mode=AI, model=Model, params=parms)
    score , food_position, snake_position, game_over = session.get_state()
    emit('game_update', {
        'score': score,
        'food_position': food_position,
        'snake_position': snake_position,
        'game_over': game_over,
        'episode': session.episodes
    })

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
    if session is None: return
    global loop
    loop = True
    highscore = session.highscore
    step = 0
    total_time = 0
    while loop: #change to condition
        step += 1
        stamp = time.perf_counter()
        score, food_position, snake_position, game_over = session.AI_step()
        total_time += (time.perf_counter() - stamp)*1000
        if game_over:
            score, food_position, snake_position = session.reset()
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
        socketio.sleep(0)  # Small delay to prevent overwhelming the client

@socketio.on('stop_loop')
def handle_stop_training():
    global loop
    loop = False


if __name__ == '__main__':
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=True, log_output=False)