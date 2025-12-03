from GameEngine import SnakeGame
from QNet import QNet
from PPO import PPO

from utils.device import get_device


class Session:
    """Controls a single game session, either manual or AI-driven."""

    def __init__(self, params, grid_size=10, AI_mode=False, model='qnet'):
        self.grid_size = grid_size
        self.game = SnakeGame(grid=grid_size, ai=AI_mode)
        self.highscore = 0
        self.episodes = 0
        self.device = get_device()
        ind, r, t, b = self.game.get_state()
        if AI_mode:
            if model == 'qnet':
                self.net = QNet({"conv": False, "indim": len(ind)}, params=params)
            else:
                self.net = PPO({"conv": False, "indim": len(ind)}, params=params)
            # Move model to best device
            if hasattr(self.net, 'to'):
                self.net.to(self.device)

    def reset(self):
        score, food_position, snake_position = self.game.reset(self.grid_size)
        return score, food_position, snake_position

    def step(self, action):
        score, food_position, snake_position, game_over = self.game.step(action)
        self.game.game_over = False
        if game_over:
            self.episodes += 1
        if score > self.highscore:
            self.highscore = score
        return score, food_position, snake_position, game_over

    def AI_step(self):
        state, target, game_over, board = self.game.get_state()
        action, self.episodes = self.net.select_action(state, board, target, game_over, device=self.device)
        self.game.game_over = False
        score, food_position, snake_position, game_over = self.game.step(action)
        if score > self.highscore:
            self.highscore = score
        return score, food_position, snake_position, game_over

    def get_state(self):
        score = self.game.score
        food_position = self.game.food_position
        snake_position = self.game.snake_position
        game_over = self.game.game_over
        return score, food_position, snake_position, game_over
