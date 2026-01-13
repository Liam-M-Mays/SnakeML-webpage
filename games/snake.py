"""
Snake game environment.

Actions (relative to current direction):
    0 = Go straight (forward)
    1 = Turn right
    2 = Turn left
"""
import random
from .base import GameEnv


class SnakeEnv(GameEnv):
    """
    Snake game environment implementing the GameEnv interface.

    The snake moves on a grid, eating food to grow.
    Game ends when hitting a wall or itself.
    """

    # Reward values
    REWARD_APPLE = 1.0
    REWARD_WALL = -1.0
    REWARD_SELF = -1.0
    REWARD_STARVE = -1.0
    REWARD_STEP = -0.01  # Base penalty per step
    REWARD_CLOSER = 0.00  # Bonus for moving toward food
    REWARD_FARTHER = -0.00  # Penalty for moving away from food

    def __init__(self, grid_size: int = 10, vision: int = 1, seg_count: int = 5,
                 use_cnn: bool = False, starvation_limit: int = None):
        """
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            vision: Sight range for danger detection
                   0 = only immediate neighbors (4 values)
                   >0 = window of (2*vision+1)^2 - 1 cells around head
                   <0 = full board view
            seg_count: Number of body segments to track in state (for non-CNN)
            use_cnn: Whether to generate board state for CNN input
            starvation_limit: Steps without food before death.
                             Default is grid_size^2 for AI, very high for human.
        """
        self.grid_size = grid_size
        self.vision = vision
        self.seg_count = seg_count
        self.use_cnn = use_cnn

        # Starvation: AI gets limited steps to encourage efficiency
        if starvation_limit is None:
            self.starvation_limit = grid_size ** 2
        else:
            self.starvation_limit = starvation_limit

        # Game state
        self.snake = []           # List of {x, y} dicts, head first
        self.food = {'x': 0, 'y': 0}
        self.direction = [1, 0]   # [dx, dy], starts going right
        self.score = 0
        self.hunger = 0           # Steps since last food
        self.game_over = False
        self.last_reward = 0.0

        self.reset()

    @property
    def action_count(self) -> int:
        """Snake has 3 actions: forward, turn right, turn left."""
        return 3

    def reset(self) -> dict:
        """Reset game to initial state."""
        # Random starting position
        start_x = random.randint(0, self.grid_size - 1)
        start_y = random.randint(0, self.grid_size - 1)
        self.snake = [{'x': start_x, 'y': start_y}]

        self.food = self._place_food()
        self.direction = [1, 0]  # Start going right
        self.score = 0
        self.hunger = 0
        self.game_over = False
        self.last_reward = 0.0

        return self._get_frontend_state()

    def step(self, action: int) -> tuple:
        """
        Execute one game step.

        Args:
            action: 0=forward, 1=turn right, 2=turn left

        Returns:
            (state_dict, reward, done, info)
        """
        if self.game_over:
            return self._get_frontend_state(), 0.0, True, {}

        # Calculate distance to food BEFORE moving
        old_head = self.snake[0]
        old_dist = abs(old_head['x'] - self.food['x']) + abs(old_head['y'] - self.food['y'])

        # Calculate new head position based on action
        head = self.snake[0].copy()
        dx, dy = self._get_movement(action)
        head['x'] += dx
        head['y'] += dy

        # Check wall collision
        if (head['x'] < 0 or head['x'] >= self.grid_size or
            head['y'] < 0 or head['y'] >= self.grid_size):
            self.game_over = True
            self.last_reward = self.REWARD_WALL
            return self._get_frontend_state(), self.last_reward, True, {'reason': 'wall'}

        # Check self collision
        if any(seg['x'] == head['x'] and seg['y'] == head['y'] for seg in self.snake):
            self.game_over = True
            self.last_reward = self.REWARD_SELF
            return self._get_frontend_state(), self.last_reward, True, {'reason': 'self'}

        # Move snake: insert new head
        self.snake.insert(0, head)

        # Check food
        if head['x'] == self.food['x'] and head['y'] == self.food['y']:
            self.score += 1
            self.last_reward = self.REWARD_APPLE
            self.hunger = 0
            self.food = self._place_food()
            # Don't pop tail - snake grows
        else:
            # Calculate distance after moving
            new_dist = abs(head['x'] - self.food['x']) + abs(head['y'] - self.food['y'])

            # Reward based on distance change + step penalty
            if new_dist < old_dist:
                self.last_reward = self.REWARD_STEP + self.REWARD_CLOSER
            else:
                self.last_reward = self.REWARD_STEP + self.REWARD_FARTHER

            self.hunger += 1

            # Check starvation
            if self.hunger >= self.starvation_limit:
                self.game_over = True
                self.last_reward = self.REWARD_STARVE
                return self._get_frontend_state(), self.last_reward, True, {'reason': 'starve'}

            # Remove tail - snake moves
            self.snake.pop()

        return self._get_frontend_state(), self.last_reward, False, {}

    def get_state_for_network(self) -> tuple:
        """
        Get state representation for neural network input.

        Returns:
            (flat_state, board_state)
            - flat_state: List of normalized features
            - board_state: 2D grid for CNN (if use_cnn=True) or empty list
        """
        head = self.snake[0]
        gs = self.grid_size  # Shorthand

        # Relative food position (normalized)
        food_dx = (self.food['x'] - head['x']) / gs
        food_dy = (self.food['y'] - head['y']) / gs

        # Current direction
        dir_x, dir_y = self.direction

        # Hunger (normalized)
        hunger_norm = self.hunger / self.starvation_limit

        # Danger detection
        danger = self._get_danger()

        # Body segment positions (relative to head, normalized)
        segments = self._get_segment_features()

        # Build flat state
        flat_state = [
            food_dx,
            food_dy,
            float(dir_x),
            float(dir_y),
            hunger_norm,
        ]
        flat_state.extend([s / gs for s in segments])
        flat_state.extend(danger)

        # Board state for CNN
        if self.use_cnn:
            board_state = self._get_board_channels()
        else:
            board_state = [[]]

        return flat_state, board_state

    def _get_movement(self, action: int) -> tuple:
        """
        Convert action to movement delta and update direction.

        Args:
            action: 0=forward, 1=right, 2=left

        Returns:
            (dx, dy) movement
        """
        dx, dy = self.direction

        if action == 0:  # Forward - no direction change
            pass
        elif action == 1:  # Turn right
            # Right turn: (1,0)->(0,1), (0,1)->(-1,0), (-1,0)->(0,-1), (0,-1)->(1,0)
            dx, dy = -dy, dx
            self.direction = [dx, dy]
        elif action == 2:  # Turn left
            # Left turn: (1,0)->(0,-1), (0,-1)->(-1,0), (-1,0)->(0,1), (0,1)->(1,0)
            dx, dy = dy, -dx
            self.direction = [dx, dy]

        return dx, dy

    def _place_food(self) -> dict:
        """Place food in a random empty cell."""
        while True:
            pos = {
                'x': random.randint(0, self.grid_size - 1),
                'y': random.randint(0, self.grid_size - 1)
            }
            # Check it's not on the snake
            if not any(seg['x'] == pos['x'] and seg['y'] == pos['y'] for seg in self.snake):
                return pos

    def _get_frontend_state(self) -> dict:
        """Get state dict for frontend display."""
        return {
            'score': self.score,
            'food_position': self.food.copy(),
            'snake_position': [seg.copy() for seg in self.snake],
            'game_over': self.game_over
        }

    def _get_danger(self) -> list:
        """
        Get danger information based on vision setting.

        Returns list of danger values (1=danger, 0=safe).
        """
        head = self.snake[0]
        snake_set = {(s['x'], s['y']) for s in self.snake[1:]}

        if self.vision == 0:
            # Immediate neighbors only: [up, down, left, right]
            danger = [0, 0, 0, 0]

            # Walls
            if head['y'] == 0:
                danger[0] = 1  # Up is wall
            if head['y'] == self.grid_size - 1:
                danger[1] = 1  # Down is wall
            if head['x'] == 0:
                danger[2] = 1  # Left is wall
            if head['x'] == self.grid_size - 1:
                danger[3] = 1  # Right is wall

            # Body
            if (head['x'], head['y'] - 1) in snake_set:
                danger[0] = 1
            if (head['x'], head['y'] + 1) in snake_set:
                danger[1] = 1
            if (head['x'] - 1, head['y']) in snake_set:
                danger[2] = 1
            if (head['x'] + 1, head['y']) in snake_set:
                danger[3] = 1

            return danger

        elif self.vision > 0:
            # Window around head
            danger = []
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip head position

                    nx = head['x'] + dx
                    ny = head['y'] + dy

                    out_of_bounds = (nx < 0 or ny < 0 or
                                    nx >= self.grid_size or ny >= self.grid_size)
                    hits_body = (nx, ny) in snake_set
                    danger.append(1 if out_of_bounds or hits_body else 0)

            return danger

        else:
            # Full board view
            danger = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if head['x'] == x and head['y'] == y:
                        danger.append(-1)  # Head
                    elif (x, y) in snake_set:
                        danger.append(-0.5)  # Body
                    elif self.food['x'] == x and self.food['y'] == y:
                        danger.append(1)  # Food
                    else:
                        danger.append(0)  # Empty
            return danger

    def _get_segment_features(self) -> list:
        """
        Get relative positions of body segments for state.

        Returns list of (dx, dy) differences between segments.
        """
        if len(self.snake) <= 1:
            # No body yet, return zeros
            return [0, 0] * self.seg_count

        segments = []
        seg_length = max(1, (len(self.snake) - 1) // self.seg_count)

        prev_x = self.snake[0]['x']
        prev_y = self.snake[0]['y']

        for i in range(1, self.seg_count):
            idx = min(i * seg_length, len(self.snake) - 1)
            dx = self.snake[idx]['x'] - prev_x
            dy = self.snake[idx]['y'] - prev_y
            segments.extend([dx, dy])
            prev_x = self.snake[idx]['x']
            prev_y = self.snake[idx]['y']

        # Add tail
        dx = self.snake[-1]['x'] - prev_x
        dy = self.snake[-1]['y'] - prev_y
        segments.extend([dx, dy])

        return segments

    def _get_board_channels(self) -> list:
        """
        Get board representation for CNN input.

        Returns 2D grid with values:
            -1 = head
            -0.5 = body
            1 = food
            0 = empty
        """
        board = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Food
        board[self.food['y']][self.food['x']] = 1

        # Head
        head = self.snake[0]
        board[head['y']][head['x']] = -1

        # Body
        for seg in self.snake[1:]:
            board[seg['y']][seg['x']] = -0.5

        return [board]  # Wrap in list for channel dimension
