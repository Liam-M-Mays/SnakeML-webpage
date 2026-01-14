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

    # Default reward values (can be overridden by config)
    DEFAULT_REWARDS = {
        'apple': 1.0,
        'wall': -1.0,
        'self': -1.0,
        'starve': -1.0,
        'step': -0.001,
        'closer': 0.0,
        'farther': 0.0,
    }

    # Default input configuration
    DEFAULT_INPUTS = {
        'foodDirection': True,
        'pathDistance': True,
        'currentDirection': True,
        'hunger': True,
        'danger': True,
        'segments': True,
        'segmentCount': 10,
        'visionRange': 1,
        'snakeLength': False,
    }

    def __init__(self, grid_size: int = 10, vision: int = 1, seg_count: int = 10,
                 use_cnn: bool = False, starvation_limit: int = None,
                 random_start_state: bool = False, random_max_length: int = None,
                 inputs: dict = None, rewards: dict = None):
        """
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            vision: Sight range for danger detection (overridden by inputs.visionRange)
            seg_count: Number of body segments to track (overridden by inputs.segmentCount)
            use_cnn: Whether to generate board state for CNN input
            starvation_limit: Steps without food before death.
            random_start_state: If True, snake starts with random length and direction
            random_max_length: Max snake length for random start (default: grid^2 - 1)
            inputs: Input feature configuration dict
            rewards: Reward values configuration dict
        """
        self.grid_size = grid_size
        self.use_cnn = use_cnn
        self.random_start_state = random_start_state
        self.random_max_length = random_max_length if random_max_length else (grid_size ** 2 - 1)

        # Merge provided configs with defaults
        self.input_config = {**self.DEFAULT_INPUTS, **(inputs or {})}
        self.reward_config = {**self.DEFAULT_REWARDS, **(rewards or {})}

        # Use input config for vision and segment count
        self.vision = self.input_config.get('visionRange', vision)
        self.seg_count = self.input_config.get('segmentCount', seg_count)

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

    def set_random_start_state(self, enabled: bool):
        """Enable or disable random start state (can be changed during gameplay)."""
        self.random_start_state = enabled

    def set_random_max_length(self, max_length: int):
        """Set max snake length for random start (can be changed during gameplay)."""
        max_possible = self.grid_size ** 2 - 1
        self.random_max_length = min(max(1, max_length), max_possible)

    def reset(self) -> dict:
        """Reset game to initial state."""
        if self.random_start_state:
            # Max length is configurable, default to grid^2 - 1 (theoretical max)
            max_possible = self.grid_size ** 2 - 1
            max_len = min(self.random_max_length, max_possible)
            is_even_grid = self.grid_size % 2 == 0

            # Retry loop for odd grids where winnability isn't guaranteed
            max_attempts = 20
            for attempt in range(max_attempts):
                length = random.randint(1, max(1, max_len))

                # Generate snake on Hamiltonian cycle/path
                self.snake = self._generate_random_snake(length)

                # Set direction based on cycle movement
                # Head moves "forward" on the cycle
                if len(self.snake) > 1:
                    head = self.snake[0]
                    second = self.snake[1]
                    # Direction is from second to head (where we're going)
                    self.direction = [head['x'] - second['x'], head['y'] - second['y']]
                else:
                    # Single segment - use cycle to determine direction
                    cycle = self._hamiltonian_cycle
                    head_idx = self._snake_head_cycle_idx
                    cycle_len = len(cycle)

                    if is_even_grid:
                        # True cycle: use modulo
                        next_idx = (head_idx + 1) % cycle_len
                    else:
                        # Path: don't wrap around
                        next_idx = min(head_idx + 1, cycle_len - 1)

                    next_pos = cycle[next_idx]
                    head = self.snake[0]
                    self.direction = [next_pos[0] - head['x'], next_pos[1] - head['y']]

                # Place food on the cycle
                self.food = self._place_food_on_cycle()

                # For even grids, cycle guarantees winnability
                # For odd grids, verify with BFS and retry if needed
                if is_even_grid or self._basic_reachability_check():
                    break

            # Score reflects actual snake length (minus starting 1)
            self.score = len(self.snake) - 1
        else:
            # Normal start: single segment at random position
            start_x = random.randint(0, self.grid_size - 1)
            start_y = random.randint(0, self.grid_size - 1)
            self.snake = [{'x': start_x, 'y': start_y}]
            self.direction = [1, 0]  # Start going right
            self.score = 0
            self.food = self._place_food()
            # Clear cycle info for normal mode
            if hasattr(self, '_hamiltonian_cycle'):
                del self._hamiltonian_cycle
            if hasattr(self, '_snake_head_cycle_idx'):
                del self._snake_head_cycle_idx
            if hasattr(self, '_is_true_cycle'):
                del self._is_true_cycle

        self.hunger = 0
        self.game_over = False
        self.last_reward = 0.0

        return self._get_frontend_state()

    def _get_hamiltonian_cycle(self) -> list:
        """
        Generate a Hamiltonian cycle/path for the grid.

        For EVEN grid sizes: generates a true cycle where last cell
        is adjacent to first.

        For ODD grid sizes: a true Hamiltonian cycle doesn't exist,
        so we generate a path and handle it carefully.

        Returns list of (x, y) tuples representing the path/cycle in order.
        """
        gs = self.grid_size
        cycle = []

        if gs % 2 == 0:
            # EVEN grid: true Hamiltonian cycle exists
            # Pattern: snake through all rows, column 0 reserved for return
            #
            # For gs=4:
            # Row 0: (0,0) -> (3,0)     [right]
            # Row 1: (3,1) -> (1,1)     [left, skip col 0]
            # Row 2: (1,2) -> (3,2)     [right]
            # Row 3: (3,3) -> (1,3)     [left, skip col 0]
            # Col 0: (0,3) -> (0,1)     [up, closing the cycle]
            #
            # Row 0: all columns
            for x in range(gs):
                cycle.append((x, 0))

            # Rows 1 to gs-1: zigzag, skipping column 0
            for y in range(1, gs):
                if y % 2 == 1:  # Odd row: right to left
                    for x in range(gs - 1, 0, -1):
                        cycle.append((x, y))
                else:  # Even row: left to right
                    for x in range(1, gs):
                        cycle.append((x, y))

            # Column 0 from bottom to row 1 (closes cycle)
            for y in range(gs - 1, 0, -1):
                cycle.append((0, y))

        else:
            # ODD grid: no true Hamiltonian cycle exists
            # Use simple zigzag path covering all cells
            # Snake placement will avoid wrapping issues
            for y in range(gs):
                if y % 2 == 0:  # Even row: left to right
                    for x in range(gs):
                        cycle.append((x, y))
                else:  # Odd row: right to left
                    for x in range(gs - 1, -1, -1):
                        cycle.append((x, y))

        return cycle

    def _generate_random_snake(self, target_length: int) -> list:
        """
        Generate a truly winnable snake configuration.

        The snake is placed along a Hamiltonian cycle/path, guaranteeing that:
        1. The snake can always reach food by following the path
        2. After eating, the snake remains on the path
        3. For even grids (true cycle): game can always reach max length
        4. For odd grids (path): validated with BFS for winnability

        This is the key insight: if the snake always follows consecutive
        positions on a Hamiltonian path, it minimizes self-trapping.
        """
        cycle = self._get_hamiltonian_cycle()
        cycle_len = len(cycle)
        is_true_cycle = self.grid_size % 2 == 0

        # Clamp target length to valid range
        target_length = min(target_length, cycle_len - 1)  # Leave room for food
        target_length = max(1, target_length)

        # Pick a random starting position on the cycle for the HEAD
        if is_true_cycle:
            # True cycle: can wrap around safely
            head_idx = random.randint(0, cycle_len - 1)
        else:
            # Path (odd grid): head must be far enough so body doesn't go negative
            # Also leave some room at the end for maneuvering
            min_head_idx = target_length - 1
            max_head_idx = cycle_len - 1
            head_idx = random.randint(min_head_idx, max_head_idx)

        # Build snake as consecutive positions on the cycle
        # Head is at head_idx, body follows "behind" on the cycle
        snake = []
        for i in range(target_length):
            if is_true_cycle:
                # True cycle: use modulo for wrap-around
                idx = (head_idx - i) % cycle_len
            else:
                # Path: no wrap-around
                idx = head_idx - i
            x, y = cycle[idx]
            snake.append({'x': x, 'y': y})

        # Store cycle info for food placement
        self._hamiltonian_cycle = cycle
        self._snake_head_cycle_idx = head_idx
        self._is_true_cycle = is_true_cycle

        return snake

    def _place_food_on_cycle(self) -> dict:
        """
        Place food on an empty cell that's on the Hamiltonian cycle/path.

        For true cycles (even grids): any empty position works.
        For paths (odd grids): prefer positions "ahead" of the head on the path.

        This ensures the snake can always reach the food by following
        the path, maintaining winnability.
        """
        if not hasattr(self, '_hamiltonian_cycle'):
            # Fallback to regular placement if no cycle
            return self._place_food()

        cycle = self._hamiltonian_cycle
        cycle_len = len(cycle)
        snake_set = {(s['x'], s['y']) for s in self.snake}
        head_idx = self._snake_head_cycle_idx
        is_true_cycle = getattr(self, '_is_true_cycle', self.grid_size % 2 == 0)

        # Find all empty positions on the cycle
        empty_positions = [(i, x, y) for i, (x, y) in enumerate(cycle)
                          if (x, y) not in snake_set]

        if not empty_positions:
            # Shouldn't happen, but fallback
            return self._place_food()

        if is_true_cycle:
            # True cycle: any position works, pick randomly
            _, x, y = random.choice(empty_positions)
        else:
            # Path: prefer positions "ahead" of the head (higher index)
            # This gives the snake more room to maneuver
            ahead_positions = [(i, x, y) for (i, x, y) in empty_positions
                              if i > head_idx]
            if ahead_positions:
                _, x, y = random.choice(ahead_positions)
            else:
                # No positions ahead, use any empty position
                _, x, y = random.choice(empty_positions)

        return {'x': x, 'y': y}

    def _is_winnable(self) -> bool:
        """
        Check if the current state is truly winnable (can reach max length).

        For true cycles (even grids): always winnable if snake on consecutive cycle positions.
        For paths (odd grids): check basic reachability with BFS.
        """
        if not hasattr(self, '_hamiltonian_cycle'):
            # No cycle info - do basic reachability check
            return self._basic_reachability_check()

        cycle = self._hamiltonian_cycle
        cycle_len = len(cycle)
        is_true_cycle = getattr(self, '_is_true_cycle', self.grid_size % 2 == 0)

        # Build position to cycle index mapping
        pos_to_idx = {pos: i for i, pos in enumerate(cycle)}

        # Check if all snake segments are on consecutive cycle positions
        snake_indices = []
        for seg in self.snake:
            pos = (seg['x'], seg['y'])
            if pos not in pos_to_idx:
                return False  # Segment not on cycle
            snake_indices.append(pos_to_idx[pos])

        # Check consecutiveness
        if len(snake_indices) <= 1:
            return True

        head_idx = snake_indices[0]
        for i, idx in enumerate(snake_indices):
            if is_true_cycle:
                # True cycle: allow wrap-around with modulo
                expected_idx = (head_idx - i) % cycle_len
            else:
                # Path: no wrap-around allowed
                expected_idx = head_idx - i
                if expected_idx < 0:
                    return False  # Would wrap around - invalid for path
            if idx != expected_idx:
                return False  # Not consecutive on cycle/path

        # Check food is on cycle and not on snake
        food_pos = (self.food['x'], self.food['y'])
        if food_pos not in pos_to_idx:
            return False
        if food_pos in {(s['x'], s['y']) for s in self.snake}:
            return False

        # For odd grids, do additional BFS check since path isn't a true cycle
        if not is_true_cycle:
            return self._basic_reachability_check()

        return True

    def _basic_reachability_check(self) -> bool:
        """Basic BFS check if head can reach food (fallback)."""
        head = self.snake[0]
        food_pos = (self.food['x'], self.food['y'])
        head_pos = (head['x'], head['y'])

        if head_pos == food_pos:
            return True

        snake_set = {(s['x'], s['y']) for s in self.snake}
        visited = {head_pos}
        queue = [head_pos]

        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == food_pos:
                    return True
                if (0 <= nx < self.grid_size and
                    0 <= ny < self.grid_size and
                    (nx, ny) not in visited and
                    (nx, ny) not in snake_set):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def _shortest_path_to_food(self) -> int:
        """
        Calculate shortest path length from head to food using BFS.

        Returns:
            Path length (number of steps), or grid^2 if unreachable
        """
        dist, _ = self._get_shortest_path_with_cells()
        return dist

    def _get_shortest_path_with_cells(self) -> tuple:
        """
        Calculate shortest path AND return the path cells.

        This is the single source of truth for path calculation.
        Used by both the network input and debug visualization.

        Returns:
            (distance, path_cells)
            - distance: int (number of steps, or grid^2 if unreachable)
            - path_cells: list of {'x': int, 'y': int} from head to food (excluding head)
        """
        head = self.snake[0]
        food_pos = (self.food['x'], self.food['y'])
        head_pos = (head['x'], head['y'])

        # Quick check: already at food
        if head_pos == food_pos:
            return 0, []

        # BFS with parent tracking for path reconstruction
        snake_set = {(s['x'], s['y']) for s in self.snake[1:]}  # Exclude head
        visited = {head_pos: 0}
        parent = {head_pos: None}  # Track parents for path reconstruction
        queue = [(head_pos, 0)]

        found = False
        while queue:
            (x, y), dist = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) == food_pos:
                    parent[(nx, ny)] = (x, y)
                    visited[(nx, ny)] = dist + 1
                    found = True
                    break
                if (0 <= nx < self.grid_size and
                    0 <= ny < self.grid_size and
                    (nx, ny) not in visited and
                    (nx, ny) not in snake_set):
                    visited[(nx, ny)] = dist + 1
                    parent[(nx, ny)] = (x, y)
                    queue.append(((nx, ny), dist + 1))
            if found:
                break

        if not found:
            # Unreachable - return large value and empty path
            return self.grid_size ** 2, []

        # Reconstruct path from food back to head
        path = []
        current = food_pos
        while current != head_pos:
            path.append({'x': current[0], 'y': current[1]})
            current = parent[current]

        # Reverse to get path from head to food (excluding head itself)
        path.reverse()
        return len(path), path

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

        # Get reward values from config
        r_apple = self.reward_config.get('apple', 1.0)
        r_wall = self.reward_config.get('wall', -1.0)
        r_self = self.reward_config.get('self', -1.0)
        r_starve = self.reward_config.get('starve', -1.0)
        r_step = self.reward_config.get('step', -0.001)
        r_closer = self.reward_config.get('closer', 0.0)
        r_farther = self.reward_config.get('farther', 0.0)

        # Check wall collision
        if (head['x'] < 0 or head['x'] >= self.grid_size or
            head['y'] < 0 or head['y'] >= self.grid_size):
            self.game_over = True
            self.last_reward = r_wall
            return self._get_frontend_state(), self.last_reward, True, {'reason': 'wall'}

        # Check self collision
        if any(seg['x'] == head['x'] and seg['y'] == head['y'] for seg in self.snake):
            self.game_over = True
            self.last_reward = r_self
            return self._get_frontend_state(), self.last_reward, True, {'reason': 'self'}

        # Move snake: insert new head
        self.snake.insert(0, head)

        # Check food
        if head['x'] == self.food['x'] and head['y'] == self.food['y']:
            self.score += 1
            self.last_reward = r_apple
            self.hunger = 0
            self.food = self._place_food()
            # Don't pop tail - snake grows
        else:
            # Calculate distance after moving
            new_dist = abs(head['x'] - self.food['x']) + abs(head['y'] - self.food['y'])

            # Reward based on distance change + step penalty
            if new_dist < old_dist:
                self.last_reward = r_step + r_closer
            else:
                self.last_reward = r_step + r_farther

            self.hunger += 1

            # Check starvation
            if self.hunger >= self.starvation_limit:
                self.game_over = True
                self.last_reward = r_starve
                return self._get_frontend_state(), self.last_reward, True, {'reason': 'starve'}

            # Remove tail - snake moves
            self.snake.pop()

        return self._get_frontend_state(), self.last_reward, False, {}

    def get_state_for_network(self) -> tuple:
        """
        Get state representation for neural network input.

        Features are conditionally included based on input_config.

        Returns:
            (flat_state, board_state)
            - flat_state: List of normalized features
            - board_state: 2D grid for CNN (if use_cnn=True) or empty list
        """
        head = self.snake[0]
        gs = self.grid_size  # Shorthand
        cfg = self.input_config

        flat_state = []

        # Food direction (unit vector style: -1, 0, or 1)
        if cfg.get('foodDirection', True):
            raw_dx = self.food['x'] - head['x']
            raw_dy = self.food['y'] - head['y']
            food_dir_x = 0 if raw_dx == 0 else (1 if raw_dx > 0 else -1)
            food_dir_y = 0 if raw_dy == 0 else (1 if raw_dy > 0 else -1)
            flat_state.extend([float(food_dir_x), float(food_dir_y)])

        # Path distance (BFS shortest path)
        if cfg.get('pathDistance', True):
            path_dist = self._shortest_path_to_food()
            path_dist_norm = path_dist / (gs * gs)  # Normalize by max possible
            flat_state.append(path_dist_norm)

        # Current direction
        if cfg.get('currentDirection', True):
            dir_x, dir_y = self.direction
            flat_state.extend([float(dir_x), float(dir_y)])

        # Hunger (normalized)
        if cfg.get('hunger', True):
            hunger_norm = self.hunger / self.starvation_limit
            flat_state.append(hunger_norm)

        # Body segment positions (relative to head, normalized)
        if cfg.get('segments', True):
            segments = self._get_segment_features()
            flat_state.extend([s / gs for s in segments])

        # Danger detection
        if cfg.get('danger', True):
            danger = self._get_danger()
            flat_state.extend(danger)

        # Snake length (normalized by max possible length)
        if cfg.get('snakeLength', False):
            max_length = gs * gs
            length_norm = len(self.snake) / max_length
            flat_state.append(length_norm)

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
        """Get state dict for frontend display (internal use)."""
        return {
            'score': self.score,
            'food_position': self.food.copy(),
            'snake_position': [seg.copy() for seg in self.snake],
            'game_over': self.game_over
        }

    def get_frontend_state(self) -> dict:
        """
        Get game state in format suitable for frontend display.

        Implements the GameEnv abstract method. The structure is snake-specific
        and the frontend snake module knows how to interpret it.

        Returns:
            dict: Snake game state for frontend rendering
        """
        return self._get_frontend_state()

    def get_debug_info(self, debug_vision: bool = False, debug_path: bool = False,
                       debug_segments: bool = False) -> dict:
        """
        Get debug visualization data.

        Args:
            debug_vision: Whether to include danger detection cells
            debug_path: Whether to include shortest path cells
            debug_segments: Whether to include segment tracking visualization

        Returns:
            dict with debug info (danger_cells, path_cells, segment_cells, segment_connections)
        """
        result = {}

        if debug_vision and self.input_config.get('danger', True):
            _, danger_cells = self._get_danger_with_cells()
            result['danger_cells'] = danger_cells

        if debug_path and self.input_config.get('pathDistance', True):
            _, path_cells = self._get_shortest_path_with_cells()
            result['path_cells'] = path_cells

        if debug_segments and self.input_config.get('segments', True):
            _, segment_cells, segment_connections = self._get_segment_features_with_cells()
            result['segment_cells'] = segment_cells
            result['segment_connections'] = segment_connections

        return result

    def _get_danger(self) -> list:
        """
        Get danger information based on vision setting.

        Returns list of danger values (1=danger, 0=safe).
        """
        danger, _ = self._get_danger_with_cells()
        return danger

    def _get_danger_with_cells(self) -> tuple:
        """
        Get danger information AND the cells being checked.

        This is the single source of truth for danger detection.
        Used by both the network input and debug visualization.

        Returns:
            (danger_values, cells_info)
            - danger_values: list of 0/1 values
            - cells_info: list of {'x': int, 'y': int, 'danger': bool, 'label': str}
        """
        head = self.snake[0]
        snake_set = {(s['x'], s['y']) for s in self.snake[1:]}
        cells_info = []

        def is_danger(x, y):
            """Check if a cell is dangerous (wall or body)."""
            out_of_bounds = (x < 0 or y < 0 or
                            x >= self.grid_size or y >= self.grid_size)
            hits_body = (x, y) in snake_set
            return out_of_bounds or hits_body

        if self.vision == 0:
            # Relative directions: [left, right, forward]
            # Based on snake's current direction
            dx, dy = self.direction

            # Calculate relative direction vectors
            # Forward: same as current direction
            # Right (turn right): [-dy, dx]
            # Left (turn left): [dy, -dx]
            forward_cell = (head['x'] + dx, head['y'] + dy)
            right_cell = (head['x'] - dy, head['y'] + dx)
            left_cell = (head['x'] + dy, head['y'] - dx)

            # Check each direction
            left_danger = is_danger(*left_cell)
            right_danger = is_danger(*right_cell)
            forward_danger = is_danger(*forward_cell)

            danger = [
                1 if left_danger else 0,
                1 if right_danger else 0,
                1 if forward_danger else 0
            ]

            # Build cells info for visualization
            cells_info = [
                {'x': left_cell[0], 'y': left_cell[1], 'danger': left_danger, 'label': 'L'},
                {'x': right_cell[0], 'y': right_cell[1], 'danger': right_danger, 'label': 'R'},
                {'x': forward_cell[0], 'y': forward_cell[1], 'danger': forward_danger, 'label': 'F'},
            ]

            return danger, cells_info

        elif self.vision > 0:
            # Window around head
            danger = []
            for v_dy in range(-self.vision, self.vision + 1):
                for v_dx in range(-self.vision, self.vision + 1):
                    if v_dx == 0 and v_dy == 0:
                        continue  # Skip head position

                    nx = head['x'] + v_dx
                    ny = head['y'] + v_dy

                    cell_danger = is_danger(nx, ny)
                    danger.append(1 if cell_danger else 0)
                    cells_info.append({
                        'x': nx, 'y': ny,
                        'danger': cell_danger,
                        'label': ''
                    })

            return danger, cells_info

        else:
            # Full board view (not typically used for debug viz)
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
            return danger, []  # No cells_info for full board mode

    def _get_segment_features(self) -> list:
        """
        Get relative positions of body segments for state.

        Returns list of (dx, dy) differences between segments.
        """
        features, _, _ = self._get_segment_features_with_cells()
        return features

    def _get_segment_features_with_cells(self) -> tuple:
        """
        Get segment features AND visualization data.

        This is the single source of truth for segment tracking.
        Used by both the network input and debug visualization.

        Returns:
            (features, cells, connections)
            - features: list of dx, dy values (the actual network input)
            - cells: list of {'x': int, 'y': int, 'label': str} for tracked segments
            - connections: list of {'from': {x, y}, 'to': {x, y}, 'dx': int, 'dy': int}
        """
        gs = self.grid_size

        if len(self.snake) <= 1:
            # No body yet, return zeros and empty visualization
            return [0, 0] * self.seg_count, [], []

        features = []
        cells = []
        connections = []
        seg_length = max(1, (len(self.snake) - 1) // self.seg_count)

        # Start with head
        prev_x = self.snake[0]['x']
        prev_y = self.snake[0]['y']
        cells.append({'x': prev_x, 'y': prev_y, 'label': 'H'})

        for i in range(1, self.seg_count):
            idx = min(i * seg_length, len(self.snake) - 1)
            curr_x = self.snake[idx]['x']
            curr_y = self.snake[idx]['y']
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            features.extend([dx, dy])

            cells.append({'x': curr_x, 'y': curr_y, 'label': str(i)})
            connections.append({
                'from': {'x': prev_x, 'y': prev_y},
                'to': {'x': curr_x, 'y': curr_y},
                'dx': dx / gs,  # Normalized values (what network sees)
                'dy': dy / gs
            })

            prev_x = curr_x
            prev_y = curr_y

        # Add tail
        tail_x = self.snake[-1]['x']
        tail_y = self.snake[-1]['y']
        dx = tail_x - prev_x
        dy = tail_y - prev_y
        features.extend([dx, dy])

        cells.append({'x': tail_x, 'y': tail_y, 'label': 'T'})
        connections.append({
            'from': {'x': prev_x, 'y': prev_y},
            'to': {'x': tail_x, 'y': tail_y},
            'dx': dx / gs,
            'dy': dy / gs
        })

        return features, cells, connections

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
