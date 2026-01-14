// Snake game configuration
export default {
  id: 'snake',
  name: 'Snake',
  description: 'Classic snake game - eat food, grow longer, avoid walls and yourself',

  // Theme colors for this game
  theme: {
    accent: '#4CAF50',
    accentHover: '#45a049',
    accentLight: '#81C784',
    headerText: '#4CAF50',
    panelHighlight: '#235425',
  },

  // Default game settings
  defaultSettings: {
    gridSize: 10,
    boardSize: 400,
    randomStartState: false,  // When true, snake starts with random length/direction
    randomMaxLength: 99,      // Max random snake length (default: grid^2 - 1)
    colors: {
      snake: '#4CAF50',
      food: '#ff5252',
      background: '#222',
    },
    // Network input configuration
    inputs: {
      foodDirection: true,    // Direction to food (-1, 0, 1)
      pathDistance: true,     // BFS shortest path distance
      currentDirection: true, // Snake's movement direction
      hunger: true,           // Steps since last food
      danger: true,           // Wall/body detection
      segments: true,         // Body segment tracking
      segmentCount: 10,       // Number of body segments to track
      visionRange: 1,         // 0=adjacent, 1-3=window
      snakeLength: false,     // Normalized snake length
    },
    // Reward values
    rewards: {
      apple: 1.0,
      wall: -1.0,
      self: -1.0,
      starve: -1.0,
      step: -0.001,
      closer: 0,
      farther: 0,
    },
    // Debug visualization settings
    debug: {
      vision: false,   // Show danger detection cells
      path: false,     // Show shortest path to food
      segments: false, // Show segment tracking
    },
  },

  // Game capabilities
  supportsHumanPlay: true,
  supportsAIvsHuman: false,  // AI and human don't play together in snake
  isRealtime: true,          // Continuous game loop, not turn-based

  // Network action count (for AI)
  actionCount: 3,  // forward, turn-right, turn-left

  // UI hints
  hasKeyboardControls: true,
  hasClickControls: false,
  hasInputVisualization: true,

  // Instructions text
  humanInstructions: 'Use arrow keys to control the snake',
  trainingInstructions: 'Set speed to 0 for maximum training speed',

  // Stats display configuration
  // Each game defines which stats to show and how to format them
  statsConfig: {
    // Stats from controller: { score, episode, highscore }
    layout: [
      { key: 'episode', label: 'Episode' },
      { key: 'score', label: 'Score' },
      { key: 'highscore', label: 'Highscore', disabledWhen: 'randomStartState' },
    ],
  },
}
