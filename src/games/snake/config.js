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
    colors: {
      snake: '#4CAF50',
      food: '#ff5252',
      background: '#222',
    },
  },

  // Game capabilities
  supportsHumanPlay: true,
  supportsAIvsHuman: false,  // AI and human don't play together in snake
  isRealtime: true,          // Continuous game loop, not turn-based

  // Network action count (for AI)
  actionCount: 3,  // forward, turn-right, turn-left
}
