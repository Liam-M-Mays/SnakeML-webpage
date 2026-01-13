// TicTacToe game configuration (placeholder for future)
export default {
  id: 'tictactoe',
  name: 'Tic-Tac-Toe',
  description: 'Classic tic-tac-toe - get three in a row',

  // Theme colors - gray scheme
  theme: {
    accent: '#78909C',        // Blue-gray
    accentHover: '#90A4AE',
    accentLight: '#B0BEC5',
    headerText: '#B0BEC5',
    panelHighlight: '#37474F',
  },

  // Default game settings
  defaultSettings: {
    boardSize: 400,
    opponentType: 'ai',  // 'ai', 'human', 'minimax'
    aiGoesFirst: false,
    colors: {
      x: '#64B5F6',
      o: '#FF8A65',
      background: '#263238',
      line: '#546E7A',
    },
  },

  // Game capabilities
  supportsHumanPlay: true,
  supportsAIvsHuman: true,   // Human can play against AI
  isRealtime: false,         // Turn-based, not continuous

  // Network action count
  actionCount: 9,  // 9 board positions

  // Flag that this isn't implemented yet
  comingSoon: true,
}
