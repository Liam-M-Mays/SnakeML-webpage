// TicTacToe game configuration
export default {
  id: 'tictactoe',
  name: 'Tic-Tac-Toe',
  description: 'Classic tic-tac-toe - get three in a row',

  // Theme colors - blue-gray scheme
  theme: {
    accent: '#78909C',
    accentHover: '#90A4AE',
    accentLight: '#B0BEC5',
    headerText: '#B0BEC5',
    panelHighlight: '#37474F',
  },

  // Default game settings
  defaultSettings: {
    boardSize: 400,
    opponentType: 'random',  // 'human', 'random', 'ai'
    aiPlayer: 'O',           // Which player AI controls
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
  supportsAIvsAI: true,      // Can watch two AIs compete
  isRealtime: false,         // Turn-based, not continuous

  // Network action count
  actionCount: 9,  // 9 board positions

  // UI hints
  hasKeyboardControls: false,
  hasClickControls: true,
  hasInputVisualization: false,

  // Instructions text
  humanInstructions: 'Click a cell to place your mark (X goes first)',
  trainingInstructions: 'AI is learning to play as O against random moves',

  // Stats display configuration
  // TicTacToe tracks wins/losses/draws instead of score/highscore
  statsConfig: {
    layout: [
      { key: 'episode', label: 'Games' },
      { key: 'wins', label: 'Wins' },
      { key: 'losses', label: 'Losses' },
      { key: 'draws', label: 'Draws' },
    ],
  },
}
