import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

/**
 * TicTacToe game controller hook.
 * Encapsulates all tictactoe-specific game logic, state, and socket handling.
 *
 * Unlike Snake, TicTacToe is turn-based so there's no game loop interval.
 * Actions are triggered by clicks or AI decisions.
 *
 * @param {object} socket - Socket.io instance
 * @param {object} gameSettings - Game settings from config
 * @param {object} options - { isTraining, isRunning, gameSpeed, selectedAgent }
 * @returns {object} Game controller interface
 */
export default function useTicTacToeController(socket, gameSettings, options) {
  const { isTraining, isRunning, gameSpeed, selectedAgent } = options

  // ========== TICTACTOE-SPECIFIC STATE ==========
  const [board, setBoard] = useState(Array(9).fill(null))
  const [currentPlayer, setCurrentPlayer] = useState('X')
  const [winner, setWinner] = useState(null)
  const [validActions, setValidActions] = useState([0, 1, 2, 3, 4, 5, 6, 7, 8])

  // ========== GENERIC STATE ==========
  const [episode, setEpisode] = useState(0)
  const [gameOver, setGameOver] = useState(false)

  // ========== TICTACTOE STATS ==========
  const [wins, setWins] = useState(0)
  const [losses, setLosses] = useState(0)
  const [draws, setDraws] = useState(0)

  // ========== SESSION TRACKING ==========
  const hasSessionRef = useRef(false)
  const sessionParamsRef = useRef(null)
  const prevWinnerRef = useRef(null)  // Track previous winner to detect new game results

  // ========== SOCKET LISTENERS ==========
  useEffect(() => {
    const handleGameUpdate = (data) => {
      // TicTacToe state comes from backend
      if (data.board) setBoard(data.board)
      if (data.current_player) setCurrentPlayer(data.current_player)
      if (data.valid_actions) setValidActions(data.valid_actions)
      if (data.episode !== undefined) setEpisode(data.episode)
      if (data.game_over !== undefined) setGameOver(data.game_over)

      // Track wins/losses/draws when game ends
      if (data.winner !== undefined) {
        setWinner(data.winner)

        // Only count if this is a new game result (winner changed from null to something)
        if (prevWinnerRef.current === null && data.winner !== null) {
          const aiPlayer = gameSettings.aiPlayer || 'O'
          if (data.winner === aiPlayer) {
            setWins(w => w + 1)
          } else if (data.winner === 'draw') {
            setDraws(d => d + 1)
          } else {
            setLosses(l => l + 1)
          }
        }
        prevWinnerRef.current = data.winner
      }
    }

    const handleGameReset = (data) => {
      if (data.board) setBoard(data.board)
      if (data.current_player) setCurrentPlayer(data.current_player)
      setWinner(null)
      setValidActions(data.valid_actions || [0, 1, 2, 3, 4, 5, 6, 7, 8])
      setGameOver(false)
      prevWinnerRef.current = null  // Reset winner tracking for new game
    }

    socket.on('game_update', handleGameUpdate)
    socket.on('game_reset', handleGameReset)

    return () => {
      socket.off('game_update', handleGameUpdate)
      socket.off('game_reset', handleGameReset)
    }
  }, [socket, gameSettings.aiPlayer])

  // ========== AI TRAINING LOOP ==========
  // For TicTacToe AI training, we continuously request steps
  useEffect(() => {
    if (!isRunning || !isTraining) return

    // If game is over, request reset and continue
    if (gameOver) {
      socket.emit('reset_game')
      return
    }

    // Determine if it's AI's turn
    const aiPlayer = gameSettings.aiPlayer || 'O'
    const isAITurn = currentPlayer === aiPlayer

    if (isAITurn) {
      // AI makes a move
      if (gameSpeed === 0) {
        // Max speed - continuous loop
        socket.emit('step', {})
      } else if (gameSpeed < 500) {
        // Delayed steps
        const timeout = setTimeout(() => {
          socket.emit('step', {})
        }, gameSpeed)
        return () => clearTimeout(timeout)
      }
      // gameSpeed === 500 means pause, do nothing
    }
  }, [isRunning, isTraining, gameSpeed, currentPlayer, gameOver, gameSettings.aiPlayer, socket])

  // ========== BOARD CLICK HANDLER ==========
  const boardClickHandler = useCallback((cellIndex) => {
    if (!isRunning) return
    if (gameOver) return
    if (winner) return

    // Check if cell is valid
    if (!validActions.includes(cellIndex)) return

    // In training mode, human plays as X (AI is O)
    // In human mode, human can play any move
    const aiPlayer = gameSettings.aiPlayer || 'O'
    if (isTraining && currentPlayer === aiPlayer) {
      // It's AI's turn, human can't move
      return
    }

    // Send the move to backend
    socket.emit('step', { action: cellIndex })
  }, [isRunning, gameOver, winner, validActions, isTraining, currentPlayer, gameSettings.aiPlayer, socket])

  // ========== MEMOIZED STATS ==========
  const statsObj = useMemo(() => ({
    episode,
    wins,
    losses,
    draws,
  }), [episode, wins, losses, draws])

  // ========== RESET ==========
  const reset = useCallback(() => {
    hasSessionRef.current = false
    setBoard(Array(9).fill(null))
    setCurrentPlayer('X')
    setWinner(null)
    setValidActions([0, 1, 2, 3, 4, 5, 6, 7, 8])
    setEpisode(0)
    setWins(0)
    setLosses(0)
    setDraws(0)
    setGameOver(false)
    prevWinnerRef.current = null
    socket.emit('stop_loop')
  }, [socket])

  // ========== RETURN CONTROLLER INTERFACE ==========
  return {
    // Game-specific state (opaque to App.jsx)
    gameState: {
      board,
      currentPlayer,
      winner,
      validActions,
      gameOver
    },

    // Stats - game-specific (TicTacToe uses wins/losses/draws, not score/highscore)
    stats: statsObj,

    // Legacy stats interface (for backward compatibility with App.jsx)
    // These will be deprecated once App.jsx uses statsConfig
    score: 0,
    episode,
    highscore: 0,

    // Session tracking
    hasSessionRef,
    sessionParamsRef,

    // Actions
    reset,

    // Input handlers
    keyboardHandler: null,  // TicTacToe doesn't use keyboard
    boardClickHandler       // TicTacToe uses click input
  }
}
