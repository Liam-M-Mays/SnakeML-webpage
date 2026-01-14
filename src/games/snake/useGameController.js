import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

/**
 * Convert desired absolute direction to relative action.
 * Actions: 0 = forward, 1 = turn right, 2 = turn left
 */
function directionToAction(desiredDir, currentDir) {
  const [wantDx, wantDy] = desiredDir
  const [dx, dy] = currentDir

  // Can't reverse (opposite direction)
  if (wantDx === -dx && wantDy === -dy) return null

  // Forward (same direction)
  if (wantDx === dx && wantDy === dy) return 0

  // Right turn check: turning right from [dx,dy] gives [-dy, dx]
  if (wantDx === -dy && wantDy === dx) return 1

  // Left turn check: turning left from [dx,dy] gives [dy, -dx]
  if (wantDx === dy && wantDy === -dx) return 2

  return null
}

const KEY_TO_DIRECTION = {
  'ArrowRight': [1, 0],
  'ArrowLeft': [-1, 0],
  'ArrowUp': [0, -1],
  'ArrowDown': [0, 1]
}

/**
 * Snake game controller hook.
 * Encapsulates all snake-specific game logic, state, and socket handling.
 *
 * @param {object} socket - Socket.io instance
 * @param {object} gameSettings - Game settings from config
 * @param {object} options - { isTraining, isRunning, gameSpeed, selectedAgent }
 * @returns {object} Game controller interface
 */
export default function useSnakeController(socket, gameSettings, options) {
  const { isTraining, isRunning, gameSpeed, selectedAgent } = options

  // ========== SNAKE-SPECIFIC STATE ==========
  const [snake, setSnake] = useState([{ x: null, y: null }])
  const [food, setFood] = useState({ x: null, y: null })
  const [debugData, setDebugData] = useState({
    danger_cells: [],
    path_cells: [],
    segment_cells: [],
    segment_connections: []
  })

  // ========== GENERIC STATE ==========
  const [score, setScore] = useState(0)
  const [highscore, setHighscore] = useState(0)
  const [episode, setEpisode] = useState(0)

  // ========== DIRECTION TRACKING ==========
  const desiredDirectionRef = useRef(null)
  const currentDirectionRef = useRef([1, 0])  // Start going right

  // ========== SESSION TRACKING ==========
  const hasSessionRef = useRef(false)
  const sessionParamsRef = useRef(null)

  // ========== SOCKET LISTENERS ==========
  useEffect(() => {
    const handleGameUpdate = (data) => {
      setSnake(data.snake_position)
      setFood(data.food_position)
      setScore(data.score)
      setEpisode(data.episode)

      if (data.game_over) {
        desiredDirectionRef.current = null
        currentDirectionRef.current = [1, 0]
      }

      // Update debug visualization data if present
      if (data.debug) {
        setDebugData({
          danger_cells: data.debug.danger_cells || [],
          path_cells: data.debug.path_cells || [],
          segment_cells: data.debug.segment_cells || [],
          segment_connections: data.debug.segment_connections || []
        })
      }
    }

    const handleGameReset = (data) => {
      setScore(data.score)
      setSnake(data.snake_position)
      setFood(data.food_position)
      desiredDirectionRef.current = null
      currentDirectionRef.current = [1, 0]
    }

    const handleHighscore = (data) => {
      setHighscore(data.highscore)
    }

    const handleRandomStartStateChanged = (data) => {
      setHighscore(data.highscore)
    }

    socket.on('game_update', handleGameUpdate)
    socket.on('game_reset', handleGameReset)
    socket.on('set_highscore', handleHighscore)
    socket.on('random_start_state_changed', handleRandomStartStateChanged)

    return () => {
      socket.off('game_update', handleGameUpdate)
      socket.off('game_reset', handleGameReset)
      socket.off('set_highscore', handleHighscore)
      socket.off('random_start_state_changed', handleRandomStartStateChanged)
    }
  }, [socket])

  // ========== LIVE SETTINGS SYNC ==========
  useEffect(() => {
    if (!isRunning) return
    socket.emit('set_random_start_state', { enabled: gameSettings.randomStartState || false })
  }, [gameSettings.randomStartState, isRunning, socket])

  useEffect(() => {
    if (!isRunning || !gameSettings.randomStartState) return
    socket.emit('set_random_max_length', { max_length: gameSettings.randomMaxLength })
  }, [gameSettings.randomMaxLength, isRunning, gameSettings.randomStartState, socket])

  useEffect(() => {
    if (!isRunning) return
    const debug = gameSettings.debug || {}
    socket.emit('set_debug_settings', {
      vision: debug.vision || false,
      path: debug.path || false,
      segments: debug.segments || false
    })
  }, [gameSettings.debug?.vision, gameSettings.debug?.path, gameSettings.debug?.segments, isRunning, socket])

  // ========== GAME LOOP ==========
  useEffect(() => {
    socket.emit('stop_loop')
    if (!isRunning) return

    if (!isTraining) {
      // Human play mode - interval with direction tracking
      const gameInterval = setInterval(() => {
        let action = 0 // Default: forward
        const [dx, dy] = currentDirectionRef.current

        if (desiredDirectionRef.current) {
          const computed = directionToAction(desiredDirectionRef.current, currentDirectionRef.current)
          if (computed !== null) {
            action = computed

            // Update tracked direction based on turn
            if (action === 1) {
              // Turn right: [dx,dy] -> [-dy, dx]
              currentDirectionRef.current = [-dy, dx]
            } else if (action === 2) {
              // Turn left: [dx,dy] -> [dy, -dx]
              currentDirectionRef.current = [dy, -dx]
            }
          }
          // Clear desired direction after turn
          if (action === 1 || action === 2) {
            desiredDirectionRef.current = null
          }
        }

        socket.emit('step', { action })
      }, gameSpeed)

      return () => clearInterval(gameInterval)
    } else if (gameSpeed === 500) {
      // Min speed (Pause) - do nothing
    } else if (gameSpeed > 0) {
      // AI training with rendering
      const gameInterval = setInterval(() => {
        socket.emit('step', {})
      }, gameSpeed)
      return () => clearInterval(gameInterval)
    } else if (gameSpeed === 0) {
      // Max speed training (no render)
      setDebugData({ danger_cells: [], path_cells: [], segment_cells: [], segment_connections: [] })
      socket.emit('AI_loop')
    }
  }, [isRunning, gameSpeed, isTraining, socket])

  // ========== KEYBOARD HANDLER ==========
  const keyboardHandler = useCallback((e) => {
    if (isTraining) return
    if (!KEY_TO_DIRECTION[e.key]) return

    e.preventDefault()
    desiredDirectionRef.current = KEY_TO_DIRECTION[e.key]
  }, [isTraining])

  // ========== MEMOIZED STATS ==========
  const statsObj = useMemo(() => ({
    episode,
    score,
    highscore,
  }), [episode, score, highscore])

  // ========== RESET ==========
  const reset = useCallback(() => {
    desiredDirectionRef.current = null
    currentDirectionRef.current = [1, 0]
    hasSessionRef.current = false
    setSnake([{ x: null, y: null }])
    setFood({ x: null, y: null })
    setScore(0)
    setEpisode(0)
    setHighscore(0)
    setDebugData({ danger_cells: [], path_cells: [], segment_cells: [], segment_connections: [] })
    socket.emit('stop_loop')
  }, [socket])

  // ========== RETURN CONTROLLER INTERFACE ==========
  return {
    // Game-specific state (opaque to App.jsx)
    gameState: {
      snake,
      food,
      debugData
    },

    // Stats - game-specific (Snake uses score/highscore)
    stats: statsObj,

    // Legacy stats interface (for backward compatibility)
    score,
    episode,
    highscore,

    // Session tracking
    hasSessionRef,
    sessionParamsRef,

    // Actions
    reset,

    // Input handlers
    keyboardHandler,
    boardClickHandler: null  // Snake doesn't use click input
  }
}
