import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react'
import { useSocket } from './SocketContext'
import { GAMES, applyTheme } from '../games'

const GameContext = createContext(null)

/**
 * Game state and actions provider.
 * Manages game selection, settings, running state, and score tracking.
 */
export function GameProvider({ children }) {
  const { socket } = useSocket()

  // Game selection
  const [selectedGame, setSelectedGame] = useState('snake')
  const gameConfig = GAMES[selectedGame]

  // Game settings (passed to game's Settings component)
  const [gameSettings, setGameSettings] = useState(gameConfig?.defaultSettings || {})

  // Game state
  const [score, setScore] = useState(0)
  const [highscore, setHighscore] = useState(0)
  const [episode, setEpisode] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [gameSpeed, setGameSpeed] = useState(250)

  // Snake-specific state (could be generalized later)
  const [snake, setSnake] = useState([{ x: null, y: null }])
  const [food, setFood] = useState({ x: null, y: null })

  // Direction tracking
  const desiredDirectionRef = useRef(null)
  const currentDirectionRef = useRef([1, 0])

  // Derived values
  const cellSize = (gameSettings.boardSize || 400) / (gameSettings.gridSize || 10)

  // Apply theme when game changes
  useEffect(() => {
    applyTheme(selectedGame)
  }, [selectedGame])

  // Reset settings when game changes
  useEffect(() => {
    const config = GAMES[selectedGame]
    if (config?.defaultSettings) {
      setGameSettings(config.defaultSettings)
    }
    resetGame()
  }, [selectedGame])

  // Socket listeners for game updates
  useEffect(() => {
    if (!socket) return

    socket.on('game_update', (data) => {
      setSnake(data.snake_position)
      setFood(data.food_position)
      setScore(data.score)
      setEpisode(data.episode)
      if (data.game_over) {
        desiredDirectionRef.current = null
        currentDirectionRef.current = [1, 0]
      }
    })

    socket.on('game_reset', (data) => {
      setScore(data.score)
      setSnake(data.snake_position)
      setFood(data.food_position)
      setIsRunning(false)
      desiredDirectionRef.current = null
      currentDirectionRef.current = [1, 0]
    })

    socket.on('set_highscore', (data) => {
      setHighscore(data.highscore)
    })

    return () => {
      socket.off('game_update')
      socket.off('game_reset')
      socket.off('set_highscore')
    }
  }, [socket])

  // Reset game function
  const resetGame = useCallback(() => {
    setIsRunning(false)
    setEpisode(0)
    setHighscore(0)
    desiredDirectionRef.current = null
    currentDirectionRef.current = [1, 0]
    setSnake([{ x: null, y: null }])
    setFood({ x: null, y: null })
    if (gameSpeed === 0 && socket) {
      socket.emit('stop_loop')
      setGameSpeed(250)
    }
  }, [gameSpeed, socket])

  const value = {
    // Game selection
    selectedGame,
    setSelectedGame,
    gameConfig,
    GameBoard: gameConfig?.Board,
    GameSettingsPanel: gameConfig?.Settings,

    // Game settings
    gameSettings,
    setGameSettings,
    cellSize,

    // Game state
    score,
    highscore,
    episode,
    isRunning,
    setIsRunning,
    gameSpeed,
    setGameSpeed,

    // Snake state
    snake,
    food,

    // Direction refs
    desiredDirectionRef,
    currentDirectionRef,

    // Actions
    resetGame,
  }

  return (
    <GameContext.Provider value={value}>
      {children}
    </GameContext.Provider>
  )
}

/**
 * Hook to access game state and actions.
 */
export function useGame() {
  const context = useContext(GameContext)
  if (!context) {
    throw new Error('useGame must be used within a GameProvider')
  }
  return context
}

export default GameContext
