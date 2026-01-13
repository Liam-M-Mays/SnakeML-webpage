import React, { useState, useEffect, useRef } from 'react'
import { io } from "socket.io-client"
import './App.css'

// Game registry and theming
import { GAMES, getGameList, applyTheme } from './games'

// Components
import Sidebar from './components/Sidebar'
import AISettings from './components/AISettings'
import SpeedControl from './components/SpeedControl'
import ErrorBoundary from './components/ErrorBoundary'

const socket = io("http://127.0.0.1:5000", {
  transports: ["websocket", "polling"],
  withCredentials: false
})

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

function App() {
  // ========== UI STATE ==========
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightCollapsed, setRightCollapsed] = useState(false)

  // ========== GAME SELECTION ==========
  const [selectedGame, setSelectedGame] = useState('snake')
  const gameConfig = GAMES[selectedGame]
  const GameBoard = gameConfig?.Board
  const GameSettingsPanel = gameConfig?.Settings

  // ========== GAME STATE ==========
  const [score, setScore] = useState(0)
  const [highscore, setHighscore] = useState(0)
  const [episode, setEpisode] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [gameSpeed, setGameSpeed] = useState(250)

  // Game-specific settings (passed to game's Settings component)
  const [gameSettings, setGameSettings] = useState(gameConfig?.defaultSettings || {})

  // ========== AI AGENT STATE ==========
  const [agents, setAgents] = useState([])
  const [selectedAgentId, setSelectedAgentId] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [savedModels, setSavedModels] = useState([])

  // ========== SNAKE-SPECIFIC STATE ==========
  const [snake, setSnake] = useState([{ x: null, y: null }])
  const [food, setFood] = useState({ x: null, y: null })

  // Store the desired direction from keypress (absolute direction)
  const desiredDirectionRef = useRef(null)
  // Track actual snake direction (updated when we send actions)
  const currentDirectionRef = useRef([1, 0])

  // ========== DERIVED VALUES ==========
  const selectedAgent = agents.find(a => a.id === selectedAgentId)
  const isHumanMode = !isTraining
  const cellSize = (gameSettings.boardSize || 400) / (gameSettings.gridSize || 10)

  // ========== THEME APPLICATION ==========
  useEffect(() => {
    applyTheme(selectedGame)
  }, [selectedGame])

  // ========== GAME CHANGE HANDLER ==========
  useEffect(() => {
    const config = GAMES[selectedGame]
    if (config?.defaultSettings) {
      setGameSettings(config.defaultSettings)
    }
    resetGame()
  }, [selectedGame])

  // ========== GAME FUNCTIONS ==========
  const resetGame = () => {
    setIsRunning(false)
    setIsTraining(false)
    setEpisode(0)
    setHighscore(0)
    desiredDirectionRef.current = null
    currentDirectionRef.current = [1, 0]  // Snake starts going right
    setSnake([{ x: null, y: null }])
    setFood({ x: null, y: null })
    if (gameSpeed === 0) {
      socket.emit('stop_loop')
      setGameSpeed(250)
    }
  }

  const handlePlayHuman = () => {
    setIsTraining(false)
    setIsRunning(true)
  }

  const handleTrain = () => {
    if (!selectedAgent) return
    setIsTraining(true)
    setIsRunning(true)
  }

  const handleStopTraining = () => {
    setIsTraining(false)
    setIsRunning(false)
    socket.emit('stop_loop')
  }

  // ========== AGENT MANAGEMENT ==========
  const handleCreateAgent = (agent) => {
    setAgents(prev => [...prev, agent])
    setSelectedAgentId(agent.id)
  }

  const handleUpdateParams = (agentId, paramKey, value) => {
    setAgents(prev => prev.map(agent => {
      if (agent.id !== agentId) return agent
      return {
        ...agent,
        params: {
          ...agent.params,
          [paramKey]: { ...agent.params[paramKey], value }
        }
      }
    }))
  }

  // ========== MODEL SAVE/LOAD ==========
  const handleSaveModel = (name, agentId, agentName, game) => {
    socket.emit('save_model', { name, agent_id: agentId, agent_name: agentName, game })
  }

  const handleDeleteModel = (filename) => {
    socket.emit('delete_model', { filename })
  }

  const handleRefreshModels = () => {
    socket.emit('list_models')
  }

  // ========== SOCKET LISTENERS ==========
  useEffect(() => {
    // Request saved models on connect
    socket.emit('list_models')

    socket.on('game_update', (data) => {
      setSnake(data.snake_position)
      setFood(data.food_position)
      setScore(data.score)
      setEpisode(data.episode)
      if (data.game_over) {
        desiredDirectionRef.current = null
        currentDirectionRef.current = [1, 0]  // Reset for next game
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

    // Model save/load listeners
    socket.on('models_list', (data) => {
      setSavedModels(data.models || [])
    })

    socket.on('save_model_result', (data) => {
      if (data.success) {
        console.log('Model saved:', data.filename)
        socket.emit('list_models')  // Refresh list
      } else {
        console.error('Save failed:', data.error)
      }
    })

    socket.on('load_model_result', (data) => {
      if (data.success) {
        console.log('Model loaded:', data.filename)
      } else {
        console.error('Load failed:', data.error)
      }
    })

    socket.on('delete_model_result', (data) => {
      if (data.success) {
        socket.emit('list_models')  // Refresh list
      }
    })

    return () => {
      socket.off('game_update')
      socket.off('game_reset')
      socket.off('set_highscore')
      socket.off('models_list')
      socket.off('save_model_result')
      socket.off('load_model_result')
      socket.off('delete_model_result')
    }
  }, [])

  // ========== INIT ON START ==========
  const initRef = useRef(0)
  useEffect(() => {
    if (!isRunning) return

    initRef.current += 1
    const initCount = initRef.current
    console.log(`[INIT #${initCount}] Creating session - isTraining=${isTraining}, agent=${selectedAgent?.name}`)

    const controlMode = isTraining ? selectedAgent?.type : 'human'
    const agentParams = selectedAgent?.params || {}
    const params = Object.fromEntries(
      Object.entries(agentParams).map(([k, d]) => [k, Number(d.value)])
    )

    socket.emit('init', {
      grid_size: gameSettings.gridSize || 10,
      control_mode: controlMode,
      params: params,
      game_type: selectedGame
    })
  }, [isRunning, isTraining, selectedAgent, gameSettings.gridSize, selectedGame])

  // ========== LOAD SAVED MODEL ==========
  useEffect(() => {
    if (!isTraining || !selectedAgent?.modelFilename) return

    // Delay to ensure session is initialized
    const timeout = setTimeout(() => {
      console.log('Loading saved model:', selectedAgent.modelFilename)
      socket.emit('load_model', { filename: selectedAgent.modelFilename })
    }, 500)  // Increased delay to ensure session is ready

    return () => clearTimeout(timeout)
  }, [isTraining, selectedAgent?.modelFilename])

  // ========== GAME LOOP ==========
  useEffect(() => {
    socket.emit('stop_loop')
    if (!isRunning) return

    if (!isTraining) {
      // Human play mode
      const gameInterval = setInterval(() => {
        let action = 0 // Default: forward
        const [dx, dy] = currentDirectionRef.current

        if (desiredDirectionRef.current) {
          const computed = directionToAction(desiredDirectionRef.current, currentDirectionRef.current)
          if (computed !== null) {
            action = computed

            // Update our tracked direction based on the turn
            if (action === 1) {
              // Turn right: [dx,dy] -> [-dy, dx]
              currentDirectionRef.current = [-dy, dx]
            } else if (action === 2) {
              // Turn left: [dx,dy] -> [dy, -dx]
              currentDirectionRef.current = [dy, -dx]
            }
          }
          // Clear desired direction after turn (user needs to press again)
          // Keep it for forward so holding a direction key keeps going that way
          if (action === 1 || action === 2) {
            desiredDirectionRef.current = null
          }
        }

        socket.emit('step', { action })
      }, gameSpeed)
      return () => clearInterval(gameInterval)
    } else if (gameSpeed > 0) {
      // AI training with rendering
      const gameInterval = setInterval(() => {
        socket.emit('step', {})
      }, gameSpeed)
      return () => clearInterval(gameInterval)
    } else if (gameSpeed === 0) {
      // Max speed training (no render)
      socket.emit('AI_loop')
    }
  }, [isRunning, gameSpeed, isTraining])

  // ========== KEYBOARD CONTROLS ==========
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (isTraining) return
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return

      e.preventDefault()

      // Store the desired absolute direction
      // The game loop will compute the relative action
      desiredDirectionRef.current = KEY_TO_DIRECTION[e.key]
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isTraining])

  // ========== RENDER ==========
  const gameList = getGameList()

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>ML Playground</h1>
        <div className="stats">
          <span>Episode: {episode}</span>
          <span>Score: {score}</span>
          <span>Highscore: {highscore}</span>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Left Sidebar - Games & Settings */}
        <Sidebar
          title="Games"
          side="left"
          collapsed={leftCollapsed}
          onToggle={() => setLeftCollapsed(!leftCollapsed)}
        >
          {/* Game Selection */}
          <div className="game-list">
            {gameList.map(game => (
              <div
                key={game.id}
                className={`game-item ${game.id === selectedGame ? 'selected' : ''} ${game.comingSoon ? 'coming-soon' : ''}`}
                onClick={() => !game.comingSoon && !isRunning && setSelectedGame(game.id)}
              >
                <span className="game-name">{game.name}</span>
                {game.comingSoon && <span className="badge">Soon</span>}
              </div>
            ))}
          </div>

          {/* Game-specific Settings */}
          {GameSettingsPanel && (
            <div className="game-settings-section">
              <h4>Game Settings</h4>
              <ErrorBoundary fallbackMessage="Failed to load game settings.">
                <GameSettingsPanel
                  settings={gameSettings}
                  onChange={setGameSettings}
                  disabled={isRunning}
                />
              </ErrorBoundary>
            </div>
          )}
        </Sidebar>

        {/* Center - Game Area */}
        <div className="game-area">
          {/* Game Board */}
          <ErrorBoundary fallbackMessage="Failed to render the game board. Try refreshing the page.">
            {GameBoard && (
              <GameBoard
                settings={gameSettings}
                snake={snake}
                food={food}
                cellSize={cellSize}
              />
            )}
          </ErrorBoundary>

          {/* Speed Control - only during training */}
          <SpeedControl
            speed={gameSpeed}
            onChange={setGameSpeed}
            isVisible={isTraining}
          />

          {/* Game Controls */}
          <div className="game-controls">
            {!isRunning ? (
              <>
                <button className="play-btn" onClick={handlePlayHuman}>
                  Play Human
                </button>
                {selectedAgent && (
                  <button className="train-btn" onClick={handleTrain}>
                    Train AI
                  </button>
                )}
              </>
            ) : (
              <>
                <button className="pause-btn" onClick={() => setIsRunning(false)}>
                  Pause
                </button>
                <button className="reset-btn" onClick={resetGame}>
                  Reset
                </button>
              </>
            )}
          </div>

          {/* Instructions */}
          <div className="instructions">
            {!isTraining && selectedGame === 'snake' && (
              <p>Use arrow keys to control the snake</p>
            )}
            {isTraining && (
              <p>Set speed to 0 for maximum training speed</p>
            )}
          </div>
        </div>

        {/* Right Sidebar - AI Settings */}
        <Sidebar
          title="AI"
          side="right"
          collapsed={rightCollapsed}
          onToggle={() => setRightCollapsed(!rightCollapsed)}
        >
          <ErrorBoundary fallbackMessage="Failed to load AI settings. Try refreshing the page.">
            <AISettings
              agents={agents}
              selectedAgentId={selectedAgentId}
              onCreateAgent={handleCreateAgent}
              onSelectAgent={setSelectedAgentId}
              onUpdateParams={handleUpdateParams}
              onTrain={handleTrain}
              onStopTraining={handleStopTraining}
              isTraining={isTraining}
              disabled={isRunning && !isTraining}
              savedModels={savedModels}
              onSaveModel={handleSaveModel}
              onDeleteModel={handleDeleteModel}
              onRefreshModels={handleRefreshModels}
              selectedGame={selectedGame}
            />
          </ErrorBoundary>
        </Sidebar>
      </div>
    </div>
  )
}

export default App
