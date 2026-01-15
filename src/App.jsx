import React, { useState, useEffect, useRef } from 'react'
import { io } from "socket.io-client"
import './App.css'

// Snake game components
import {
  config as snakeConfig,
  Board as SnakeBoard,
  Settings as SnakeSettings,
  useGameController,
  InputVisualization,
  applyTheme
} from './games'

// Components
import Sidebar from './components/Sidebar'
import AISettings from './components/AISettings'
import SpeedControl from './components/SpeedControl'
import ErrorBoundary from './components/ErrorBoundary'
import TrainingChart from './components/TrainingChart'

const socket = io("http://127.0.0.1:5000", {
  transports: ["websocket", "polling"],
  withCredentials: false
})

/**
 * GameArea component - handles Snake game logic.
 */
function GameArea({
  gameSettings,
  socket,
  isTraining,
  isRunning,
  gameSpeed,
  selectedAgent,
  cellSize,
  onStatsChange,
  onReset,
}) {
  // Call the game controller hook
  const gameController = useGameController?.(socket, gameSettings, {
    isTraining,
    isRunning,
    gameSpeed,
    selectedAgent
  })

  // Sync controller stats up to parent
  useEffect(() => {
    if (gameController?.stats) {
      onStatsChange(gameController.stats)
    }
  }, [gameController?.stats, onStatsChange])

  // Expose reset function to parent
  useEffect(() => {
    if (gameController?.reset) {
      onReset(gameController.reset)
    }
  }, [gameController?.reset])

  // Register keyboard handler
  useEffect(() => {
    const handler = gameController?.keyboardHandler
    if (!handler) return
    if (!isRunning) return

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [gameController?.keyboardHandler, isRunning])

  // Session tracking for init
  const initRef = useRef(0)
  const wasRunningRef = useRef(false)

  useEffect(() => {
    const wasRunning = wasRunningRef.current
    wasRunningRef.current = isRunning

    if (!isRunning) return

    // Build current session params to detect if we need new session
    const currentSessionParams = JSON.stringify({
      isTraining,
      agentId: selectedAgent?.id,
      agentType: selectedAgent?.type,
      gridSize: gameSettings.gridSize,
    })

    // Skip init if resuming from pause with same params
    const hasSession = gameController?.hasSessionRef?.current
    const prevParams = gameController?.sessionParamsRef?.current
    if (wasRunning === false && hasSession && prevParams === currentSessionParams) {
      console.log('[INIT] Resuming existing session')
      return
    }

    // Mark new session
    if (gameController?.hasSessionRef) {
      gameController.hasSessionRef.current = true
    }
    if (gameController?.sessionParamsRef) {
      gameController.sessionParamsRef.current = currentSessionParams
    }

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
      random_start_state: gameSettings.randomStartState || false,
      random_max_length: gameSettings.randomMaxLength || null,
      inputs: gameSettings.inputs || {},
      rewards: gameSettings.rewards || {},
      device: gameSettings.device || 'cpu',
    })
  }, [isRunning, isTraining, selectedAgent, gameSettings.gridSize])

  // Load saved model
  useEffect(() => {
    if (!isTraining || !selectedAgent?.modelFilename) return

    const timeout = setTimeout(() => {
      console.log('Loading saved model:', selectedAgent.modelFilename)
      socket.emit('load_model', { filename: selectedAgent.modelFilename })
    }, 500)

    return () => clearTimeout(timeout)
  }, [isTraining, selectedAgent?.modelFilename])

  return (
    <>
      {/* Game Board */}
      <ErrorBoundary fallbackMessage="Failed to render the game board. Try refreshing the page.">
        <SnakeBoard
          settings={gameSettings}
          gameState={gameController?.gameState}
          cellSize={cellSize}
        />
      </ErrorBoundary>

      {/* Input Visualization */}
      <InputVisualization
        socket={socket}
        isTraining={isTraining}
        gameSettings={gameSettings}
      />
    </>
  )
}

function App() {
  // ========== UI STATE ==========
  const [leftCollapsed, setLeftCollapsed] = useState(false)
  const [rightCollapsed, setRightCollapsed] = useState(false)

  // ========== GAME STATE ==========
  const [isRunning, setIsRunning] = useState(false)
  const [gameSpeed, setGameSpeed] = useState(250)

  // Game settings
  const [gameSettings, setGameSettings] = useState(snakeConfig.defaultSettings || {})

  // ========== AI AGENT STATE ==========
  const [agents, setAgents] = useState([])
  const [selectedAgentId, setSelectedAgentId] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [savedModels, setSavedModels] = useState([])
  const [selectedDevice, setSelectedDevice] = useState('cpu')

  // ========== VALUES FROM GAME CONTROLLER ==========
  const [stats, setStats] = useState({})
  const resetFnRef = useRef(null)

  // ========== DERIVED VALUES ==========
  const selectedAgent = agents.find(a => a.id === selectedAgentId)
  const cellSize = (gameSettings.boardSize || 400) / (gameSettings.gridSize || 10)

  // ========== THEME APPLICATION ==========
  useEffect(() => {
    applyTheme()
  }, [])

  // ========== GAME FUNCTIONS ==========
  const resetGame = () => {
    setIsRunning(false)
    setIsTraining(false)
    setStats({})
    if (gameSpeed === 0) {
      socket.emit('stop_loop')
      setGameSpeed(250)
    }
    resetFnRef.current?.()
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

    // Model management listeners
    socket.on('models_list', (data) => {
      setSavedModels(data.models || [])
    })

    socket.on('save_model_result', (data) => {
      if (data.success) {
        console.log('Model saved:', data.filename)
        socket.emit('list_models')
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
        socket.emit('list_models')
      }
    })

    return () => {
      socket.off('models_list')
      socket.off('save_model_result')
      socket.off('load_model_result')
      socket.off('delete_model_result')
    }
  }, [])

  // ========== RENDER ==========
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>SnakeML</h1>
        <div className="stats">
          {snakeConfig.statsConfig?.layout.map(({ key, label, disabledWhen }) => {
            const isDisabled = disabledWhen && gameSettings[disabledWhen]
            const value = isDisabled ? '--' : (stats[key] ?? 0)
            return (
              <span key={key} className={isDisabled ? 'stat-disabled' : ''}>
                {label}: {value}
              </span>
            )
          })}
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Left Sidebar - Settings */}
        <Sidebar
          title="Settings"
          side="left"
          collapsed={leftCollapsed}
          onToggle={() => setLeftCollapsed(!leftCollapsed)}
        >
          <ErrorBoundary fallbackMessage="Failed to load game settings.">
            <SnakeSettings
              settings={gameSettings}
              onChange={setGameSettings}
              disabled={isRunning}
            />
          </ErrorBoundary>
        </Sidebar>

        {/* Center - Game Area */}
        <div className="game-area">
          <GameArea
            gameSettings={gameSettings}
            socket={socket}
            isTraining={isTraining}
            isRunning={isRunning}
            gameSpeed={gameSpeed}
            selectedAgent={selectedAgent}
            cellSize={cellSize}
            onStatsChange={setStats}
            onReset={(fn) => { resetFnRef.current = fn }}
          />

          {/* Training Metrics Charts */}
          <TrainingChart
            socket={socket}
            isTraining={isTraining}
            agentType={selectedAgent?.type}
            gameSpeed={gameSpeed}
          />

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
            {!isTraining && snakeConfig.humanInstructions && (
              <p>{snakeConfig.humanInstructions}</p>
            )}
            {isTraining && snakeConfig.trainingInstructions && (
              <p>{snakeConfig.trainingInstructions}</p>
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
              selectedGame="snake"
              socket={socket}
              selectedDevice={selectedDevice}
              onDeviceChange={setSelectedDevice}
            />
          </ErrorBoundary>
        </Sidebar>
      </div>
    </div>
  )
}

export default App
