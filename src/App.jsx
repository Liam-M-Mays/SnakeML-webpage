import React, { useState, useEffect, useRef } from 'react'
import { io } from "socket.io-client";
// Let it negotiate first; add transports:['websocket'] later if you want
const socket = io("http://127.0.0.1:5000", {
  // let it negotiate; you can force websocket later if you want
  transports: ["websocket", "polling"],
  withCredentials: false
});
import './App.css'
import { GameSettings, AIsettings, GameBoard } from './CustomComp';

function App() {
  // ========== GAME STATE ==========
  const [score, setScore] = useState(0)
  const [highscore, setHighscore] = useState(0)
  const [episode, setEpisode] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [gridSize, setGridSize] = useState(10)
  const [boardSize, setBoardSize] = useState(400)
  const [gameSpeed, setGameSpeed] = useState(250)

  const [colors, setColors] = useState({
  snake: '#4CAF50',    // Green
  food: '#ff5252',     // Red 
  background: '#222'   // Dark gray
  })

  // ========== AI CONTROL STATE ==========
  const [controlMode, setControlMode] = useState('human') // 'human', 'qnet', 'ppo'

  // ========== DEVICE STATE ==========
  const [deviceInfo, setDeviceInfo] = useState(null)

  const [params, setParams] = useState({
  qnet:
  {
    buffer: {min: 0, step: 1, placeholder: "Buffer Size", value: 10000 },
    batch: {min: 0, step: 1, placeholder: "Batch Size", value: 128 },
    gamma: {min: 0, step: 1, placeholder: "Gamma", value: 0.9 },
    decay: {min: 0, step: 0.00001, placeholder: "Decay", value: 0.999 },
  },
  ppo:
  {
    buffer: {min: 0, step: 1, placeholder: "Buffer Size", value: 1000 },
    batch: {min: 0, step: 1, placeholder: "Batch Size", value: 128 },
    gamma: {min: 0, step: .01, placeholder: "Gamma", value: 0.99 },
    decay: {min: 0, step: 1, placeholder: "Decay Steps", value: 1000 },
    epoch: {min: 1, step: 1, placeholder: "Epoch", value: 8}
  }
  })

  // ========== GAME LOGIC STATE ==========
  const [snake, setSnake] = useState([{x: null, y: null}]) // Start snake in middle
  const [food, setFood] = useState({x: null, y: null})
  const [direction, setDirection] = useState('right')
  const directionRef = useRef('right') // Use ref for instant updates

  // ========== GRID CONFIG ==========
  // Calculate cell size based on grid size to keep board consistent
  const cellSize = boardSize / gridSize

  // ========== GAME FUNCTIONS ==========

  const resetGame = () => {
    setIsRunning(false)
    setEpisode(0)
    setHighscore(0)
    directionRef.current = 'right' // Reset direction
    setSnake([{x: null, y: null}])
    setFood({x: null, y: null})
    if (gameSpeed === 0){
      socket.emit('stop_loop')
      setGameSpeed(250)
    }
  }

  useEffect(() => {
    // ========== SOCKET EVENT LISTENERS ==========
    socket.on('game_update', (data) => {
      // This runs when Python sends game state
      setSnake(data.snake_position)
      setFood(data.food_position)
      setScore(data.score)
      setEpisode(data.episode)

      // IMPORTANT: Only reset game on game_over for HUMAN mode
      // AI modes handle reset internally in the backend - they continue training
      // across episodes. Calling resetGame() during AI training would stop it!
      if (data.game_over && controlMode === 'human') {
        resetGame()
      }
    })

    socket.on('game_reset', (data) => {
      setScore(data.score)
      setSnake(data.snake_position)
      setFood(data.food_position)
      setIsRunning(false)
      directionRef.current = 'right' // Reset direction
    })

    socket.on('set_highscore', (data) => {
      setHighscore(data.highscore)
    })

    // Cleanup: remove listeners on unmount
    return () => {
      socket.off('game_update')
      socket.off('game_reset')
      socket.off('set_highscore')
    }
  }, [isRunning, controlMode]) // Re-run if isRunning or controlMode changes


  useEffect(() => {
    if (!isRunning) return
      const parms = Object.fromEntries(
        Object.entries(params[controlMode]).map(([k, d]) => [k, Number(d.value)])
      );
      socket.emit('init', {grid_size: gridSize, AI_mode: (controlMode !== 'human'), params: parms, modelType: controlMode})
  }, [gridSize, isRunning, controlMode, params]) // Re-run if grid size changes

  // ========== GAME LOOP ==========
  useEffect(() => {
    console.log("Game loop effect triggered. isRunning: " + isRunning + ", gameSpeed: " + gameSpeed + ", controlMode: " + controlMode)
    socket.emit('stop_loop')
    if (!isRunning) return

    if(controlMode == 'human'){
      const gameInterval = setInterval(async () => {
        socket.emit('step', { action: directionRef.current })
      }, gameSpeed)
      return () => clearInterval(gameInterval)
    }
    else if (gameSpeed > 0){
      const gameInterval = setInterval(async () => {
        socket.emit('AI_step')
      }, gameSpeed)
      return () => clearInterval(gameInterval)
    } // Game speed for AI
    else if (gameSpeed === 0) {
      console.log('infinity net called')
      socket.emit('AI_loop')
    }
  }, [isRunning, gameSpeed, controlMode]) // Dependencies: effect runs when these change



  // ========== KEYBOARD CONTROLS ==========
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Only process arrow keys if in human control mode
      if (controlMode === 'human') {
        // Prevent reversing direction (can't go right if moving left, etc.)
        if (e.key === 'ArrowRight' && direction !== 'left') {
          setDirection('right')
          directionRef.current = 'right'
        }
        if (e.key === 'ArrowLeft' && direction !== 'right') {
          setDirection('left')
          directionRef.current = 'left'
        }
        if (e.key === 'ArrowUp' && direction !== 'down') {
          setDirection('up')
          directionRef.current = 'up'
        }
        if (e.key === 'ArrowDown' && direction !== 'up') {
          setDirection('down')
          directionRef.current = 'down'
        }
      }
    }

    // Add event listener when component mounts
    window.addEventListener('keydown', handleKeyDown)
    
    // Cleanup: remove event listener when component unmounts
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [controlMode, direction]) // Only re-run if controlMode changes
  
  

  // ========== RENDER COMPONENT ==========
  return (
    <div className="app">
      {/* ========== HEADER ========== */}
      <header className="header">
        <h1>Snake Game</h1>
        <div className="score">Episode: {episode} - Score: {score} - Highscore: {highscore}</div>
      </header>

      {/* ========== MAIN CONTENT ========== */}
      <div className="content">
         
        <GameSettings gridSize={gridSize} setGridSize={setGridSize}
        controlMode={controlMode} setControlMode={setControlMode}
        colors={colors} setColors={setColors}
        boardSize={boardSize} setBoardSize={setBoardSize}
        gameSpeed={gameSpeed} setGameSpeed={setGameSpeed}
        isRunning={isRunning} resetGame={resetGame}
        setIsRunning={setIsRunning}
        deviceInfo={deviceInfo} setDeviceInfo={setDeviceInfo} />

        <GameBoard gridSize={gridSize} cellSize={cellSize} snake={snake} food={food}
        colors={colors} isRunning={isRunning} setIsRunning={setIsRunning} resetGame={resetGame}/>
        
        {controlMode !== 'human' && !isRunning && (<AIsettings params={params} setParams={setParams} controlMode={controlMode} />)}
      </div>

      {/* ========== INSTRUCTIONS ========== */}
      <div className="instructions">
          {controlMode === 'human' && (<p>Use arrow keys to control the snake. Avoid walls and yourself!</p>)}
      </div>
    </div>
  )
}

export default App
