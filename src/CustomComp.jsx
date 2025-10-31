import React, { useState } from 'react'
import './App.css'

export function GameSettings({
    gridSize, setGridSize, 
    controlMode, setControlMode, 
    colors, setColors, 
    boardSize, setBoardSize, 
    gameSpeed, setGameSpeed,
    isRunning, resetGame}){
    const [activeTab, setActiveTab] = useState('board')


    return (
        <div className="control-panel">
          <h3 style={{textAlign: 'center'}}>Game Settings</h3>
          <div className='tab-container' >
            <div className="tab-buttons">
              <button 
                  className={`tab-button ${activeTab === 'board' ? 'active' : ''}`}
                  onClick={() => setActiveTab('board')}
                >
                  Board
                </button>
                <button 
                  className={`tab-button ${activeTab === 'color' ? 'active' : ''}`}
                  onClick={() => setActiveTab('color')}
                >
                   Colors
                </button>
                <button 
                  className={`tab-button ${activeTab === 'speed' ? 'active' : ''}`}
                  onClick={() => setActiveTab('speed')}
                >
                  Speed
                </button>
            </div>
          </div>

          <div className="tab-content">
            {activeTab === 'board' && (
              <div className="control-group">
                {/* ========== GRID SIZE CONTROL ========== */}
                <div>
                  <label>Grid Size: {gridSize}x{gridSize}</label>
                  <input
                    type="range"
                    min="5"
                    max="25"
                    disabled={isRunning}
                    value={gridSize}
                    onChange={(e) => {
                      const newSize = parseInt(e.target.value)
                      setGridSize(newSize)
                      resetGame // Reset game when grid size changes
                    }}
                  />
                </div>
                <div>
                  <label>Board Size: {boardSize}</label>
                  <input 
                    type="range"
                    min="300"
                    max="750"
                    value={boardSize}
                    onChange={(w) => {
                      const newSize = parseInt(w.target.value)
                      setBoardSize(newSize)
                    }}
                  />
                </div>
              </div>
            )}
            {activeTab === 'color' && (
              <div className="control-group">
                {/* ========== COLOR SETTINGS ========== */}
                <div>
                  <label>Snake Color:</label>
                  <div className="color-input-wrapper">
                    <input 
                      type="color"
                      value={colors.snake}
                      onChange={(C) => setColors(prev => ({...prev, snake: C.target.value}))}
                      className="color-picker"
                    />
                  </div>
                  <div>
                    <label>Food Color:</label>
                    <div className="color-input-wrapper">
                      <input 
                        type="color"
                        value={colors.food}
                        onChange={(c) => setColors(prev => ({...prev, food: c.target.value}))}
                        className="color-picker"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
            {activeTab === 'speed' && (
              <div>
                {/* ====== SPEED SETTINGS ===== */}
                { controlMode !== 'human' ? (<>
                  <label>AI Game Speed: {gameSpeed}</label>
                  <input
                    type="range"
                    min="0"
                    max="500"
                    value={gameSpeed}
                    onChange={(s) => {
                      const newSize = parseInt(s.target.value)
                      setGameSpeed(newSize)
                    }}
                  />
                </>) : (
                  <p>Speeds are only in AI modes.</p>
                )}
                
              </div>
            )}
          </div>

          {/* ========== CONTROL MODE SELECTION ========== */}
          <div className="control-group">
            <label>Control Mode:</label>
            <select 
              value={controlMode}
              disabled={isRunning}
              onChange={(m) => setControlMode(m.target.value)}
            >
              <option value="human">Human Player</option>
              <option value="qnet">Q-Network AI</option>
              <option value="ppo">PPO AI</option>
            </select>
          </div>
        </div>
        )
    
    }

export function AIsettings({ params, setParams, controlMode }){
  const [activeTab, setActiveTab] = useState('param')

    return (
    <div className="ai-settings">
        <h3 style={{textAlign: 'center'}} >AI Settings</h3>
          <div className='ai-tab-container' >
            <div className="ai-tab-buttons">
              <button 
                  className={`ai-tab-button ${activeTab === 'param' ? 'active' : ''}`}
                  onClick={() => setActiveTab('param')}
                >
                  Params
                </button>
                <button 
                  className={`ai-tab-button ${activeTab === 'state' ? 'active' : ''}`}
                  onClick={() => setActiveTab('state')}
                >
                   State
                </button>
                <button 
                  className={`ai-tab-button ${activeTab === 'network' ? 'active' : ''}`}
                  onClick={() => setActiveTab('network')}
                >
                  Net
                </button>
            </div>
          </div>
          {activeTab === 'param' && (
                Object.entries(params[controlMode]).map(([key, details]) => (

                 <label key={key} style={{ display: 'block', marginBottom: 8 }}>
                {details.placeholder}--
                <input
                    type="number"
                    min={details.min}
                    step={details.step}
                    value={details.value ?? ''}
                    onChange={(e) =>
                    setParams(prev => ({
                      ...prev,
                      [controlMode]:
                        {...prev[controlMode],
                        [key]: {...prev[controlMode][key], value : Number(e.target.value) }}
                    }))
                    }
                />
                </label>
            )))}

            
  
    </div>
    )
}

export function GameBoard({ gridSize, cellSize, colors, isRunning, setIsRunning, resetGame, snake, food }){

  const renderGrid = () => {
    const cells = []
    // Create gridSize x gridSize grid
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        // Check if current cell contains snake
        const isSnake = snake.some(segment => segment.x === x && segment.y === y)
        // Check if current cell contains food
        const isFood = food.x === x && food.y === y
        
        // Determine CSS class based on cell content
        let cellClass = 'cell'
        if (isSnake) cellClass += ' snake'
        if (isFood) cellClass += ' food'
        
        // Create cell element with unique key
        cells.push(
          <div 
            key={`${x}-${y}`} 
            className={cellClass}
          />
        )
      }
    }
    return cells
  }



  return(
    <div className="game-section">
          <div 
            className="game-board"
            style={{
              // Dynamic grid styling based on gridSize
              gridTemplateColumns: `repeat(${gridSize}, ${cellSize}px)`,
              gridTemplateRows: `repeat(${gridSize}, ${cellSize}px)`,
              margin: `${cellSize}px, auto`,
              // Add these lines for dynamic colors:
              '--snake-color': colors.snake,
              '--food-color': colors.food,
              '--grid-bg': colors.background
            }}
          >
            {renderGrid()}
          </div>
          
          {/* ========== GAME CONTROLS ========== */}
          <div className="game-controls">
            <button onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? 'Pause' : 'Start'}
            </button>
            <button onClick={resetGame}>Reset</button>
          </div>
        </div>
  )
}