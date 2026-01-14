import React from 'react'

/**
 * TicTacToe board component.
 * Renders the 3x3 grid with X's and O's.
 * Handles click events for making moves.
 */
export default function TicTacToeBoard({ settings, gameState, cellSize, onCellClick }) {
  const { colors = {} } = settings || {}
  const { board = Array(9).fill(null), currentPlayer, winner, validActions = [], gameOver } = gameState || {}

  const boardSize = settings?.boardSize || 400
  const cellDimension = boardSize / 3

  // Determine cell styling
  const getCellStyle = (index) => {
    const isValid = validActions.includes(index)
    const isEmpty = board[index] === null

    return {
      width: `${cellDimension}px`,
      height: `${cellDimension}px`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: `${cellDimension * 0.6}px`,
      fontWeight: 'bold',
      cursor: isEmpty && isValid && !gameOver ? 'pointer' : 'default',
      backgroundColor: colors.background || '#263238',
      color: board[index] === 'X' ? (colors.x || '#64B5F6') : (colors.o || '#FF8A65'),
      borderRight: (index % 3 !== 2) ? `3px solid ${colors.line || '#546E7A'}` : 'none',
      borderBottom: (index < 6) ? `3px solid ${colors.line || '#546E7A'}` : 'none',
      transition: 'background-color 0.2s',
      userSelect: 'none',
    }
  }

  const handleClick = (index) => {
    if (onCellClick && validActions.includes(index) && !gameOver) {
      onCellClick(index)
    }
  }

  // Render X symbol
  const renderX = () => (
    <svg viewBox="0 0 100 100" style={{ width: '70%', height: '70%' }}>
      <line
        x1="20" y1="20" x2="80" y2="80"
        stroke={colors.x || '#64B5F6'}
        strokeWidth="12"
        strokeLinecap="round"
      />
      <line
        x1="80" y1="20" x2="20" y2="80"
        stroke={colors.x || '#64B5F6'}
        strokeWidth="12"
        strokeLinecap="round"
      />
    </svg>
  )

  // Render O symbol
  const renderO = () => (
    <svg viewBox="0 0 100 100" style={{ width: '70%', height: '70%' }}>
      <circle
        cx="50" cy="50" r="35"
        fill="none"
        stroke={colors.o || '#FF8A65'}
        strokeWidth="12"
      />
    </svg>
  )

  // Get winner message
  const getStatusMessage = () => {
    if (winner === 'draw') return "It's a Draw!"
    if (winner) return `${winner} Wins!`
    return `${currentPlayer}'s Turn`
  }

  return (
    <div className="game-board tictactoe-board" style={{ position: 'relative' }}>
      {/* Status display */}
      <div
        className="tictactoe-status"
        style={{
          textAlign: 'center',
          marginBottom: '10px',
          fontSize: '1.2em',
          fontWeight: 'bold',
          color: winner ? (winner === 'X' ? colors.x : colors.o) : '#fff',
        }}
      >
        {getStatusMessage()}
      </div>

      {/* Game grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          width: `${boardSize}px`,
          height: `${boardSize}px`,
          backgroundColor: colors.line || '#546E7A',
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        }}
      >
        {board.map((cell, index) => (
          <div
            key={index}
            style={getCellStyle(index)}
            onClick={() => handleClick(index)}
            onMouseEnter={(e) => {
              if (validActions.includes(index) && !gameOver) {
                e.currentTarget.style.backgroundColor = '#37474F'
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = colors.background || '#263238'
            }}
          >
            {cell === 'X' && renderX()}
            {cell === 'O' && renderO()}
          </div>
        ))}
      </div>

      {/* Winner overlay */}
      {winner && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: '20px 40px',
            borderRadius: '10px',
            color: winner === 'draw' ? '#fff' : (winner === 'X' ? colors.x : colors.o),
            fontSize: '2em',
            fontWeight: 'bold',
            zIndex: 10,
            animation: 'fadeIn 0.3s ease-in',
          }}
        >
          {winner === 'draw' ? "Draw!" : `${winner} Wins!`}
        </div>
      )}
    </div>
  )
}
