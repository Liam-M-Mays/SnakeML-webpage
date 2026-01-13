import React from 'react'

/**
 * TicTacToe settings placeholder.
 * Will be implemented when the game is built.
 */
export default function TicTacToeSettings({ settings, onChange, disabled }) {
  return (
    <div className="game-settings-content">
      <div className="coming-soon-notice">
        <p>Tic-Tac-Toe settings coming soon!</p>
        <p className="coming-soon-detail">
          Will include opponent type (AI/Human/Minimax), who goes first, etc.
        </p>
      </div>
    </div>
  )
}
