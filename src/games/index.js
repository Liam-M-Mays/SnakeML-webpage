/**
 * Game Registry
 *
 * Central registry for all games. To add a new game:
 * 1. Create folder: src/games/yourgame/
 * 2. Add config.js, Board.jsx, Settings.jsx
 * 3. Import and add to GAMES object below
 *
 * Each game is self-contained and doesn't know about other games.
 */

import * as snake from './snake'
import * as tictactoe from './tictactoe'

// Registry of all available games
export const GAMES = {
  snake: {
    ...snake.config,
    Board: snake.Board,
    Settings: snake.Settings,
  },
  tictactoe: {
    ...tictactoe.config,
    Board: tictactoe.Board,
    Settings: tictactoe.Settings,
  },
}

// Helper to get game list for UI
export const getGameList = () => {
  return Object.entries(GAMES).map(([id, game]) => ({
    id,
    name: game.name,
    description: game.description,
    comingSoon: game.comingSoon || false,
  }))
}

// Helper to apply a game's theme
export const applyTheme = (gameId) => {
  const game = GAMES[gameId]
  if (!game?.theme) return

  const root = document.documentElement
  Object.entries(game.theme).forEach(([key, value]) => {
    // Convert camelCase to kebab-case for CSS variable
    const cssVar = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`
    root.style.setProperty(cssVar, value)
  })
}

// Default export
export default GAMES
