/**
 * Snake Game Exports
 *
 * Re-exports all Snake game components for easy importing.
 */

import * as snake from './snake'

// Export snake components directly
export const config = snake.config
export const Board = snake.Board
export const Settings = snake.Settings
export const useGameController = snake.useGameController
export const InputVisualization = snake.InputVisualization

// Helper to apply theme
export const applyTheme = () => {
  const theme = config.theme
  if (!theme) return

  const root = document.documentElement
  Object.entries(theme).forEach(([key, value]) => {
    const cssVar = `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`
    root.style.setProperty(cssVar, value)
  })
}

export default config
