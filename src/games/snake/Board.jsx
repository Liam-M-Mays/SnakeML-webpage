import React from 'react'

/**
 * Snake game board component.
 * Renders the grid with snake and food.
 */
export default function SnakeBoard({ settings, snake, food, cellSize }) {
  const { gridSize, colors } = settings

  const renderGrid = () => {
    const cells = []
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const isSnake = snake.some(segment => segment.x === x && segment.y === y)
        const isFood = food.x === x && food.y === y

        let cellClass = 'cell'
        if (isSnake) cellClass += ' snake'
        if (isFood) cellClass += ' food'

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

  return (
    <div
      className="game-board snake-board"
      style={{
        gridTemplateColumns: `repeat(${gridSize}, ${cellSize}px)`,
        gridTemplateRows: `repeat(${gridSize}, ${cellSize}px)`,
        '--snake-color': colors.snake,
        '--food-color': colors.food,
        '--grid-bg': colors.background,
      }}
    >
      {renderGrid()}
    </div>
  )
}
