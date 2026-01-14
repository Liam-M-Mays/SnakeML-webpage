import React from 'react'

/**
 * Snake game board component.
 * Renders the grid with snake (with eyes!) and food.
 * Supports debug visualization for danger detection and shortest path.
 */
export default function SnakeBoard({ settings, gameState, cellSize }) {
  const { gridSize, colors, debug = {} } = settings

  // Extract snake-specific state from gameState (opaque to App.jsx)
  const { snake, food, debugData = {} } = gameState || {}
  const { danger_cells = [], path_cells = [], segment_cells = [], segment_connections = [] } = debugData

  // Calculate direction snake is facing (from head to second segment)
  const getDirection = () => {
    if (!snake || snake.length < 2 || snake[0].x === null) {
      return { dx: 1, dy: 0 } // Default: facing right
    }
    const head = snake[0]
    const neck = snake[1]

    // Direction the snake moved: head position minus neck position
    let dx = head.x - neck.x
    let dy = head.y - neck.y

    // Handle wrap-around (if snake wraps, direction will be large)
    // Large positive dx means head wrapped to left side (was moving LEFT, so dx should be -1)
    // Large negative dx means head wrapped to right side (was moving RIGHT, so dx should be +1)
    if (Math.abs(dx) > 1) dx = dx > 0 ? -1 : 1
    if (Math.abs(dy) > 1) dy = dy > 0 ? -1 : 1

    // Clamp to -1, 0, 1
    dx = Math.max(-1, Math.min(1, dx))
    dy = Math.max(-1, Math.min(1, dy))

    // Default to facing right if no movement detected
    if (dx === 0 && dy === 0) dx = 1

    return { dx, dy }
  }

  // Get eye positions based on direction
  const getEyeStyle = (direction, isLeftEye) => {
    const { dx, dy } = direction
    const eyeSize = Math.max(3, cellSize * 0.18)
    const eyeOffset = cellSize * 0.22
    const forwardOffset = cellSize * 0.25

    let left, top

    if (dy < 0) {
      // Facing up (dy is negative)
      top = forwardOffset - eyeSize / 2
      left = isLeftEye ? eyeOffset : cellSize - eyeOffset - eyeSize
    } else if (dy > 0) {
      // Facing down (dy is positive)
      top = cellSize - forwardOffset - eyeSize / 2
      left = isLeftEye ? cellSize - eyeOffset - eyeSize : eyeOffset
    } else if (dx > 0) {
      // Facing right
      left = cellSize - forwardOffset - eyeSize / 2
      top = isLeftEye ? eyeOffset : cellSize - eyeOffset - eyeSize
    } else {
      // Facing left (dx <= 0)
      left = forwardOffset - eyeSize / 2
      top = isLeftEye ? cellSize - eyeOffset - eyeSize : eyeOffset
    }

    return {
      position: 'absolute',
      width: `${eyeSize}px`,
      height: `${eyeSize}px`,
      borderRadius: '50%',
      backgroundColor: '#111',
      left: `${left}px`,
      top: `${top}px`,
      boxShadow: 'inset 1px 1px 2px rgba(255,255,255,0.3)',
    }
  }

  // Determine segment type for styling
  const getSegmentInfo = (x, y) => {
    if (!snake || snake.length === 0) return null

    const index = snake.findIndex(seg => seg.x === x && seg.y === y)
    if (index === -1) return null

    const isHead = index === 0
    const isTail = index === snake.length - 1
    const progress = snake.length > 1 ? index / (snake.length - 1) : 0

    // Determine connections for rounded corners
    let prev = null, next = null
    if (index > 0) prev = snake[index - 1]
    if (index < snake.length - 1) next = snake[index + 1]

    return { index, isHead, isTail, progress, prev, next }
  }

  // Calculate border radius for smooth snake body
  const getBorderRadius = (segInfo, x, y) => {
    if (!segInfo) return '2px'
    const { isHead, isTail, prev, next } = segInfo

    if (isHead) return '4px'
    if (isTail) return '4px'

    // For body segments, round corners that aren't connected
    const hasTop = (prev?.y === y - 1) || (next?.y === y - 1)
    const hasBottom = (prev?.y === y + 1) || (next?.y === y + 1)
    const hasLeft = (prev?.x === x - 1) || (next?.x === x - 1)
    const hasRight = (prev?.x === x + 1) || (next?.x === x + 1)

    const r = '3px'
    const corners = [
      (!hasTop && !hasLeft) ? r : '0',
      (!hasTop && !hasRight) ? r : '0',
      (!hasBottom && !hasRight) ? r : '0',
      (!hasBottom && !hasLeft) ? r : '0',
    ]

    return corners.join(' ')
  }

  const direction = getDirection()

  // Check if a cell is in the danger detection zone
  const getDangerCellInfo = (x, y) => {
    if (!debug.vision) return null
    return danger_cells.find(cell => cell.x === x && cell.y === y)
  }

  // Check if a cell is on the shortest path
  const isOnPath = (x, y) => {
    if (!debug.path) return false
    return path_cells.some(cell => cell.x === x && cell.y === y)
  }

  // Check if a cell is a tracked segment
  const getSegmentCellInfo = (x, y) => {
    if (!debug.segments) return null
    return segment_cells.find(cell => cell.x === x && cell.y === y)
  }

  const renderGrid = () => {
    const cells = []
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const segInfo = getSegmentInfo(x, y)
        const isSnake = segInfo !== null
        const isFood = food.x === x && food.y === y

        // Debug info for this cell
        const dangerInfo = getDangerCellInfo(x, y)
        const onPath = isOnPath(x, y)
        const segmentInfo = getSegmentCellInfo(x, y)

        let cellClass = 'cell'
        if (isSnake) cellClass += ' snake'
        if (segInfo?.isHead) cellClass += ' snake-head'
        if (segInfo?.isTail) cellClass += ' snake-tail'
        if (isFood) cellClass += ' food'

        // Calculate body gradient (darker toward tail)
        const segmentStyle = {}
        if (isSnake && !segInfo.isHead) {
          const darkness = segInfo.progress * 0.35
          segmentStyle.filter = `brightness(${1 - darkness})`
          segmentStyle.borderRadius = getBorderRadius(segInfo, x, y)
        }
        if (segInfo?.isHead) {
          segmentStyle.borderRadius = '4px'
          segmentStyle.boxShadow = '0 0 4px rgba(0,0,0,0.5)'
          segmentStyle.zIndex = 2
        }
        if (segInfo?.isTail) {
          segmentStyle.borderRadius = '4px'
          // Make tail slightly smaller
          segmentStyle.transform = 'scale(0.85)'
        }

        cells.push(
          <div
            key={`${x}-${y}`}
            className={cellClass}
            style={segmentStyle}
          >
            {/* Render eyes on head */}
            {segInfo?.isHead && (
              <>
                <div style={getEyeStyle(direction, true)} className="snake-eye" />
                <div style={getEyeStyle(direction, false)} className="snake-eye" />
              </>
            )}

            {/* Danger detection overlay */}
            {dangerInfo && (
              <div
                className="debug-overlay danger-overlay"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: dangerInfo.danger
                    ? 'rgba(255, 80, 80, 0.5)'   // Red for danger
                    : 'rgba(80, 255, 80, 0.35)', // Green for safe
                  border: dangerInfo.danger
                    ? '2px solid rgba(255, 0, 0, 0.7)'
                    : '2px solid rgba(0, 255, 0, 0.5)',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: `${Math.max(10, cellSize * 0.4)}px`,
                  fontWeight: 'bold',
                  color: dangerInfo.danger ? '#ff0000' : '#00ff00',
                  textShadow: '1px 1px 2px rgba(0,0,0,0.8)',
                }}
              >
                {dangerInfo.label}
              </div>
            )}

            {/* Shortest path overlay */}
            {onPath && !isSnake && !isFood && (
              <div
                className="debug-overlay path-overlay"
                style={{
                  position: 'absolute',
                  top: '15%',
                  left: '15%',
                  width: '70%',
                  height: '70%',
                  backgroundColor: 'rgba(100, 200, 255, 0.6)',
                  borderRadius: '50%',
                  pointerEvents: 'none',
                  boxShadow: '0 0 6px rgba(100, 200, 255, 0.8)',
                }}
              />
            )}

            {/* Segment tracking overlay */}
            {segmentInfo && (
              <div
                className="debug-overlay segment-overlay"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(150, 150, 150, 0.4)',
                  border: '2px solid rgba(200, 200, 200, 0.7)',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: `${Math.max(10, cellSize * 0.35)}px`,
                  fontWeight: 'bold',
                  color: '#fff',
                  textShadow: '1px 1px 3px rgba(0,0,0,0.9)',
                }}
              >
                {segmentInfo.label}
              </div>
            )}
          </div>
        )
      }
    }
    return cells
  }

  // Render SVG lines for segment connections
  const renderSegmentConnections = () => {
    if (!debug.segments || segment_connections.length === 0) return null

    const boardSize = gridSize * cellSize
    return (
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: boardSize,
          height: boardSize,
          pointerEvents: 'none',
          zIndex: 10,
        }}
      >
        {segment_connections.map((conn, idx) => {
          const x1 = conn.from.x * cellSize + cellSize / 2
          const y1 = conn.from.y * cellSize + cellSize / 2
          const x2 = conn.to.x * cellSize + cellSize / 2
          const y2 = conn.to.y * cellSize + cellSize / 2
          return (
            <line
              key={idx}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="rgba(255, 255, 255, 0.8)"
              strokeWidth="2"
              strokeDasharray="4,2"
            />
          )
        })}
      </svg>
    )
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
        position: 'relative',
      }}
    >
      {renderGrid()}
      {renderSegmentConnections()}
    </div>
  )
}
