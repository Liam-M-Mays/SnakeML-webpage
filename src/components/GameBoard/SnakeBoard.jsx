import React from "react";
import "./SnakeBoard.css";

function SnakeBoard({ gameState, fullscreen = false }) {
  if (!gameState) {
    return <div className="snake-board placeholder">No game running</div>;
  }

  const {
    grid_size: gridSize,
    snake_position: snakePosition = [],
    food_position: foodPosition,
    game_over: gameOver,
    death_reason: deathReason,
    score,
  } = gameState;

  const head = snakePosition[0];
  const bodySet = new Set(
    (snakePosition || []).slice(1).map((seg) => `${seg.x}-${seg.y}`)
  );
  const headKey = head ? `${head.x}-${head.y}` : null;
  const foodKey = foodPosition ? `${foodPosition.x}-${foodPosition.y}` : null;

  const cells = [];
  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const key = `${x}-${y}`;
      const isHead = headKey === key;
      const isBody = bodySet.has(key);
      const isFood = foodKey === key;

      let content = null;
      if (isHead) {
        content = <div className="snake-cell-head" />;
      } else if (isBody) {
        content = <div className="snake-cell-body" />;
      } else if (isFood) {
        content = <div className="snake-cell-food" />;
      }

      cells.push(
        <div className="snake-cell" key={key}>
          {content}
        </div>
      );
    }
  }

  return (
    <div className={`snake-board-wrapper ${fullscreen ? "fullscreen" : ""}`}>
      <div
        className="snake-board"
        style={{
          gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
          gridTemplateRows: `repeat(${gridSize}, 1fr)`,
        }}
      >
        {cells}
        {gameOver && (
          <div className="snake-board-overlay">
            <div className="overlay-content">
              <h3>Game Over</h3>
              {deathReason && <p>Reason: {deathReason}</p>}
              <p>Final Score: {score}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SnakeBoard;
