/**
 * Replay Viewer component for watching saved game replays.
 * Renders as a modal overlay.
 */

import React, { useState, useEffect, useRef } from "react";
import SnakeBoard from "./GameBoard/SnakeBoard";

// Simple snake game engine for replay simulation
class ReplayEngine {
  constructor(gridSize = 10, initialState = null) {
    this.gridSize = gridSize;
    this.initialState = initialState;
    this.reset();
  }

  reset() {
    if (this.initialState) {
      // Use initial state from replay
      this.snake = [...this.initialState.snake_position];
      this.food = { ...this.initialState.food_position };
    } else {
      const center = Math.floor(this.gridSize / 2);
      this.snake = [{ x: center, y: center }];
      this.food = { x: center + 2, y: center };
    }
    this.direction = { x: 1, y: 0 };
    this.score = 0;
    this.gameOver = false;
    this.deathReason = null;
  }

  placeFood() {
    let position;
    let attempts = 0;
    do {
      position = {
        x: Math.floor(Math.random() * this.gridSize),
        y: Math.floor(Math.random() * this.gridSize),
      };
      attempts++;
    } while (
      this.snake.some((s) => s.x === position.x && s.y === position.y) &&
      attempts < 100
    );
    this.food = position;
  }

  step(action) {
    if (this.gameOver) return this.getState();

    // Update direction based on action (0=straight, 1=right, 2=left)
    if (action === 1) {
      // Turn right
      const newDir = { x: -this.direction.y, y: this.direction.x };
      this.direction = newDir;
    } else if (action === 2) {
      // Turn left
      const newDir = { x: this.direction.y, y: -this.direction.x };
      this.direction = newDir;
    }

    // Calculate new head position
    const head = this.snake[0];
    const newHead = {
      x: head.x + this.direction.x,
      y: head.y + this.direction.y,
    };

    // Check wall collision
    if (
      newHead.x < 0 ||
      newHead.x >= this.gridSize ||
      newHead.y < 0 ||
      newHead.y >= this.gridSize
    ) {
      this.gameOver = true;
      this.deathReason = "wall";
      return this.getState();
    }

    // Check self collision
    if (this.snake.some((s) => s.x === newHead.x && s.y === newHead.y)) {
      this.gameOver = true;
      this.deathReason = "self";
      return this.getState();
    }

    // Move snake
    this.snake.unshift(newHead);

    // Check food collision
    if (newHead.x === this.food.x && newHead.y === this.food.y) {
      this.score++;
      this.placeFood();
    } else {
      this.snake.pop();
    }

    return this.getState();
  }

  getState() {
    return {
      snake_position: [...this.snake],
      food_position: { ...this.food },
      score: this.score,
      game_over: this.gameOver,
      death_reason: this.deathReason,
      grid_size: this.gridSize,
    };
  }
}

function ReplayViewer({ replay, onClose }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [speed, setSpeed] = useState(100);
  const [gameState, setGameState] = useState(null);

  const engineRef = useRef(null);
  const intervalRef = useRef(null);
  const actionsRef = useRef([]);

  // Initialize engine when replay changes
  useEffect(() => {
    if (!replay) return;

    actionsRef.current = replay.actions || [];
    const gridSize = replay.grid_size || 10;
    const initialState = replay.initial_state || null;

    engineRef.current = new ReplayEngine(gridSize, initialState);
    setGameState(engineRef.current.getState());
    setCurrentStep(0);
    setIsPlaying(false);
  }, [replay]);

  // Handle playback
  useEffect(() => {
    if (!isPlaying || !engineRef.current) return;

    intervalRef.current = setInterval(() => {
      if (currentStep >= actionsRef.current.length) {
        setIsPlaying(false);
        return;
      }

      const action = actionsRef.current[currentStep];
      const state = engineRef.current.step(action);
      setGameState(state);
      setCurrentStep((prev) => prev + 1);

      if (state.game_over) {
        setIsPlaying(false);
      }
    }, speed);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, currentStep, speed]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleStepForward = () => {
    if (currentStep >= actionsRef.current.length || !engineRef.current) return;

    const action = actionsRef.current[currentStep];
    const state = engineRef.current.step(action);
    setGameState(state);
    setCurrentStep((prev) => prev + 1);
  };

  const handleReset = () => {
    if (!engineRef.current || !replay) return;

    const gridSize = replay.grid_size || 10;
    const initialState = replay.initial_state || null;
    engineRef.current = new ReplayEngine(gridSize, initialState);
    setGameState(engineRef.current.getState());
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const handleSeek = (e) => {
    const targetStep = parseInt(e.target.value);
    if (!replay) return;

    // Reset and replay to target step
    const gridSize = replay.grid_size || 10;
    const initialState = replay.initial_state || null;
    engineRef.current = new ReplayEngine(gridSize, initialState);

    for (let i = 0; i < targetStep && i < actionsRef.current.length; i++) {
      engineRef.current.step(actionsRef.current[i]);
    }
    setGameState(engineRef.current.getState());
    setCurrentStep(targetStep);
  };

  if (!replay) return null;

  const totalSteps = actionsRef.current.length;

  return (
    <div className="replay-modal" onClick={onClose}>
      <div className="replay-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="replay-modal-header">
          <h3>Replay - Episode {replay.episode || "?"}</h3>
          <button className="btn btn-small" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="game-container">
          <div className="game-container-inner">
            <SnakeBoard gameState={gameState} />
          </div>
        </div>

        <div className="replay-controls">
          <button className="btn" onClick={handleReset}>
            Reset
          </button>
          <button className="btn btn-primary" onClick={handlePlayPause}>
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button
            className="btn"
            onClick={handleStepForward}
            disabled={isPlaying || currentStep >= totalSteps}
          >
            Step
          </button>
          <select
            className="btn"
            value={speed}
            onChange={(e) => setSpeed(parseInt(e.target.value))}
          >
            <option value="200">0.5x</option>
            <option value="100">1x</option>
            <option value="50">2x</option>
            <option value="25">4x</option>
          </select>
        </div>

        <div className="replay-timeline">
          <label>
            Step {currentStep} / {totalSteps}
            <input
              type="range"
              min="0"
              max={totalSteps}
              value={currentStep}
              onChange={handleSeek}
              disabled={isPlaying}
            />
          </label>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Score</div>
            <div className="stat-value">{gameState?.score ?? replay.score}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Steps</div>
            <div className="stat-value">{replay.length || totalSteps}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Death</div>
            <div className="stat-value">
              {gameState?.death_reason || replay.death_reason || "-"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ReplayViewer;
