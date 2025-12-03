/**
 * Custom hook for human play mode.
 * Manages keyboard input, game state, and high scores.
 */

import { useState, useEffect, useCallback, useRef } from "react";

// Simple snake game engine for human play
class SnakeGame {
  constructor(gridSize = 10) {
    this.gridSize = gridSize;
    this.reset();
  }

  reset() {
    const center = Math.floor(this.gridSize / 2);
    this.snake = [{ x: center, y: center }];
    this.direction = { x: 1, y: 0 };
    this.nextDirection = { x: 1, y: 0 };
    this.score = 0;
    this.gameOver = false;
    this.deathReason = null;
    this.steps = 0;
    this.placeFood();
    return this.getState();
  }

  placeFood() {
    let position;
    do {
      position = {
        x: Math.floor(Math.random() * this.gridSize),
        y: Math.floor(Math.random() * this.gridSize),
      };
    } while (this.snake.some((s) => s.x === position.x && s.y === position.y));
    this.food = position;
  }

  setDirection(dir) {
    // Prevent 180-degree turns
    if (dir === "up" && this.direction.y !== 1) {
      this.nextDirection = { x: 0, y: -1 };
    } else if (dir === "down" && this.direction.y !== -1) {
      this.nextDirection = { x: 0, y: 1 };
    } else if (dir === "left" && this.direction.x !== 1) {
      this.nextDirection = { x: -1, y: 0 };
    } else if (dir === "right" && this.direction.x !== -1) {
      this.nextDirection = { x: 1, y: 0 };
    }
  }

  step() {
    if (this.gameOver) return this.getState();

    this.direction = this.nextDirection;
    this.steps++;

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
      snake_position: this.snake,
      food_position: this.food,
      score: this.score,
      game_over: this.gameOver,
      death_reason: this.deathReason,
      grid_size: this.gridSize,
      steps: this.steps,
    };
  }
}

const HIGHSCORE_KEY = "snakeml_human_highscore";

export function useHumanPlay(gridSize = 10) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [gameState, setGameState] = useState(null);
  const [highScore, setHighScore] = useState(0);
  const [speed, setSpeed] = useState(150); // ms per tick

  const gameRef = useRef(null);
  const intervalRef = useRef(null);

  // Load high score from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(HIGHSCORE_KEY);
    if (saved) {
      setHighScore(parseInt(saved) || 0);
    }
  }, []);

  // Update high score
  const updateHighScore = useCallback((score) => {
    if (score > highScore) {
      setHighScore(score);
      localStorage.setItem(HIGHSCORE_KEY, score.toString());
    }
  }, [highScore]);

  // Start game
  const startGame = useCallback(() => {
    gameRef.current = new SnakeGame(gridSize);
    setGameState(gameRef.current.getState());
    setIsPlaying(true);
    setIsPaused(false);
  }, [gridSize]);

  // Stop game
  const stopGame = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsPlaying(false);
    setIsPaused(false);
    if (gameRef.current) {
      updateHighScore(gameRef.current.score);
    }
  }, [updateHighScore]);

  // Toggle pause
  const togglePause = useCallback(() => {
    setIsPaused((prev) => !prev);
  }, []);

  // Handle key press
  const handleKeyPress = useCallback((event) => {
    if (!gameRef.current || !isPlaying) return;

    switch (event.key) {
      case "ArrowUp":
      case "w":
      case "W":
        event.preventDefault();
        gameRef.current.setDirection("up");
        break;
      case "ArrowDown":
      case "s":
      case "S":
        event.preventDefault();
        gameRef.current.setDirection("down");
        break;
      case "ArrowLeft":
      case "a":
      case "A":
        event.preventDefault();
        gameRef.current.setDirection("left");
        break;
      case "ArrowRight":
      case "d":
      case "D":
        event.preventDefault();
        gameRef.current.setDirection("right");
        break;
      case " ":
        event.preventDefault();
        togglePause();
        break;
      case "Escape":
        event.preventDefault();
        stopGame();
        break;
      default:
        break;
    }
  }, [isPlaying, togglePause, stopGame]);

  // Set up keyboard listener
  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    return () => {
      window.removeEventListener("keydown", handleKeyPress);
    };
  }, [handleKeyPress]);

  // Game loop
  useEffect(() => {
    if (!isPlaying || isPaused) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      if (!gameRef.current) return;

      const state = gameRef.current.step();
      setGameState(state);

      if (state.game_over) {
        updateHighScore(state.score);
        setIsPlaying(false);
      }
    }, speed);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, isPaused, speed, updateHighScore]);

  return {
    isPlaying,
    isPaused,
    gameState,
    highScore,
    speed,
    setSpeed,
    startGame,
    stopGame,
    togglePause,
  };
}
