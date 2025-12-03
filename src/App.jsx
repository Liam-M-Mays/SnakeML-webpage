import React, { useState, useEffect, useCallback, useMemo } from "react";
import { api } from "./utils/api";
import { useTraining } from "./hooks/useTraining";
import { useHumanPlay } from "./hooks/useHumanPlay";
import SnakeBoard from "./components/GameBoard/SnakeBoard";
import DeviceSelector from "./components/DeviceSelector";
import RewardDesigner from "./components/RewardDesigner";
import ObservationConfig from "./components/ObservationConfig";
import NetworkBuilder from "./components/NetworkBuilder";
import ReplayViewer from "./components/ReplayViewer";
import "./App.css";
import "./styles/theme.css";
import "./styles/layout.css";

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState("train");
  const [configSubTab, setConfigSubTab] = useState("general");

  // Data state
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [config, setConfig] = useState(null);
  const [models, setModels] = useState([]);
  const [replays, setReplays] = useState([]);
  const [selectedReplay, setSelectedReplay] = useState(null);
  const [metricsHistory, setMetricsHistory] = useState([]);

  // UI state
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelError, setModelError] = useState(null);

  // Training hook
  const {
    isTraining,
    runId,
    status,
    progress,
    gameState,
    tickSpeed,
    setTickSpeed,
    isPaused,
    setIsPaused,
    isFullscreen,
    setIsFullscreen,
    episodes,
    error: trainingError,
    startTraining,
    stopTraining,
  } = useTraining();

  // Human play hook
  const humanPlay = useHumanPlay(config?.env_config?.grid_size || 10);

  // Load device info and default config on mount
  useEffect(() => {
    api.getDeviceInfo().then(setDeviceInfo).catch(console.error);
    api.getDefaultConfig("dqn").then(setConfig).catch(console.error);
  }, []);

  // Load models when switching to models tab
  useEffect(() => {
    if (activeTab === "models") {
      loadModels();
    }
  }, [activeTab]);

  // Load metrics for current training run
  useEffect(() => {
    if (isTraining && runId && config?.env_name) {
      const interval = setInterval(() => {
        api
          .getMetrics(config.env_name, runId, 100)
          .then((data) => setMetricsHistory(data.metrics || []))
          .catch(() => {});
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [isTraining, runId, config?.env_name]);

  const loadModels = useCallback(async () => {
    setLoadingModels(true);
    setModelError(null);
    try {
      const data = await api.listModels();
      setModels(data.models || []);
    } catch (err) {
      setModelError(err.message);
    } finally {
      setLoadingModels(false);
    }
  }, []);

  const loadReplays = useCallback(async (envName, modelRunId) => {
    try {
      const data = await api.listReplays(envName, modelRunId);
      setReplays(data.replays || []);
    } catch (err) {
      console.error("Failed to load replays:", err);
      setReplays([]);
    }
  }, []);

  const handleStartTraining = async () => {
    if (!config) return;
    try {
      const maxSpeed = tickSpeed === "MAX";
      await startTraining({ ...config, max_speed: maxSpeed });
    } catch (err) {
      console.error("Failed to start training:", err);
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopTraining();
    } catch (err) {
      console.error("Failed to stop training:", err);
    }
  };

  const handleAlgoChange = async (algo) => {
    try {
      const newConfig = await api.getDefaultConfig(algo);
      setConfig(newConfig);
    } catch (err) {
      console.error("Failed to load config:", err);
    }
  };

  const handleDeviceChange = async (preference) => {
    try {
      const result = await api.setDevice(preference);
      setDeviceInfo(result);
    } catch (err) {
      console.error("Failed to set device:", err);
    }
  };

  const handleResumeTraining = async (model, continueTraining = true) => {
    try {
      // Load the model's config first
      const modelConfig = await api.getModelConfig(model.env_name, model.run_id);
      setConfig(modelConfig);

      // Resume training
      const maxSpeed = tickSpeed === "MAX";
      await api.resumeTraining(
        model.env_name,
        model.run_id,
        continueTraining,
        maxSpeed
      );
      setActiveTab("train");
    } catch (err) {
      console.error("Failed to resume training:", err);
      setModelError(err.message);
    }
  };

  const handleDeleteModel = async (envName, modelRunId) => {
    if (!confirm(`Delete model ${modelRunId}?`)) return;
    try {
      await api.deleteModel(envName, modelRunId);
      await loadModels();
    } catch (err) {
      console.error("Failed to delete model:", err);
      setModelError(err.message);
    }
  };

  const handleViewReplay = async (envName, modelRunId, replayId) => {
    try {
      const data = await api.getReplay(envName, modelRunId, replayId);
      setSelectedReplay(data);
    } catch (err) {
      console.error("Failed to load replay:", err);
    }
  };

  const handleDeleteReplay = async (envName, modelRunId, replayId) => {
    if (!confirm("Delete this replay?")) return;
    try {
      await api.deleteReplay(envName, modelRunId, replayId);
      await loadReplays(envName, modelRunId);
    } catch (err) {
      console.error("Failed to delete replay:", err);
    }
  };

  // Calculate metrics stats for charts
  const metricsStats = useMemo(() => {
    if (metricsHistory.length === 0) return null;

    const scores = metricsHistory.map((m) => m.score);
    const rewards = metricsHistory.map((m) => m.reward);
    const lengths = metricsHistory.map((m) => m.length);

    return {
      avgScore: (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2),
      maxScore: Math.max(...scores),
      avgReward: (rewards.reduce((a, b) => a + b, 0) / rewards.length).toFixed(2),
      avgLength: (lengths.reduce((a, b) => a + b, 0) / lengths.length).toFixed(1),
    };
  }, [metricsHistory]);

  // Loading state
  if (!config || !deviceInfo) {
    return (
      <div className="app loading">
        <div className="loading-spinner"></div>
        <p>Loading SnakeML Playground...</p>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>SnakeML Playground</h1>
        <div className="header-info">
          <span className="device-badge" title={deviceInfo.name}>
            {deviceInfo.type?.toUpperCase() || "CPU"}
          </span>
          {isTraining && <span className="training-badge">Training</span>}
          {humanPlay.isPlaying && <span className="training-badge">Playing</span>}
        </div>
      </header>

      {/* Tabs */}
      <nav className="tabs">
        <button
          className={activeTab === "train" ? "active" : ""}
          onClick={() => setActiveTab("train")}
        >
          Train
        </button>
        <button
          className={activeTab === "play" ? "active" : ""}
          onClick={() => setActiveTab("play")}
        >
          Play
        </button>
        <button
          className={activeTab === "config" ? "active" : ""}
          onClick={() => setActiveTab("config")}
        >
          Configure
        </button>
        <button
          className={activeTab === "models" ? "active" : ""}
          onClick={() => setActiveTab("models")}
        >
          Models
        </button>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* ========== TRAIN TAB ========== */}
        {activeTab === "train" && (
          <div className="train-tab page">
            <div className="train-controls panel">
              <div className="panel-header">
                <h2 className="panel-title">Training Control</h2>
                <div className="controls-row">
                  {!isTraining ? (
                    <button
                      className="btn btn-primary"
                      onClick={handleStartTraining}
                      disabled={humanPlay.isPlaying}
                    >
                      Start Training
                    </button>
                  ) : (
                    <button className="btn btn-danger" onClick={handleStopTraining}>
                      Stop Training
                    </button>
                  )}
                </div>
              </div>

              {trainingError && <div className="error-message">{trainingError}</div>}

              {!isTraining && (
                <p className="info-text">
                  Configure your training in the Configure tab, then start training here.
                </p>
              )}
            </div>

            {/* Training Dashboard */}
            {isTraining && (
              <div className="dashboard panel">
                <div className="panel-header">
                  <h2 className="panel-title">Training Progress</h2>
                  <div className="controls-row">
                    <div className="speed-buttons">
                      {["1x", "2x", "5x", "10x", "MAX"].map((speed) => (
                        <button
                          key={speed}
                          className={`btn ${tickSpeed === speed ? "active" : ""}`}
                          onClick={() => setTickSpeed(speed)}
                        >
                          {speed}
                        </button>
                      ))}
                    </div>
                    <button
                      className="btn"
                      onClick={() => setIsPaused((prev) => !prev)}
                    >
                      {isPaused ? "Resume Viz" : "Pause Viz"}
                    </button>
                    <button
                      className="btn"
                      onClick={() => setIsFullscreen((prev) => !prev)}
                    >
                      {isFullscreen ? "Exit FS" : "Fullscreen"}
                    </button>
                  </div>
                </div>

                {progress?.is_max_speed && (
                  <div className="max-speed-indicator">
                    MAX SPEED MODE -{" "}
                    {progress?.episodes_per_second?.toFixed(1) || "0.0"} eps/sec
                  </div>
                )}

                <div
                  className={`game-container ${isFullscreen ? "fullscreen" : ""}`}
                >
                  <div className="game-container-inner">
                    <SnakeBoard
                      gameState={gameState || progress?.game_state}
                      fullscreen={isFullscreen}
                    />
                  </div>
                </div>

                {isPaused && (
                  <p className="info-text">
                    Visualization paused. Training continues in background.
                  </p>
                )}

                {/* Stats Grid */}
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Episode</div>
                    <div className="stat-value">
                      {progress?.episode || status?.episode_count || 0}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Score</div>
                    <div className="stat-value">{progress?.current_score || 0}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Best</div>
                    <div className="stat-value">
                      {progress?.best_score || status?.best_score || 0}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Avg (100)</div>
                    <div className="stat-value">
                      {(progress?.avg_score || status?.avg_score || 0).toFixed(1)}
                    </div>
                  </div>
                </div>

                {/* Metrics Summary */}
                {metricsStats && (
                  <div className="metrics-summary">
                    <h3>Session Metrics</h3>
                    <div className="stats-grid">
                      <div className="stat-card">
                        <div className="stat-label">Avg Score</div>
                        <div className="stat-value">{metricsStats.avgScore}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Max Score</div>
                        <div className="stat-value">{metricsStats.maxScore}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Avg Reward</div>
                        <div className="stat-value">{metricsStats.avgReward}</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Avg Steps</div>
                        <div className="stat-value">{metricsStats.avgLength}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Recent Episodes Table */}
                {episodes.length > 0 && (
                  <div className="recent-episodes">
                    <h3>Recent Episodes</h3>
                    <div className="episodes-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Ep</th>
                            <th>Score</th>
                            <th>Reward</th>
                            <th>Steps</th>
                            <th>Death</th>
                            <th>Loss</th>
                            <th>Eps/Ent</th>
                          </tr>
                        </thead>
                        <tbody>
                          {episodes
                            .slice(-10)
                            .reverse()
                            .map((ep, idx) => (
                              <tr key={idx}>
                                <td>{ep.episode}</td>
                                <td>{ep.score}</td>
                                <td>{ep.reward?.toFixed(1) ?? "-"}</td>
                                <td>{ep.length}</td>
                                <td>{ep.death_reason || "-"}</td>
                                <td>{ep.loss?.toFixed(4) ?? "-"}</td>
                                <td>
                                  {ep.epsilon?.toFixed(3) ??
                                    ep.entropy?.toFixed(3) ??
                                    "-"}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ========== PLAY TAB ========== */}
        {activeTab === "play" && (
          <div className="play-tab page">
            <div className="panel">
              <div className="panel-header">
                <h2 className="panel-title">Human Play Mode</h2>
                <div className="controls-row">
                  {!humanPlay.isPlaying ? (
                    <button
                      className="btn btn-primary"
                      onClick={humanPlay.startGame}
                      disabled={isTraining}
                    >
                      Start Game
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn"
                        onClick={humanPlay.togglePause}
                      >
                        {humanPlay.isPaused ? "Resume" : "Pause"}
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={humanPlay.stopGame}
                      >
                        End Game
                      </button>
                    </>
                  )}
                </div>
              </div>

              <p className="info-text">
                Use Arrow keys or WASD to move. Press Space to pause, Escape to quit.
              </p>

              {/* Speed Control */}
              <div className="speed-control">
                <label>
                  Speed:{" "}
                  <input
                    type="range"
                    min="50"
                    max="300"
                    step="25"
                    value={300 - humanPlay.speed}
                    onChange={(e) => humanPlay.setSpeed(300 - parseInt(e.target.value))}
                    disabled={humanPlay.isPlaying && !humanPlay.isPaused}
                  />
                  <span>{humanPlay.speed}ms</span>
                </label>
              </div>
            </div>

            {/* Game Display */}
            <div className="game-container">
              <div className="game-container-inner">
                {humanPlay.gameState ? (
                  <SnakeBoard gameState={humanPlay.gameState} />
                ) : (
                  <div className="game-placeholder">
                    <p>Press Start Game to play!</p>
                  </div>
                )}
              </div>
            </div>

            {/* Stats */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-label">Score</div>
                <div className="stat-value">
                  {humanPlay.gameState?.score || 0}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">High Score</div>
                <div className="stat-value">{humanPlay.highScore}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Steps</div>
                <div className="stat-value">
                  {humanPlay.gameState?.steps || 0}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Status</div>
                <div className="stat-value">
                  {humanPlay.gameState?.game_over
                    ? `Dead: ${humanPlay.gameState.death_reason}`
                    : humanPlay.isPaused
                    ? "Paused"
                    : humanPlay.isPlaying
                    ? "Playing"
                    : "Ready"}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========== CONFIGURE TAB ========== */}
        {activeTab === "config" && (
          <div className="config-tab page">
            {/* Sub-tabs */}
            <div className="config-subtabs">
              {["general", "rewards", "observation", "network", "device"].map(
                (tab) => (
                  <button
                    key={tab}
                    className={`subtab ${configSubTab === tab ? "active" : ""}`}
                    onClick={() => setConfigSubTab(tab)}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                )
              )}
            </div>

            {/* General Config */}
            {configSubTab === "general" && (
              <div className="config-section panel">
                <h3>General Settings</h3>

                <div className="config-group">
                  <label>
                    Algorithm:
                    <select
                      value={config.algo}
                      onChange={(e) => handleAlgoChange(e.target.value)}
                      disabled={isTraining}
                    >
                      <option value="dqn">DQN (Deep Q-Network)</option>
                      <option value="ppo">PPO (Proximal Policy Optimization)</option>
                    </select>
                  </label>
                </div>

                <div className="config-group">
                  <label>
                    Grid Size:
                    <input
                      type="number"
                      value={config.env_config?.grid_size || 10}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          env_config: {
                            ...config.env_config,
                            grid_size: parseInt(e.target.value) || 10,
                          },
                        })
                      }
                      disabled={isTraining}
                      min="5"
                      max="30"
                    />
                  </label>
                </div>

                <h4>Hyperparameters</h4>
                <div className="param-grid">
                  {Object.entries(config.hyperparams || {}).map(([key, value]) => (
                    <label key={key}>
                      {key.replace(/_/g, " ")}:
                      <input
                        type="number"
                        step="any"
                        value={value}
                        onChange={(e) => {
                          const val = e.target.value;
                          const newValue =
                            val.includes(".") || val.includes("e")
                              ? parseFloat(val)
                              : parseInt(val);
                          setConfig({
                            ...config,
                            hyperparams: {
                              ...config.hyperparams,
                              [key]: newValue,
                            },
                          });
                        }}
                        disabled={isTraining}
                      />
                    </label>
                  ))}
                </div>
              </div>
            )}

            {/* Rewards Config */}
            {configSubTab === "rewards" && (
              <div className="config-section panel">
                <RewardDesigner
                  rewardConfig={config.reward_config || {}}
                  onChange={(newRewards) =>
                    setConfig({ ...config, reward_config: newRewards })
                  }
                  disabled={isTraining}
                />
              </div>
            )}

            {/* Observation Config */}
            {configSubTab === "observation" && (
              <div className="config-section panel">
                <ObservationConfig
                  envConfig={config.env_config || {}}
                  onChange={(newEnvConfig) =>
                    setConfig({ ...config, env_config: newEnvConfig })
                  }
                  disabled={isTraining}
                />
              </div>
            )}

            {/* Network Config */}
            {configSubTab === "network" && (
              <div className="config-section panel">
                <NetworkBuilder
                  networkConfig={config.network_config || { layers: [] }}
                  onChange={(newNetwork) =>
                    setConfig({ ...config, network_config: newNetwork })
                  }
                  disabled={isTraining}
                />
              </div>
            )}

            {/* Device Config */}
            {configSubTab === "device" && (
              <div className="config-section panel">
                <DeviceSelector
                  deviceInfo={deviceInfo}
                  onDeviceChange={handleDeviceChange}
                  disabled={isTraining}
                />
              </div>
            )}
          </div>
        )}

        {/* ========== MODELS TAB ========== */}
        {activeTab === "models" && (
          <div className="models-tab page">
            <div className="panel">
              <div className="panel-header">
                <h2 className="panel-title">Saved Models</h2>
                <button className="btn" onClick={loadModels} disabled={loadingModels}>
                  {loadingModels ? "Loading..." : "Refresh"}
                </button>
              </div>

              {modelError && <div className="error-message">{modelError}</div>}

              {models.length === 0 && !loadingModels ? (
                <p className="info-text">
                  No saved models yet. Train a model and it will be saved when you
                  stop training.
                </p>
              ) : (
                <div className="models-list">
                  {models.map((model) => (
                    <div
                      key={`${model.env_name}-${model.run_id}`}
                      className="model-card"
                    >
                      <div className="model-header">
                        <h3>{model.name || model.run_id}</h3>
                        <span className="model-algo">
                          {model.algo?.toUpperCase() || "DQN"}
                        </span>
                      </div>

                      <div className="model-stats">
                        <div>
                          <strong>Best:</strong> {model.best_score ?? "-"}
                        </div>
                        <div>
                          <strong>Avg Reward:</strong>{" "}
                          {model.avg_reward?.toFixed(1) ?? "-"}
                        </div>
                        <div>
                          <strong>Episodes:</strong> {model.total_episodes ?? "-"}
                        </div>
                      </div>

                      <div className="model-meta">
                        <div>Env: {model.env_name}</div>
                        <div>
                          Created:{" "}
                          {model.created_at
                            ? new Date(model.created_at).toLocaleString()
                            : "-"}
                        </div>
                      </div>

                      <div className="model-actions">
                        <button
                          className="btn btn-primary"
                          onClick={() => handleResumeTraining(model, true)}
                          disabled={isTraining}
                        >
                          Resume
                        </button>
                        <button
                          className="btn"
                          onClick={() => handleResumeTraining(model, false)}
                          disabled={isTraining}
                        >
                          Watch
                        </button>
                        <button
                          className="btn"
                          onClick={() => loadReplays(model.env_name, model.run_id)}
                        >
                          Replays
                        </button>
                        <button
                          className="btn btn-danger"
                          onClick={() =>
                            handleDeleteModel(model.env_name, model.run_id)
                          }
                          disabled={isTraining}
                        >
                          Delete
                        </button>
                      </div>

                      {/* Replays for this model */}
                      {replays.length > 0 &&
                        replays[0]?.run_id === model.run_id && (
                          <div className="model-replays">
                            <h4>Replays</h4>
                            <div className="replay-list">
                              {replays.map((replay) => (
                                <div key={replay.id} className="replay-item">
                                  <span>
                                    Ep {replay.episode} - Score {replay.score}
                                  </span>
                                  <div className="replay-actions">
                                    <button
                                      className="btn btn-small"
                                      onClick={() =>
                                        handleViewReplay(
                                          model.env_name,
                                          model.run_id,
                                          replay.id
                                        )
                                      }
                                    >
                                      View
                                    </button>
                                    <button
                                      className="btn btn-small btn-danger"
                                      onClick={() =>
                                        handleDeleteReplay(
                                          model.env_name,
                                          model.run_id,
                                          replay.id
                                        )
                                      }
                                    >
                                      Delete
                                    </button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Replay Viewer Modal */}
      {selectedReplay && (
        <ReplayViewer
          replay={selectedReplay}
          onClose={() => setSelectedReplay(null)}
        />
      )}

      {/* Footer */}
      <footer className="footer">
        <p>
          SnakeML Playground - A modular RL playground for Snake and beyond.
        </p>
      </footer>
    </div>
  );
}

export default App;
