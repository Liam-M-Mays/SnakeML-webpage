import React, { useState, useEffect } from "react";
import { api } from "./utils/api";
import { useTraining } from "./hooks/useTraining";
import SnakeBoard from "./components/GameBoard/SnakeBoard";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("train");
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [config, setConfig] = useState(null);
  const [models, setModels] = useState([]);

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
    error,
    startTraining,
    stopTraining,
  } = useTraining();

  // Load device info and default config
  useEffect(() => {
    api.getDeviceInfo().then(setDeviceInfo);
    api.getDefaultConfig("dqn").then(setConfig);
  }, []);

  // Load models when switching to models tab
  useEffect(() => {
    if (activeTab === "models") {
      api.listModels().then((data) => setModels(data.models || []));
    }
  }, [activeTab]);

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
    const newConfig = await api.getDefaultConfig(algo);
    setConfig(newConfig);
  };

  const handleDeleteModel = async (envName, runId) => {
    if (!confirm(`Delete model ${runId}?`)) return;
    await api.deleteModel(envName, runId);
    const data = await api.listModels();
    setModels(data.models || []);
  };

  if (!config || !deviceInfo) {
    return <div className="app loading">Loading...</div>;
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🐍 SnakeML Playground</h1>
        <div className="header-info">
          <span className="device-badge">
            Device: {deviceInfo.name || deviceInfo.type}
          </span>
          {isTraining && (
            <span className="training-badge">● Training</span>
          )}
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
          className={activeTab === "config" ? "active" : ""}
          onClick={() => setActiveTab("config")}
        >
          Configuration
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
        {/* Training Tab */}
        {activeTab === "train" && (
          <div className="train-tab">
            <div className="train-controls">
              <h2>Training Control</h2>

              <div className="button-group">
                {!isTraining ? (
                  <button
                    className="btn btn-primary"
                    onClick={handleStartTraining}
                  >
                    Start Training
                  </button>
                ) : (
                  <button
                    className="btn btn-danger"
                    onClick={handleStopTraining}
                  >
                    Stop Training
                  </button>
                )}
              </div>

              {error && <div className="error-message">{error}</div>}
            </div>

            {/* Training Dashboard */}
            {isTraining && (
              <div className="dashboard">
                <h2>Training Progress</h2>

                <div className={`game-container ${isFullscreen ? "fullscreen" : ""}`}>
                  <div className="game-controls">
                    <div className="left">
                      <div className="speed-buttons">
                        {["1x", "2x", "5x", "10x", "MAX"].map((speed) => (
                          <button
                            key={speed}
                            className={tickSpeed === speed ? "active" : ""}
                            onClick={() => setTickSpeed(speed)}
                          >
                            {speed}
                          </button>
                        ))}
                      </div>
                      {progress?.is_max_speed && (
                        <div className="max-speed-indicator">
                          ⚡ MAX SPEED MODE -
                          {" "}
                          {progress?.episodes_per_second
                            ? progress.episodes_per_second.toFixed(1)
                            : "0.0"}
                          {" "}
                          eps/sec
                        </div>
                      )}
                    </div>
                    <div className="right">
                      {/* TODO: connect pause/resume to backend training control if exposed */}
                      <button
                        className="btn btn-secondary"
                        onClick={() => setIsPaused((prev) => !prev)}
                      >
                        {isPaused ? "Resume" : "Pause"}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => setIsFullscreen((prev) => !prev)}
                      >
                        {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                      </button>
                    </div>
                  </div>

                  {isPaused && (
                    <div className="info-message">
                      Visualization paused. Training continues in the background.
                    </div>
                  )}

                  <SnakeBoard gameState={gameState || progress?.game_state} fullscreen={isFullscreen} />
                </div>

                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Episode</div>
                    <div className="stat-value">
                      {progress?.episode || status?.episode_count || 0}
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-label">Current Score</div>
                    <div className="stat-value">
                      {progress?.current_score || 0}
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-label">Best Score</div>
                    <div className="stat-value">
                      {progress?.best_score || status?.best_score || 0}
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-label">Avg Score (100 eps)</div>
                    <div className="stat-value">
                      {progress?.avg_score || status?.avg_score || 0}
                    </div>
                  </div>
                </div>

                {/* Recent Episodes */}
                {episodes.length > 0 && (
                  <div className="recent-episodes">
                    <h3>Recent Episodes</h3>
                    <div className="episodes-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Episode</th>
                            <th>Score</th>
                            <th>Reward</th>
                            <th>Length</th>
                            <th>Death</th>
                            <th>Loss</th>
                            <th>Epsilon / Entropy</th>
                          </tr>
                        </thead>
                        <tbody>
                          {episodes.slice(-10).reverse().map((ep, idx) => (
                            <tr key={idx}>
                              <td>{ep.episode}</td>
                              <td>{ep.score}</td>
                              <td>{ep.reward?.toFixed(2) || "N/A"}</td>
                              <td>{ep.length}</td>
                              <td>{ep.death_reason || "N/A"}</td>
                              <td>{ep.loss?.toFixed(4) || "N/A"}</td>
                              <td>
                                {ep.epsilon?.toFixed(3) ||
                                  ep.entropy?.toFixed(3) ||
                                  "N/A"}
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

            {!isTraining && (
              <div className="info-message">
                Configure your training parameters in the Configuration tab, then click Start Training.
              </div>
            )}
          </div>
        )}

        {/* Configuration Tab */}
        {activeTab === "config" && (
          <div className="config-tab">
            <h2>Training Configuration</h2>

            <div className="config-section">
              <h3>Algorithm</h3>
              <select
                value={config.algo}
                onChange={(e) => handleAlgoChange(e.target.value)}
                disabled={isTraining}
              >
                <option value="dqn">DQN (Deep Q-Network)</option>
                <option value="ppo">PPO (Proximal Policy Optimization)</option>
              </select>
            </div>

            <div className="config-section">
              <h3>Environment</h3>
              <div className="param-grid">
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
                          grid_size: parseInt(e.target.value),
                        },
                      })
                    }
                    disabled={isTraining}
                    min="5"
                    max="30"
                  />
                </label>
              </div>
            </div>

            <div className="config-section">
              <h3>Rewards</h3>
              <div className="param-grid">
                {Object.entries(config.reward_config || {}).map(([key, value]) => (
                  <label key={key}>
                    {key}:
                    <input
                      type="number"
                      step="0.1"
                      value={value}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          reward_config: {
                            ...config.reward_config,
                            [key]: parseFloat(e.target.value),
                          },
                        })
                      }
                      disabled={isTraining}
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="config-section">
              <h3>Hyperparameters</h3>
              <div className="param-grid">
                {Object.entries(config.hyperparams || {}).map(([key, value]) => (
                  <label key={key}>
                    {key.replace(/_/g, " ")}:
                    <input
                      type="number"
                      step="any"
                      value={value}
                      onChange={(e) => {
                        const newValue = e.target.value.includes(".")
                          ? parseFloat(e.target.value)
                          : parseInt(e.target.value);
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

            <div className="config-section">
              <h3>Network Architecture</h3>
              <div className="network-info">
                {config.network_config?.layers?.map((layer, idx) => (
                  <div key={idx} className="layer-card">
                    Layer {idx + 1}: {layer.units} units, {layer.activation}
                  </div>
                ))}
              </div>
              <p className="info-text">
                <small>
                  Network architecture can be customized by editing the config.
                  Future versions will include a visual network builder.
                </small>
              </p>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === "models" && (
          <div className="models-tab">
            <h2>Saved Models</h2>

            {models.length === 0 ? (
              <div className="info-message">
                No saved models yet. Train a model and it will be saved automatically when you stop training.
              </div>
            ) : (
              <div className="models-list">
                {models.map((model) => (
                  <div key={`${model.env_name}-${model.run_id}`} className="model-card">
                    <div className="model-header">
                      <h3>{model.name || model.run_id}</h3>
                      <span className="model-algo">{model.algo.toUpperCase()}</span>
                    </div>

                    <div className="model-stats">
                      <div>Best Score: {model.best_score}</div>
                      <div>Avg Reward: {model.avg_reward}</div>
                      <div>Episodes: {model.total_episodes}</div>
                    </div>

                    <div className="model-meta">
                      <div>Environment: {model.env_name}</div>
                      <div>Created: {new Date(model.created_at).toLocaleString()}</div>
                    </div>

                    <div className="model-actions">
                      <button
                        className="btn btn-small btn-danger"
                        onClick={() => handleDeleteModel(model.env_name, model.run_id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          SnakeML Playground - A modular RL playground for Snake and beyond.
          <br />
          <small>
            Supports DQN and PPO algorithms with customizable networks and rewards.
          </small>
        </p>
      </footer>
    </div>
  );
}

export default App;
