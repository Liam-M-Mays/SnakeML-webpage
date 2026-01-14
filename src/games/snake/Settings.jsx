import React, { useState } from 'react'

/**
 * Snake-specific game settings.
 * These settings are unique to snake (grid size, colors, inputs, rewards, etc.)
 */
export default function SnakeSettings({ settings, onChange, disabled }) {
  const [activeTab, setActiveTab] = useState('board')

  const updateSetting = (key, value) => {
    onChange({ ...settings, [key]: value })
  }

  const updateColor = (colorKey, value) => {
    onChange({
      ...settings,
      colors: { ...settings.colors, [colorKey]: value }
    })
  }

  const updateInput = (inputKey, value) => {
    onChange({
      ...settings,
      inputs: { ...settings.inputs, [inputKey]: value }
    })
  }

  const updateReward = (rewardKey, value) => {
    onChange({
      ...settings,
      rewards: { ...settings.rewards, [rewardKey]: value }
    })
  }

  const updateDebug = (debugKey, value) => {
    onChange({
      ...settings,
      debug: { ...settings.debug, [debugKey]: value }
    })
  }

  // Get current input/reward/debug values with defaults
  const inputs = settings.inputs || {}
  const rewards = settings.rewards || {}
  const debug = settings.debug || {}

  return (
    <div className="game-settings-content">
      {/* Tabs */}
      <div className="settings-tabs">
        <button
          className={`settings-tab ${activeTab === 'board' ? 'active' : ''}`}
          onClick={() => setActiveTab('board')}
        >
          Board
        </button>
        <button
          className={`settings-tab ${activeTab === 'inputs' ? 'active' : ''}`}
          onClick={() => setActiveTab('inputs')}
        >
          Inputs
        </button>
        <button
          className={`settings-tab ${activeTab === 'rewards' ? 'active' : ''}`}
          onClick={() => setActiveTab('rewards')}
        >
          Rewards
        </button>
        <button
          className={`settings-tab ${activeTab === 'colors' ? 'active' : ''}`}
          onClick={() => setActiveTab('colors')}
        >
          Colors
        </button>
      </div>

      {/* Tab Content */}
      <div className="settings-tab-content">
        {activeTab === 'board' && (
          <div className="settings-group">
            <div className="setting-item">
              <label>Grid Size: {settings.gridSize}x{settings.gridSize}</label>
              <input
                type="range"
                min="5"
                max="25"
                disabled={disabled}
                value={settings.gridSize}
                onChange={(e) => updateSetting('gridSize', parseInt(e.target.value))}
              />
            </div>
            <div className="setting-item">
              <label>Board Size: {settings.boardSize}px</label>
              <input
                type="range"
                min="300"
                max="600"
                value={settings.boardSize}
                onChange={(e) => updateSetting('boardSize', parseInt(e.target.value))}
              />
            </div>
            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.randomStartState || false}
                  onChange={(e) => updateSetting('randomStartState', e.target.checked)}
                />
                Random Start State
              </label>
              <span className="setting-hint">
                Start with random length & direction. High score disabled.
              </span>
            </div>
            {settings.randomStartState && (
              <div className="setting-item">
                <label>Max Random Length: {settings.randomMaxLength || (settings.gridSize * settings.gridSize - 1)}</label>
                <input
                  type="range"
                  min="1"
                  max={settings.gridSize * settings.gridSize - 1}
                  value={settings.randomMaxLength || (settings.gridSize * settings.gridSize - 1)}
                  onChange={(e) => updateSetting('randomMaxLength', parseInt(e.target.value))}
                />
                <span className="setting-hint">
                  1 to {settings.gridSize * settings.gridSize - 1} (theoretical max)
                </span>
              </div>
            )}

            {/* Debug Visualization Section */}
            <div className="settings-divider" style={{ borderTop: '1px solid #444', margin: '15px 0', paddingTop: '10px' }}>
              <span className="setting-hint" style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', color: '#888' }}>
                Debug Visualization
              </span>
              <div className="setting-item toggle-item">
                <label>
                  <input
                    type="checkbox"
                    checked={debug.vision || false}
                    onChange={(e) => updateDebug('vision', e.target.checked)}
                  />
                  Show Danger Detection
                </label>
                <span className="setting-hint">
                  Green = safe, Red = danger (L/R/F labels)
                </span>
              </div>
              <div className="setting-item toggle-item">
                <label>
                  <input
                    type="checkbox"
                    checked={debug.path || false}
                    onChange={(e) => updateDebug('path', e.target.checked)}
                  />
                  Show Shortest Path
                </label>
                <span className="setting-hint">
                  BFS path to food (cyan dots)
                </span>
              </div>
              <div className="setting-item toggle-item">
                <label>
                  <input
                    type="checkbox"
                    checked={debug.segments || false}
                    onChange={(e) => updateDebug('segments', e.target.checked)}
                  />
                  Show Segment Tracking
                </label>
                <span className="setting-hint">
                  Gray markers with lines showing chained segments
                </span>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'inputs' && (
          <div className="settings-group">
            <span className="setting-hint" style={{ marginBottom: '10px', display: 'block' }}>
              Configure which features are sent to the AI network.
            </span>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.foodDirection !== false}
                  onChange={(e) => updateInput('foodDirection', e.target.checked)}
                  disabled={disabled}
                />
                Food Direction
              </label>
              <span className="setting-hint">Direction to food (-1, 0, 1)</span>
            </div>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.pathDistance !== false}
                  onChange={(e) => updateInput('pathDistance', e.target.checked)}
                  disabled={disabled}
                />
                Path Distance (BFS)
              </label>
              <span className="setting-hint">Actual shortest path length</span>
            </div>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.currentDirection !== false}
                  onChange={(e) => updateInput('currentDirection', e.target.checked)}
                  disabled={disabled}
                />
                Current Direction
              </label>
              <span className="setting-hint">Snake's movement direction</span>
            </div>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.hunger !== false}
                  onChange={(e) => updateInput('hunger', e.target.checked)}
                  disabled={disabled}
                />
                Hunger Level
              </label>
              <span className="setting-hint">Steps since last food</span>
            </div>

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.danger !== false}
                  onChange={(e) => updateInput('danger', e.target.checked)}
                  disabled={disabled}
                />
                Danger Detection
              </label>
              <span className="setting-hint">Walls and body nearby</span>
            </div>

            {/* Vision range - only shown when danger is enabled */}
            {inputs.danger !== false && (
              <div className="setting-item sub-setting">
                <label>Vision Range: {inputs.visionRange ?? 1}</label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  value={inputs.visionRange ?? 1}
                  onChange={(e) => updateInput('visionRange', parseInt(e.target.value))}
                  disabled={disabled}
                />
                <span className="setting-hint">0=adjacent only (4 values), 1-3=window size</span>
              </div>
            )}

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.segments !== false}
                  onChange={(e) => updateInput('segments', e.target.checked)}
                  disabled={disabled}
                />
                Body Segments
              </label>
              <span className="setting-hint">Relative positions of body parts</span>
            </div>

            {/* Segment count - only shown when segments is enabled */}
            {inputs.segments !== false && (
              <div className="setting-item sub-setting">
                <label>Segment Count: {inputs.segmentCount ?? 10}</label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={inputs.segmentCount ?? 10}
                  onChange={(e) => updateInput('segmentCount', parseInt(e.target.value))}
                  disabled={disabled}
                />
                <span className="setting-hint">Number of body positions tracked</span>
              </div>
            )}

            <div className="setting-item toggle-item">
              <label>
                <input
                  type="checkbox"
                  checked={inputs.snakeLength === true}
                  onChange={(e) => updateInput('snakeLength', e.target.checked)}
                  disabled={disabled}
                />
                Snake Length
              </label>
              <span className="setting-hint">Normalized length (length / max possible)</span>
            </div>
          </div>
        )}

        {activeTab === 'rewards' && (
          <div className="settings-group">
            <span className="setting-hint" style={{ marginBottom: '10px', display: 'block' }}>
              Configure reward values for AI training.
            </span>

            <div className="setting-item reward-input">
              <label>Apple (eat food)</label>
              <input
                type="number"
                step="0.1"
                value={rewards.apple ?? 1.0}
                onChange={(e) => updateReward('apple', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Wall Death</label>
              <input
                type="number"
                step="0.1"
                value={rewards.wall ?? -1.0}
                onChange={(e) => updateReward('wall', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Self Collision</label>
              <input
                type="number"
                step="0.1"
                value={rewards.self ?? -1.0}
                onChange={(e) => updateReward('self', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Starvation</label>
              <input
                type="number"
                step="0.1"
                value={rewards.starve ?? -1.0}
                onChange={(e) => updateReward('starve', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Step Penalty</label>
              <input
                type="number"
                step="0.001"
                value={rewards.step ?? 0}
                onChange={(e) => updateReward('step', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Closer to Food</label>
              <input
                type="number"
                step="0.01"
                value={rewards.closer ?? 0}
                onChange={(e) => updateReward('closer', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>

            <div className="setting-item reward-input">
              <label>Farther from Food</label>
              <input
                type="number"
                step="0.01"
                value={rewards.farther ?? 0}
                onChange={(e) => updateReward('farther', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                className="reward-text-input"
              />
            </div>
          </div>
        )}

        {activeTab === 'colors' && (
          <div className="settings-group">
            <div className="setting-item">
              <label>Snake Color</label>
              <input
                type="color"
                value={settings.colors.snake}
                onChange={(e) => updateColor('snake', e.target.value)}
                className="color-picker"
              />
            </div>
            <div className="setting-item">
              <label>Food Color</label>
              <input
                type="color"
                value={settings.colors.food}
                onChange={(e) => updateColor('food', e.target.value)}
                className="color-picker"
              />
            </div>
            <div className="setting-item">
              <label>Background</label>
              <input
                type="color"
                value={settings.colors.background}
                onChange={(e) => updateColor('background', e.target.value)}
                className="color-picker"
              />
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
