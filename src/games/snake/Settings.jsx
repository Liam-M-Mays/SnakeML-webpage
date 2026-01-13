import React, { useState } from 'react'

/**
 * Snake-specific game settings.
 * These settings are unique to snake (grid size, colors, etc.)
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
