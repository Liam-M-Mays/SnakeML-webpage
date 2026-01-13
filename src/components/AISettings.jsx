import React, { useState, useEffect } from 'react'
import { validateAgentName, validateHyperparameter } from '../utils/validation'

/**
 * AI Settings panel.
 * Handles agent creation, selection, parameter tuning, and model save/load.
 */

// Default parameters for each network type
const DEFAULT_PARAMS = {
  dqn: {
    buffer: { min: 100, step: 100, label: 'Buffer Size', value: 10000 },
    batch: { min: 16, step: 16, label: 'Batch Size', value: 128 },
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma', value: 0.9 },
    decay: { min: 0.9, max: 1, step: 0.0001, label: 'Epsilon Decay', value: 0.999 },
  },
  ppo: {
    buffer: { min: 100, step: 100, label: 'Buffer Size', value: 1000 },
    batch: { min: 16, step: 16, label: 'Batch Size', value: 128 },
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma', value: 0.99 },
    decay: { min: 100, step: 100, label: 'Entropy Decay Steps', value: 1000 },
    epoch: { min: 1, step: 1, label: 'PPO Epochs', value: 8 },
  },
}

export default function AISettings({
  agents,
  selectedAgentId,
  onCreateAgent,
  onSelectAgent,
  onUpdateParams,
  onTrain,
  onStopTraining,
  isTraining,
  disabled,
  savedModels,
  onSaveModel,
  onDeleteModel,
  onRefreshModels,
  selectedGame,
}) {
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newAgentName, setNewAgentName] = useState('')
  const [newAgentType, setNewAgentType] = useState('dqn')
  const [saveStatus, setSaveStatus] = useState(null)
  const [nameError, setNameError] = useState(null)
  const [paramErrors, setParamErrors] = useState({})

  const selectedAgent = agents.find(a => a.id === selectedAgentId)

  // Load models list on mount and when game changes
  useEffect(() => {
    if (onRefreshModels) {
      onRefreshModels()
    }
  }, [selectedGame])

  const handleNameChange = (e) => {
    const value = e.target.value
    setNewAgentName(value)

    // Validate on change
    if (value.trim()) {
      const validation = validateAgentName(value)
      setNameError(validation.error)
    } else {
      setNameError(null)
    }
  }

  const handleCreate = () => {
    const validation = validateAgentName(newAgentName)
    if (!validation.valid) {
      setNameError(validation.error)
      return
    }

    const params = {}
    Object.entries(DEFAULT_PARAMS[newAgentType]).forEach(([key, config]) => {
      params[key] = { ...config }
    })

    onCreateAgent({
      id: Date.now().toString(),
      name: newAgentName.trim(),
      type: newAgentType,
      params,
    })

    setNewAgentName('')
    setNameError(null)
    setShowCreateForm(false)
  }

  const handleParamChange = (paramKey, value) => {
    if (!selectedAgent) return

    const config = selectedAgent.params[paramKey]
    const validation = validateHyperparameter(value, config)

    if (!validation.valid) {
      setParamErrors(prev => ({ ...prev, [paramKey]: validation.error }))
      // Still update with corrected value
      onUpdateParams(selectedAgentId, paramKey, validation.correctedValue)
    } else {
      setParamErrors(prev => {
        const next = { ...prev }
        delete next[paramKey]
        return next
      })
      onUpdateParams(selectedAgentId, paramKey, validation.correctedValue)
    }
  }

  const handleSave = () => {
    if (!selectedAgent || !onSaveModel) return
    setSaveStatus('saving')
    onSaveModel(selectedAgent.name, selectedAgent.id, selectedAgent.name, selectedGame)
    setTimeout(() => setSaveStatus(null), 2000)
  }

  const handleLoad = (model) => {
    // Create an agent from the saved model
    const params = {}
    Object.entries(DEFAULT_PARAMS[model.network_type]).forEach(([key, config]) => {
      params[key] = { ...config }
    })

    const agent = {
      id: `saved_${model.filename}`,
      name: model.name,
      type: model.network_type,
      params,
      modelFilename: model.filename,  // Mark this agent as from a saved model
    }

    onCreateAgent(agent)
  }

  const handleDelete = (e, model) => {
    e.stopPropagation()
    if (!onDeleteModel) return
    if (confirm(`Delete model "${model.name}"?`)) {
      onDeleteModel(model.filename)
    }
  }

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  // Filter models by game and optionally by agent type
  const filteredModels = savedModels?.filter(m => {
    // Must match game
    if (m.game && m.game !== selectedGame) return false
    // If agent selected, must match network type
    if (selectedAgent && m.network_type !== selectedAgent.type) return false
    return true
  }) || []

  return (
    <div className="ai-settings">
      {/* Create Agent Button / Form */}
      {!showCreateForm ? (
        <button
          className="create-agent-btn"
          onClick={() => setShowCreateForm(true)}
          disabled={disabled || isTraining}
        >
          + New Agent
        </button>
      ) : (
        <div className="create-agent-form">
          <input
            type="text"
            placeholder="Agent name..."
            value={newAgentName}
            onChange={handleNameChange}
            onKeyDown={(e) => e.key === 'Enter' && !nameError && handleCreate()}
            autoFocus
            className={nameError ? 'input-error' : ''}
          />
          {nameError && <span className="error-message">{nameError}</span>}
          <select
            value={newAgentType}
            onChange={(e) => setNewAgentType(e.target.value)}
          >
            <option value="dqn">DQN</option>
            <option value="ppo">PPO</option>
          </select>
          <div className="create-agent-actions">
            <button onClick={handleCreate} disabled={!newAgentName.trim() || nameError}>
              Create
            </button>
            <button onClick={() => { setShowCreateForm(false); setNameError(null) }} className="cancel-btn">
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Agent List */}
      {agents.length > 0 && (
        <div className="agent-list">
          <h4>Agents</h4>
          {agents.map(agent => (
            <div
              key={agent.id}
              className={`agent-item ${agent.id === selectedAgentId ? 'selected' : ''}`}
              onClick={() => !isTraining && onSelectAgent(agent.id)}
            >
              <span className="agent-name">{agent.name}</span>
              <span className="agent-type">{agent.type.toUpperCase()}</span>
            </div>
          ))}
        </div>
      )}

      {/* Selected Agent Parameters */}
      {selectedAgent && (
        <div className="agent-params">
          <h4>Parameters {selectedAgent.modelFilename && <span className="params-locked">(Locked)</span>}</h4>
          {Object.entries(selectedAgent.params).map(([key, config]) => (
            <div key={key} className="param-item">
              <label>{config.label}</label>
              <input
                type="number"
                min={config.min}
                max={config.max}
                step={config.step}
                value={config.value}
                onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
                disabled={isTraining || selectedAgent.modelFilename}
                className={paramErrors[key] ? 'input-error' : ''}
              />
              {paramErrors[key] && <span className="error-message small">{paramErrors[key]}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Save Button */}
      {selectedAgent && (
        <div className="model-actions">
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={!isTraining && disabled}
            title="Save current model"
          >
            {saveStatus === 'saving' ? 'Saved!' : 'Save Model'}
          </button>
        </div>
      )}

      {/* Saved Models List - always visible if there are models */}
      {filteredModels.length > 0 && (
        <div className="saved-models">
          <h4>Saved Models</h4>
          <div className="models-list">
            {filteredModels.map(model => (
              <div
                key={model.filename}
                className={`model-item ${selectedAgent?.modelFilename === model.filename ? 'selected' : ''}`}
                onClick={() => !isTraining && handleLoad(model)}
                style={{ cursor: isTraining ? 'not-allowed' : 'pointer', opacity: isTraining ? 0.6 : 1 }}
              >
                <div className="model-info">
                  <span className="model-name">
                    {model.name}
                    {selectedAgent?.modelFilename === model.filename && ' ✓'}
                  </span>
                  <span className="model-meta">
                    {model.network_type.toUpperCase()} | {model.episodes} eps | {formatDate(model.timestamp)}
                  </span>
                </div>
                <button
                  className="delete-model-btn"
                  onClick={(e) => handleDelete(e, model)}
                  title="Delete model"
                  disabled={isTraining}
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Train Button */}
      {selectedAgent && (
        <button
          className={`train-btn ${isTraining ? 'stop' : ''}`}
          onClick={isTraining ? onStopTraining : onTrain}
          disabled={disabled && !isTraining}
        >
          {isTraining ? 'Stop Training' : 'Train'}
        </button>
      )}
    </div>
  )
}
