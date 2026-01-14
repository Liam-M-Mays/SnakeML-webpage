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
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma (Discount)', value: 0.9 },
    lr: { min: 0.00001, max: 0.1, step: 0.0001, label: 'Learning Rate', value: 0.001 },
    decay: { min: 0.9, max: 1, step: 0.0001, label: 'Epsilon Decay', value: 0.999 },
    eps_start: { min: 0, max: 1, step: 0.1, label: 'Epsilon Start', value: 1.0 },
    eps_end: { min: 0, max: 1, step: 0.01, label: 'Epsilon End', value: 0.1 },
    target_update: { min: 1, step: 10, label: 'Target Update (eps)', value: 50 },
  },
  ppo: {
    buffer: { min: 100, step: 100, label: 'Buffer Size', value: 1000 },
    batch: { min: 16, step: 16, label: 'Batch Size', value: 128 },
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma (Discount)', value: 0.99 },
    lr: { min: 0.00001, max: 0.1, step: 0.0001, label: 'Learning Rate', value: 0.0002 },
    epoch: { min: 1, step: 1, label: 'PPO Epochs', value: 8 },
    clip: { min: 0.05, max: 0.5, step: 0.05, label: 'Clip Epsilon', value: 0.15 },
    ent_start: { min: 0, max: 0.5, step: 0.01, label: 'Entropy Start', value: 0.05 },
    ent_end: { min: 0, max: 0.5, step: 0.01, label: 'Entropy End', value: 0.01 },
    ent_decay: { min: 100, step: 100, label: 'Entropy Decay Steps', value: 1000 },
    vf_coef: { min: 0.1, max: 2, step: 0.1, label: 'Value Loss Coef', value: 0.5 },
  },
  mann: {
    // Basic MANN - simpler policy gradient (no PPO)
    batch: { min: 16, step: 16, label: 'Batch Size', value: 32 },
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma (Discount)', value: 0.99 },
    lr: { min: 0.00001, max: 0.1, step: 0.0001, label: 'Learning Rate', value: 0.001 },
    entropy: { min: 0, max: 0.5, step: 0.01, label: 'Entropy Coef', value: 0.01 },
    experts: { min: 2, max: 8, step: 1, label: 'Num Experts', value: 4 },
  },
  mapo: {
    // MAPO - PPO-based mixture of experts
    buffer: { min: 100, step: 100, label: 'Buffer Size', value: 2000 },
    batch: { min: 16, step: 16, label: 'Batch Size', value: 64 },
    gamma: { min: 0, max: 1, step: 0.01, label: 'Gamma (Discount)', value: 0.99 },
    lr: { min: 0.00001, max: 0.1, step: 0.0001, label: 'Learning Rate', value: 0.0003 },
    epoch: { min: 1, step: 1, label: 'Training Epochs', value: 10 },
    clip: { min: 0.05, max: 0.5, step: 0.05, label: 'Clip Epsilon', value: 0.15 },
    ent_start: { min: 0, max: 0.5, step: 0.01, label: 'Entropy Start', value: 0.05 },
    ent_end: { min: 0, max: 0.5, step: 0.01, label: 'Entropy End', value: 0.01 },
    ent_decay: { min: 100, step: 100, label: 'Entropy Decay Steps', value: 2000 },
    experts: { min: 2, max: 8, step: 1, label: 'Num Experts', value: 4 },
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
  socket,
  selectedDevice: parentSelectedDevice,
  onDeviceChange,
}) {
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newAgentName, setNewAgentName] = useState('')
  const [newAgentType, setNewAgentType] = useState('dqn')
  const [saveStatus, setSaveStatus] = useState(null)
  const [nameError, setNameError] = useState(null)

  // Store raw input values as strings (allows free typing)
  const [inputValues, setInputValues] = useState({})
  // Validation errors shown only after clicking Train
  const [validationErrors, setValidationErrors] = useState({})

  // Device management
  const [availableDevices, setAvailableDevices] = useState(['cpu'])
  const [currentDevice, setCurrentDevice] = useState('cpu')
  // Use parent's selected device if provided, otherwise local state
  const [localSelectedDevice, setLocalSelectedDevice] = useState('cpu')
  const selectedDevice = parentSelectedDevice ?? localSelectedDevice

  const selectedAgent = agents.find(a => a.id === selectedAgentId)

  // Initialize input values when agent changes
  useEffect(() => {
    if (selectedAgent) {
      const values = {}
      Object.entries(selectedAgent.params).forEach(([key, config]) => {
        values[key] = String(config.value)
      })
      setInputValues(values)
      setValidationErrors({}) // Clear errors when switching agents
    }
  }, [selectedAgentId])

  // Load models list on mount and when game changes
  useEffect(() => {
    if (onRefreshModels) {
      onRefreshModels()
    }
  }, [selectedGame])

  // Fetch available devices on mount
  useEffect(() => {
    if (!socket) return

    const handleDevicesList = (data) => {
      const devices = data.available || ['cpu']
      setAvailableDevices(devices)
      if (data.current) {
        setCurrentDevice(data.current)
        setLocalSelectedDevice(data.current)
        if (onDeviceChange) onDeviceChange(data.current)
      } else {
        // Default to first available device (prefer cpu for stability)
        const defaultDevice = devices.includes('cpu') ? 'cpu' : devices[0]
        setLocalSelectedDevice(defaultDevice)
        if (onDeviceChange) onDeviceChange(defaultDevice)
      }
    }

    const handleDeviceChanged = (data) => {
      if (data.success) {
        setCurrentDevice(data.device)
        setLocalSelectedDevice(data.device)
        if (onDeviceChange) onDeviceChange(data.device)
      } else {
        console.error('Device switch failed:', data.error)
        // Reset dropdown to current device
        setLocalSelectedDevice(currentDevice)
        if (onDeviceChange) onDeviceChange(currentDevice)
      }
    }

    socket.on('devices_list', handleDevicesList)
    socket.on('device_changed', handleDeviceChanged)

    // Request device list
    socket.emit('get_devices')

    return () => {
      socket.off('devices_list', handleDevicesList)
      socket.off('device_changed', handleDeviceChanged)
    }
  }, [socket, onDeviceChange])

  // Handle device change (during training, sends set_device; before training, just updates selection)
  const handleDeviceChange = (device) => {
    setLocalSelectedDevice(device)
    if (onDeviceChange) onDeviceChange(device)
    // If training is active, switch device live
    if (isTraining && socket) {
      socket.emit('set_device', { device })
    }
  }

  const handleNameChange = (e) => {
    const value = e.target.value
    setNewAgentName(value)
    // Clear error while typing - only validate on submit
    if (nameError) setNameError(null)
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

  // Allow free typing - just update the string value
  const handleInputChange = (paramKey, value) => {
    setInputValues(prev => ({ ...prev, [paramKey]: value }))
    // Clear error for this field while typing
    if (validationErrors[paramKey]) {
      setValidationErrors(prev => {
        const next = { ...prev }
        delete next[paramKey]
        return next
      })
    }
  }

  // Validate all params and start training
  const handleTrainClick = () => {
    if (!selectedAgent) return

    const errors = {}
    const validValues = {}

    // Validate each parameter
    Object.entries(selectedAgent.params).forEach(([key, config]) => {
      const rawValue = inputValues[key]
      const numValue = parseFloat(rawValue)

      if (rawValue === '' || rawValue === undefined) {
        errors[key] = `${config.label} is required`
        return
      }

      if (isNaN(numValue)) {
        errors[key] = `${config.label} must be a number`
        return
      }

      const validation = validateHyperparameter(numValue, config)
      if (!validation.valid) {
        errors[key] = validation.error
      } else {
        validValues[key] = validation.correctedValue
      }
    })

    // If there are errors, show them and don't train
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors)
      return
    }

    // Update params with validated values
    Object.entries(validValues).forEach(([key, value]) => {
      onUpdateParams(selectedAgentId, key, value)
    })

    // Clear any previous errors and start training
    setValidationErrors({})
    onTrain()
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
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
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
            <option value="mann">MANN</option>
            <option value="mapo">MAPO</option>
          </select>
          <div className="create-agent-actions">
            <button onClick={handleCreate} disabled={!newAgentName.trim()}>
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
                type="text"
                value={inputValues[key] ?? String(config.value)}
                onChange={(e) => handleInputChange(key, e.target.value)}
                disabled={isTraining || selectedAgent.modelFilename}
                className={validationErrors[key] ? 'input-error' : ''}
                placeholder={String(config.value)}
              />
              {validationErrors[key] && <span className="error-message small">{validationErrors[key]}</span>}
            </div>
          ))}

          {/* Device Selection */}
          <div className="param-item device-select">
            <label>Compute Device</label>
            <select
              value={selectedDevice}
              onChange={(e) => handleDeviceChange(e.target.value)}
            >
              {availableDevices.map(device => (
                <option key={device} value={device}>
                  {device.toUpperCase()}
                </option>
              ))}
            </select>
            {currentDevice !== selectedDevice && (
              <span className="device-switching">Switching...</span>
            )}
          </div>
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
          onClick={isTraining ? onStopTraining : handleTrainClick}
          disabled={disabled && !isTraining}
        >
          {isTraining ? 'Stop Training' : 'Train'}
        </button>
      )}
    </div>
  )
}
