/**
 * Visual Network Builder component.
 * Allows adding/removing/reordering layers with units and activations.
 */

import React, { useState } from "react";

const ACTIVATIONS = [
  { value: "relu", label: "ReLU" },
  { value: "leaky_relu", label: "Leaky ReLU" },
  { value: "tanh", label: "Tanh" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "linear", label: "Linear" },
];

const PRESETS = {
  small: {
    name: "Small",
    layers: [
      { type: "dense", units: 64, activation: "relu" },
      { type: "dense", units: 64, activation: "relu" },
    ],
  },
  medium: {
    name: "Medium",
    layers: [
      { type: "dense", units: 128, activation: "relu" },
      { type: "dense", units: 256, activation: "relu" },
      { type: "dense", units: 256, activation: "relu" },
    ],
  },
  large: {
    name: "Large",
    layers: [
      { type: "dense", units: 256, activation: "relu" },
      { type: "dense", units: 512, activation: "relu" },
      { type: "dense", units: 512, activation: "relu" },
      { type: "dense", units: 256, activation: "relu" },
    ],
  },
};

function NetworkBuilder({ networkConfig, onChange, disabled }) {
  const layers = networkConfig?.layers || PRESETS.medium.layers;

  const updateLayers = (newLayers) => {
    if (onChange) {
      onChange({ ...networkConfig, layers: newLayers });
    }
  };

  const addLayer = () => {
    updateLayers([...layers, { type: "dense", units: 128, activation: "relu" }]);
  };

  const removeLayer = (index) => {
    if (layers.length > 1) {
      updateLayers(layers.filter((_, i) => i !== index));
    }
  };

  const updateLayer = (index, field, value) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], [field]: value };
    updateLayers(newLayers);
  };

  const moveLayer = (index, direction) => {
    if (
      (direction === -1 && index === 0) ||
      (direction === 1 && index === layers.length - 1)
    ) {
      return;
    }
    const newLayers = [...layers];
    const temp = newLayers[index];
    newLayers[index] = newLayers[index + direction];
    newLayers[index + direction] = temp;
    updateLayers(newLayers);
  };

  const applyPreset = (presetKey) => {
    updateLayers([...PRESETS[presetKey].layers]);
  };

  // Calculate total parameters (rough estimate)
  const calculateParams = () => {
    let total = 0;
    let prevSize = 15; // Approximate input size

    for (const layer of layers) {
      total += prevSize * layer.units + layer.units; // weights + bias
      prevSize = layer.units;
    }
    total += prevSize * 3 + 3; // Output layer (3 actions)
    return total;
  };

  return (
    <div className="network-builder">
      <div className="network-presets">
        <span className="presets-label">Presets:</span>
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button
            key={key}
            className="preset-btn"
            onClick={() => applyPreset(key)}
            disabled={disabled}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <div className="network-layers">
        <div className="layers-header">
          <span className="layer-col">Layer</span>
          <span className="units-col">Units</span>
          <span className="activation-col">Activation</span>
          <span className="actions-col"></span>
        </div>

        {layers.map((layer, index) => (
          <div key={index} className="layer-row">
            <span className="layer-col">
              <span className="layer-number">{index + 1}</span>
              <span className="layer-type">Dense</span>
            </span>

            <input
              type="number"
              className="units-input"
              value={layer.units}
              onChange={(e) =>
                updateLayer(index, "units", parseInt(e.target.value) || 64)
              }
              disabled={disabled}
              min="16"
              max="1024"
              step="16"
            />

            <select
              className="activation-select"
              value={layer.activation}
              onChange={(e) => updateLayer(index, "activation", e.target.value)}
              disabled={disabled}
            >
              {ACTIVATIONS.map((act) => (
                <option key={act.value} value={act.value}>
                  {act.label}
                </option>
              ))}
            </select>

            <div className="layer-actions">
              <button
                className="layer-btn"
                onClick={() => moveLayer(index, -1)}
                disabled={disabled || index === 0}
                title="Move up"
              >
                ↑
              </button>
              <button
                className="layer-btn"
                onClick={() => moveLayer(index, 1)}
                disabled={disabled || index === layers.length - 1}
                title="Move down"
              >
                ↓
              </button>
              <button
                className="layer-btn remove"
                onClick={() => removeLayer(index)}
                disabled={disabled || layers.length <= 1}
                title="Remove layer"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="network-footer">
        <button
          className="btn btn-small"
          onClick={addLayer}
          disabled={disabled || layers.length >= 8}
        >
          + Add Layer
        </button>

        <div className="network-stats">
          <span className="stat-item">
            Layers: <strong>{layers.length}</strong>
          </span>
          <span className="stat-item">
            Parameters: <strong>~{calculateParams().toLocaleString()}</strong>
          </span>
        </div>
      </div>
    </div>
  );
}

export default NetworkBuilder;
