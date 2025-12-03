/**
 * Observation/Input Configuration component.
 * Allows configuring vision modes and segment tracking.
 */

import React from "react";

const VISION_MODES = [
  { value: 0, label: "Immediate Danger", desc: "4 danger flags (up/down/left/right)" },
  { value: 1, label: "3×3 Vision", desc: "8 cells around head" },
  { value: 2, label: "5×5 Vision", desc: "24 cells around head" },
  { value: -1, label: "Full Grid", desc: "Entire grid as observation" },
];

function ObservationConfig({ envConfig, onChange, disabled }) {
  const vision = envConfig?.vision ?? 0;
  const segSize = envConfig?.seg_size ?? 3;
  const gridSize = envConfig?.grid_size ?? 10;

  // Calculate observation size
  const calculateObsSize = () => {
    const baseSize = 5; // apple_x, apple_y, dir_x, dir_y, hunger
    const segmentSize = segSize * 2;

    let dangerSize;
    if (vision === 0) {
      dangerSize = 4;
    } else if (vision > 0) {
      dangerSize = Math.pow(2 * vision + 1, 2) - 1;
    } else {
      dangerSize = gridSize * gridSize;
    }

    return baseSize + segmentSize + dangerSize;
  };

  const handleVisionChange = (newVision) => {
    if (onChange) {
      onChange({
        ...envConfig,
        vision: parseInt(newVision),
      });
    }
  };

  const handleSegSizeChange = (newSegSize) => {
    if (onChange) {
      onChange({
        ...envConfig,
        seg_size: parseInt(newSegSize),
      });
    }
  };

  return (
    <div className="observation-config">
      <div className="config-group">
        <label>Vision Mode</label>
        <div className="vision-options">
          {VISION_MODES.map((mode) => (
            <button
              key={mode.value}
              className={`vision-option ${vision === mode.value ? "active" : ""}`}
              onClick={() => handleVisionChange(mode.value)}
              disabled={disabled}
            >
              <span className="vision-label">{mode.label}</span>
              <span className="vision-desc">{mode.desc}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="config-group">
        <label>Body Segment Tracking: {segSize}</label>
        <input
          type="range"
          min="1"
          max="5"
          value={segSize}
          onChange={(e) => handleSegSizeChange(e.target.value)}
          disabled={disabled}
        />
        <div className="slider-labels">
          <span>1</span>
          <span>5</span>
        </div>
      </div>

      <div className="obs-size-display">
        <span className="obs-label">Observation Size:</span>
        <span className="obs-value">{calculateObsSize()}</span>
        <span className="obs-units">inputs</span>
      </div>

      <div className="obs-breakdown">
        <div className="breakdown-item">
          <span>Base features:</span>
          <span>5</span>
        </div>
        <div className="breakdown-item">
          <span>Body segments:</span>
          <span>{segSize * 2}</span>
        </div>
        <div className="breakdown-item">
          <span>Danger/vision:</span>
          <span>
            {vision === 0
              ? 4
              : vision > 0
              ? Math.pow(2 * vision + 1, 2) - 1
              : gridSize * gridSize}
          </span>
        </div>
      </div>
    </div>
  );
}

export default ObservationConfig;
