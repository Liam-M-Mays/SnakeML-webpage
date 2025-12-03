/**
 * Reward Designer component for configuring reward functions.
 * Supports both simple numeric rewards and formula-based rewards.
 */

import React, { useState } from "react";

const REWARD_EVENTS = [
  { key: "apple", label: "Eat Apple", default: 1.0 },
  { key: "death_wall", label: "Hit Wall", default: -1.0 },
  { key: "death_self", label: "Hit Self", default: -1.0 },
  { key: "death_starv", label: "Starvation", default: -0.5 },
  { key: "step", label: "Each Step", default: -0.001 },
];

const FORMULA_VARIABLES = [
  { name: "snake_length", desc: "Current length of the snake" },
  { name: "hunger", desc: "Steps since last food" },
  { name: "score", desc: "Current game score" },
  { name: "distance_to_food", desc: "Manhattan distance to food" },
  { name: "grid_size", desc: "Size of the game grid" },
];

function RewardDesigner({ rewardConfig, onChange, disabled }) {
  const [mode, setMode] = useState("simple"); // simple or advanced
  const [formulas, setFormulas] = useState({});

  const handleRewardChange = (key, value) => {
    if (onChange) {
      onChange({
        ...rewardConfig,
        [key]: parseFloat(value) || 0,
      });
    }
  };

  const handleFormulaChange = (key, formula) => {
    setFormulas((prev) => ({ ...prev, [key]: formula }));
  };

  const applyFormula = (key) => {
    const formula = formulas[key];
    if (formula && onChange) {
      // For now, just update the reward config
      // Full formula support would require backend evaluation
      onChange({
        ...rewardConfig,
        formulas: [
          ...(rewardConfig.formulas || []).filter((f) => f.event !== key),
          { event: key, formula, enabled: true },
        ],
      });
    }
  };

  return (
    <div className="reward-designer">
      <div className="reward-mode-toggle">
        <button
          className={`mode-btn ${mode === "simple" ? "active" : ""}`}
          onClick={() => setMode("simple")}
        >
          Simple
        </button>
        <button
          className={`mode-btn ${mode === "advanced" ? "active" : ""}`}
          onClick={() => setMode("advanced")}
        >
          Advanced
        </button>
      </div>

      {mode === "simple" ? (
        <div className="reward-simple">
          <div className="reward-grid">
            {REWARD_EVENTS.map((event) => (
              <div key={event.key} className="reward-item">
                <label>{event.label}</label>
                <input
                  type="number"
                  step="0.1"
                  value={rewardConfig?.[event.key] ?? event.default}
                  onChange={(e) => handleRewardChange(event.key, e.target.value)}
                  disabled={disabled}
                />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="reward-advanced">
          <div className="formula-help">
            <h4>Available Variables</h4>
            <div className="variable-list">
              {FORMULA_VARIABLES.map((v) => (
                <div key={v.name} className="variable-item">
                  <code>{v.name}</code>
                  <span>{v.desc}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="formula-grid">
            {REWARD_EVENTS.map((event) => (
              <div key={event.key} className="formula-item">
                <label>{event.label}</label>
                <div className="formula-input-group">
                  <input
                    type="text"
                    placeholder={`e.g., ${event.default} + 0.1 * snake_length`}
                    value={formulas[event.key] || ""}
                    onChange={(e) => handleFormulaChange(event.key, e.target.value)}
                    disabled={disabled}
                  />
                  <button
                    className="btn btn-small"
                    onClick={() => applyFormula(event.key)}
                    disabled={disabled || !formulas[event.key]}
                  >
                    Apply
                  </button>
                </div>
                <div className="formula-current">
                  Current: {rewardConfig?.[event.key] ?? event.default}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default RewardDesigner;
